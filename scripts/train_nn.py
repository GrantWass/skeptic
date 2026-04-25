#!/usr/bin/env python3
"""
GRU + MLP-head model: P(resolved_up) at any second of a 5-minute window.

Architecture
------------
Each 5-minute window is treated as a variable-length sequence of seconds.
A 2-layer GRU processes the sequence and emits a hidden state h_t at each
second.  A small MLP head maps h_t → sigmoid probability.

Why GRU over Transformer:
  - At inference we receive one new second at a time and need an instant
    answer; a GRU updates its hidden state in O(1) per step.
  - Transformers require reprocessing the full sequence each second (O(t²)
    attention) unless key-value caching is implemented.
  - For sequences of ≤300 steps with our feature set, GRU empirically
    matches Transformer quality at ~5× lower inference cost.

Features
--------
25 signals — 17 instantaneous + 8 path-dependent quant features that encode
WHERE the price has been in the window, not just where it is now.
The GRU then learns WHEN each feature matters across the sequence.

No Platt calibration — BCELoss directly optimises calibrated probabilities.
Empirically Brier(raw) < Brier(Platt) for every asset in prior runs.

Usage:
    python scripts/train_nn.py
    python scripts/train_nn.py --assets BTC ETH
    python scripts/train_nn.py --out-report data/reports/nn_report.md
"""
import argparse
import gc
import joblib
import logging
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # catches ImportError, OSError, RuntimeError on MPS init, etc.
    _HAS_TORCH = False
    torch = None  # type: ignore[assignment]
    class _NNStub:  # minimal stub so class _GRUModel(nn.Module) parses at import time
        Module = object
        utils  = type("_u", (), {"clip_grad_norm_": staticmethod(lambda *a, **k: None)})()
        BCELoss = object
    nn = _NNStub()  # type: ignore[assignment]

# Import data-loading and feature-engineering helpers from train_model.
from train_model import (  # type: ignore[import]
    WINDOW_SECS,
    MIN_COIN_ROWS,
    EWMA_LAMBDA,
    ASSET_TO_SYMBOL,
    BUY_FEE_RATE,  # deprecated alias
    FEE_GAMMA,
    DEFAULT_THRESHOLD,
    PM_FEATURES,
    QUANT_FEATURES,
    build_asset_dataset,
    build_market_features_dataset,
    _augment_both_sides,
    _pnl_per_trade,
    section_calibration,
    section_threshold_ev,
    section_entry_timing,
    section_slippage,
    section_hour_of_day,
    section_day_of_week,
    section_edge_by_decile,
    section_recent_windows,
    section_orderbook_imbalance,
    _load_asset,
)
from skeptic.models.calibration import PlattScaledModel  # noqa: F401 — joblib compat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%M:%S",
)
log = logging.getLogger(__name__)


# ── Feature set ───────────────────────────────────────────────────────────────
# Trained only on PM windows (where prediction market is active).
# Simplified vs previous version — dropped low-importance features based on
# permutation results: mom_slope, acc_10s, vol_expansion, move_efficiency,
# pv_corr_10, vol_accel.
# Added PM order-flow features: up_imbalance, dn_imbalance.
# Note: up_ask / dn_ask intentionally excluded — the model builds its own
# independent probability estimate; the market price is used only at inference
# to compute edge.  Including it as a training feature would cause the model
# to regress toward the market price rather than develop independent signal.
NN_FEATURES = [
    # Core directional
    "move_sigmas",
    "elapsed_second",
    "move_x_elapsed",
    "hour_sin",
    "hour_cos",
    # Momentum
    "vel_5s",
    "vel_10s",
    "vel_decay",
    # Range
    "dist_low_30",
    "dist_high_30",
    # Volume / participation
    "vol_10s_log",
    "signed_vol_imb",
    # Trend structure
    "trend_str_30",
    "dir_consistency_10",
    # Path features (carry over from previous window via prev_state)
    "vwap_dev",
    "chan_pos",
    "max_up_excursion",
    "max_dn_excursion",
    "dir_consistency_window",
    # PM order-flow (imbalance = buying/selling pressure not yet priced in)
    "up_imbalance",
    "dn_imbalance",
    # Rolling MA deviations — price vs recent average (different from velocity,
    # which compares endpoints; MA deviation captures position within recent range)
    "vel_20s",      # 20s velocity — medium-term momentum timescale
    "ma_dev_10s",   # price deviation from 10s rolling mean
    "ma_dev_30s",   # price deviation from 30s rolling mean
]

# Columns that arrive sparsely (only at PM update ticks); forward-filled within
# each window so the MLP always sees the last known value rather than NaN.
PM_FILL_COLS = ["up_imbalance", "dn_imbalance"]

# ── MLP hyperparameters ───────────────────────────────────────────────────────
MLP_HIDDEN       = (256, 128, 64)  # hidden layer sizes
MLP_DROPOUT      = 0.20
MLP_LR           = 3e-4
MLP_WEIGHT_DECAY = 1e-4
MLP_BATCH_SIZE   = 4096   # flat rows — can use large batches
MLP_MAX_EPOCHS   = 120
MLP_PATIENCE     = 20
MLP_GRAD_CLIP    = 1.0
MLP_VAL_WIN_FRAC = 0.15   # last 15% of train windows for early stopping
# Elapsed weighting: sample weight = 1/(elapsed_second+1).
# Second 1 weighs 1.0×, second 10 weighs 0.1×, second 300 weighs 0.003×.
# Focuses the loss on early-window rows where entry triggers actually fire.


# ── Device detection ──────────────────────────────────────────────────────────

def _get_device() -> "torch.device":
    """MPS (Apple Silicon GPU) → CUDA → CPU."""
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch not installed.  pip install torch")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Network architecture ──────────────────────────────────────────────────────

class _MLPModel(nn.Module):
    """BatchNorm → (Linear → GELU → Dropout) × N → Linear → Sigmoid."""

    def __init__(
        self,
        n_in:    int,
        hidden:  tuple[int, ...] = MLP_HIDDEN,
        dropout: float           = MLP_DROPOUT,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.BatchNorm1d(n_in)]
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """x: [N, n_features] → [N] probabilities."""
        return torch.sigmoid(self.net(x)).squeeze(-1)


# ── Inference wrapper (serializable) ─────────────────────────────────────────

class TorchMLPModel:
    """
    Sklearn-compatible predict_proba around a trained _MLPModel.

    Weights stored as a CPU state-dict for safe joblib serialization.
    Device is chosen lazily on first call (MPS → CUDA → CPU).
    Each row is scored independently — no sequence grouping needed.
    """

    def __init__(
        self,
        imputer:    SimpleImputer,
        scaler:     StandardScaler,
        state_dict: dict,
        n_in:       int,
        features:   list[str],
    ) -> None:
        self._imputer    = imputer
        self._scaler     = scaler
        self._state_dict = {k: v.cpu() for k, v in state_dict.items()}
        self._n_in       = n_in
        self._features   = features
        self._net        = None
        self._device     = None

    def __getstate__(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("_net", "_device")}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._net    = None
        self._device = None

    def _ensure_net(self) -> None:
        if self._net is None:
            self._device = _get_device()
            net = _MLPModel(self._n_in).to(self._device)
            net.load_state_dict(
                {k: v.to(self._device) for k, v in self._state_dict.items()}
            )
            net.eval()
            self._net = net

    def _preprocess(self, X_np: np.ndarray) -> np.ndarray:
        """Impute → scale → float32."""
        return self._scaler.transform(
            self._imputer.transform(X_np.astype(np.float64))
        ).astype(np.float32)

    def predict_proba(self, X) -> np.ndarray:
        """Score all rows independently. Forward-fills PM columns per window."""
        self._ensure_net()

        if isinstance(X, pd.DataFrame):
            feat_cols = [f for f in self._features if f in X.columns]
            fill_cols = [c for c in PM_FILL_COLS if c in X.columns and c in feat_cols]
            if fill_cols and "window_ts" in X.columns:
                X = X.copy()
                X[fill_cols] = X.groupby("window_ts")[fill_cols].transform("ffill")
            X_np = X[feat_cols].to_numpy()
        else:
            X_np = np.asarray(X)

        X_f32 = self._preprocess(X_np)
        with torch.no_grad():
            t     = torch.from_numpy(X_f32).to(self._device)
            probs = self._net(t).cpu().numpy()
        return np.column_stack([1 - probs, probs])


# ── MLP training loop ─────────────────────────────────────────────────────────

def _train_mlp(
    X_tr:   "torch.Tensor",   # [N_tr, F]  preprocessed float32
    y_tr:   "torch.Tensor",   # [N_tr]     binary labels
    w_tr:   "torch.Tensor",   # [N_tr]     elapsed-based sample weights
    X_vl:   "torch.Tensor",
    y_vl:   "torch.Tensor",
    w_vl:   "torch.Tensor",
    n_in:   int,
    device: "torch.device",
) -> tuple[dict, int]:
    """
    Train _MLPModel with AdamW + cosine LR + gradient clipping.
    Elapsed-weighted BCE: each row's loss is multiplied by 1/(elapsed+1),
    so early-window rows (where entries fire) dominate the gradient signal.
    Best weights copied to CPU; GPU tensors freed before return.
    """
    X_tr = X_tr.to(device); y_tr = y_tr.to(device); w_tr = w_tr.to(device)
    X_vl = X_vl.to(device); y_vl = y_vl.to(device); w_vl = w_vl.to(device)

    N_tr    = X_tr.shape[0]
    net     = _MLPModel(n_in).to(device)
    opt     = torch.optim.AdamW(net.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_MAX_EPOCHS)
    crit_nr = nn.BCELoss(reduction="none")
    rng     = torch.Generator(device="cpu")
    rng.manual_seed(42)

    best_loss:  float     = float("inf")
    best_state: dict | None = None
    no_improve: int       = 0
    epoch:      int       = 0

    for epoch in range(MLP_MAX_EPOCHS):
        net.train()
        perm = torch.randperm(N_tr, generator=rng)
        for start in range(0, N_tr, MLP_BATCH_SIZE):
            idx = perm[start: start + MLP_BATCH_SIZE]
            opt.zero_grad(set_to_none=True)
            loss = (crit_nr(net(X_tr[idx]), y_tr[idx]) * w_tr[idx]).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), MLP_GRAD_CLIP)
            opt.step()

        sched.step()

        net.eval()
        with torch.no_grad():
            vl_loss = (crit_nr(net(X_vl), y_vl) * w_vl).mean().item()

        if vl_loss < best_loss - 1e-6:
            best_loss  = vl_loss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            log.info("  epoch %3d  val_bce=%.5f  no_improve=%d",
                     epoch + 1, vl_loss, no_improve)

        if no_improve >= MLP_PATIENCE:
            log.info("  early stop at epoch %d", epoch + 1)
            break

    del net, X_tr, y_tr, w_tr, X_vl, y_vl, w_vl
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return best_state or {}, epoch + 1


# ── Pre-processor fitting ─────────────────────────────────────────────────────

def _fit_preprocessors(
    X_np: np.ndarray,
) -> tuple[SimpleImputer, StandardScaler, np.ndarray]:
    """Fit median imputer + StandardScaler. Returns (imp, sc, X_float32)."""
    imp   = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_np)
    sc    = StandardScaler()
    X_sc  = sc.fit_transform(X_imp).astype(np.float32)
    return imp, sc, X_sc


# ── Asset training ────────────────────────────────────────────────────────────

def train_asset_nn(
    asset:    str,
    df:       pd.DataFrame,
    features: list[str] = NN_FEATURES,
) -> dict | None:
    """
    Train a flat MLP on PM-window per-second data.

    Each row is scored independently — elapsed_second is a feature that
    encodes timing context.  Sample weights = 1/(elapsed+1) focus the loss
    on early-window rows where entry triggers fire.
    """
    if not _HAS_TORCH:
        log.error("PyTorch not installed — cannot train. pip install torch")
        return None

    device = _get_device()
    log.info("%s MLP: device=%s  features=%d", asset, device, len(features))

    train_df = df[df["split"] == "train"]
    test_df  = df[df["split"] == "test"]

    production = len(test_df) == 0
    if len(train_df) < 100 or (not production and len(test_df) < 50):
        log.warning("%s: too few rows — skipping", asset)
        return None

    # ── Slim test_df: metadata + features needed by report sections ────────
    _META    = ["ts", "window_ts", "elapsed_second", "resolved_up"]
    _KEEP    = list(dict.fromkeys(c for c in _META + features if c in test_df.columns))
    test_slim = test_df[_KEEP].copy() if not production else pd.DataFrame()
    n_test_wins = test_df["window_ts"].nunique() if not production else 0
    n_test_rows = len(test_df) if not production else 0

    # ── Temporal val split: last MLP_VAL_WIN_FRAC of training windows ─────
    all_train_wins = sorted(train_df["window_ts"].unique())
    n_val_wins     = max(1, int(len(all_train_wins) * MLP_VAL_WIN_FRAC))
    val_set        = set(all_train_wins[-n_val_wins:])
    fit_set        = set(all_train_wins[:-n_val_wins])

    fit_df = train_df[train_df["window_ts"].isin(fit_set)]
    val_df = train_df[train_df["window_ts"].isin(val_set)]

    n_train_wins = len(all_train_wins)
    n_train_rows = len(train_df)
    baseline_wr  = float(train_df["resolved_up"].mean())

    # Forward-fill PM columns within each window before numpy extraction
    fill_cols = [c for c in PM_FILL_COLS if c in fit_df.columns and c in features]
    if fill_cols:
        fit_df = fit_df.copy()
        fit_df[fill_cols] = fit_df.groupby("window_ts")[fill_cols].transform("ffill")
        val_df = val_df.copy()
        val_df[fill_cols] = val_df.groupby("window_ts")[fill_cols].transform("ffill")

    # Fit preprocessors on fit split only (no data leakage into val/test)
    imp, sc, _ = _fit_preprocessors(fit_df[features].to_numpy(dtype=np.float64))

    def _to_tensors(sub_df: pd.DataFrame):
        X = torch.from_numpy(
            sc.transform(imp.transform(sub_df[features].to_numpy(np.float64))).astype(np.float32)
        )
        y = torch.from_numpy(sub_df["resolved_up"].to_numpy(np.float32))
        e = sub_df["elapsed_second"].to_numpy(np.float32) if "elapsed_second" in sub_df.columns \
            else np.arange(len(sub_df), dtype=np.float32)
        w = torch.from_numpy((1.0 / (e + 1.0)).astype(np.float32))
        return X, y, w

    X_tr, y_tr, w_tr = _to_tensors(fit_df)
    X_vl, y_vl, w_vl = _to_tensors(val_df)
    del fit_df, val_df, train_df, test_df
    gc.collect()

    log.info("%s MLP: training on %s — fit=%d rows, val=%d rows",
             asset, device, len(X_tr), len(X_vl))
    best_state, n_epochs = _train_mlp(X_tr, y_tr, w_tr, X_vl, y_vl, w_vl, len(features), device)
    gc.collect()

    log.info("%s MLP: converged in %d epochs", asset, n_epochs)
    model = TorchMLPModel(imp, sc, best_state, len(features), features)

    if production:
        return {
            "asset":           asset,
            "pipe":            model,
            "features":        features,
            "train_df":        pd.DataFrame(),
            "test_df":         pd.DataFrame(),
            "n_iter":          n_epochs,
            "auc":             None,
            "brier":           None,
            "brier_raw":       None,
            "isotonic_auc":    None,
            "isotonic_brier":  None,
            "baseline_wr":     baseline_wr,
            "n_train":         n_train_rows,
            "n_test":          0,
            "n_windows_train": n_train_wins,
            "n_windows_test":  0,
        }

    probs  = model.predict_proba(test_slim)[:, 1]
    y_test = test_slim["resolved_up"].to_numpy(dtype=np.int32)

    try:
        auc = float(roc_auc_score(y_test, probs))
    except ValueError as e:
        log.warning("%s MLP: roc_auc_score failed: %s", asset, e)
        return None

    brier = float(brier_score_loss(y_test, probs))
    test_slim = test_slim.copy()
    test_slim["predicted_prob"] = probs

    log.info(
        "%s MLP: AUC=%.4f  Brier=%.4f  baseline=%.1f%%  "
        "train=%d rows/%d wins  test=%d rows/%d wins  epochs=%d",
        asset, auc, brier, baseline_wr * 100,
        n_train_rows, n_train_wins, n_test_rows, n_test_wins, n_epochs,
    )

    return {
        "asset":           asset,
        "pipe":            model,
        "features":        features,
        "train_df":        pd.DataFrame(),
        "test_df":         test_slim,
        "n_iter":          n_epochs,
        "auc":             auc,
        "brier":           brier,
        "brier_raw":       brier,
        "isotonic_auc":    None,
        "isotonic_brier":  None,
        "baseline_wr":     baseline_wr,
        "n_train":         n_train_rows,
        "n_test":          n_test_rows,
        "n_windows_train": n_train_wins,
        "n_windows_test":  n_test_wins,
    }


# ── Inference timing benchmark ────────────────────────────────────────────────

def benchmark_inference(results: list[dict], n_repeats: int = 100) -> str:
    """
    Time a flat MLP forward pass on one window's rows (~300 rows) per asset.
    """
    device_str = str(_get_device()) if _HAS_TORCH else "cpu (no torch)"
    out = [
        f"_Device: **{device_str}**.  Flat MLP — each row scored independently._\n",
        "| asset | features | rows | ms/window | ms/row | pass? |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for r in results:
        if r["test_df"].empty or "window_ts" not in r["test_df"].columns:
            continue

        wins = sorted(r["test_df"]["window_ts"].unique())
        mid  = r["test_df"][r["test_df"]["window_ts"] == wins[len(wins) // 2]]
        if mid.empty:
            continue

        model  = r["pipe"]
        model._ensure_net()
        net    = model._net
        device = model._device
        feats  = r["features"]

        batch_df = mid.sort_values("elapsed_second") if "elapsed_second" in mid.columns else mid
        X_f32    = model._preprocess(batch_df[feats].to_numpy())
        rows     = len(X_f32)
        t_in     = torch.from_numpy(X_f32).to(device)

        with torch.no_grad():
            _ = net(t_in)  # warm up

        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = net(t_in)
            times.append((time.perf_counter() - t0) * 1000)

        med  = float(np.median(times))
        flag = "✓" if med < 250 else "✗ SLOW"
        out.append(f"| {r['asset']} | {len(feats)} | {rows} | {med:.2f} | {med/rows:.3f} | {flag} |")

    out.append("\n_Median over 100 repeats. Does not include feature engineering time._")
    return "\n".join(out)


# ── Feature importance (permutation) ─────────────────────────────────────────

def compute_permutation_importance(
    result:     dict,
    n_repeats:  int = 5,
    max_windows: int = 100,
) -> dict[str, float]:
    """
    Permutation importance on the test set using AUC as the scoring metric.
    Shuffles one feature column globally, re-scores all rows independently
    (flat MLP — no sequence grouping needed).
    Capped at max_windows to keep runtime reasonable.
    """
    test_df  = result["test_df"]
    features = result["features"]
    model    = result["pipe"]

    if test_df.empty or len(test_df) < 100:
        return {}

    # Sample a subset of windows (not rows) to keep runtime reasonable
    all_wins  = sorted(test_df["window_ts"].unique())
    sampled   = all_wins[:max_windows] if len(all_wins) > max_windows else all_wins
    df_sub    = test_df[test_df["window_ts"].isin(sampled)].copy()
    y         = df_sub["resolved_up"].values

    # Baseline AUC — pass full DataFrame so GRU groups by window_ts
    p_base   = model.predict_proba(df_sub)[:, 1]
    auc_base = roc_auc_score(y, p_base)

    importances: dict[str, float] = {}
    rng = np.random.default_rng(42)

    for feat in features:
        drop_aucs = []
        for _ in range(n_repeats):
            df_perm          = df_sub.copy()
            df_perm[feat]    = rng.permutation(df_perm[feat].values)
            p_perm           = model.predict_proba(df_perm)[:, 1]
            drop_aucs.append(auc_base - roc_auc_score(y, p_perm))
        importances[feat] = float(np.mean(drop_aucs))

    return importances


def section_nn_feature_importance(results: list[dict]) -> str:
    out = [
        "_Permutation importance: ΔAUC = AUC drop when feature is globally shuffled "
        "(breaking its within-window variation while keeping sequence structure). "
        "Computed on up to 100 test windows, averaged over 5 shuffles._\n"
    ]
    for r in results:
        asset = r["asset"]
        out.append(f"### {asset}\n")
        imp = compute_permutation_importance(r)
        if not imp:
            out.append("_Importance unavailable._\n")
            continue
        pairs = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        out.append("| feature | ΔAUC |")
        out.append("|---|---:|")
        for feat, delta in pairs:
            flag = " ←" if delta > 0.001 else ""
            out.append(f"| `{feat}` | {delta:+.5f}{flag} |")
        out.append("")
    return "\n".join(out)


# ── Metrics section ───────────────────────────────────────────────────────────

def section_nn_metrics(results: list[dict]) -> str:
    rows = []
    for r in results:
        rows.append({
            "asset":         r["asset"],
            "win_train":     r["n_windows_train"],
            "win_test":      r["n_windows_test"],
            "n_train":       r["n_train"],
            "n_test":        r["n_test"],
            "epochs":        r.get("n_iter", "—"),
            "AUC-ROC":       round(r["auc"], 4) if r["auc"] else "—",
            "Brier":         round(r["brier"], 4) if r["brier"] else "—",
            "baseline_win%": f"{r['baseline_wr']*100:.1f}%",
        })
    return pd.DataFrame(rows).to_markdown(index=False)


# ── Report builder ────────────────────────────────────────────────────────────

def build_nn_report(
    results:       list[dict],
    pm_lookup:     dict,
    dn_lookup:     dict,
    generated_at:  str,
    up_imb_lookup: dict | None = None,
    dn_imb_lookup: dict | None = None,
) -> str:
    dn = dn_lookup or {}

    summary_lines = []
    for r in results:
        asset  = r["asset"]
        valid  = _augment_both_sides(r["test_df"].copy(), asset, pm_lookup, dn)
        cands  = valid[valid["edge"] >= DEFAULT_THRESHOLD].sort_values("elapsed_second")
        trades = cands.groupby("window_ts").first().reset_index() if not cands.empty else pd.DataFrame()
        if trades.empty:
            summary_lines.append(f"- **{asset}**: AUC={r['auc']:.3f} — no trades at {DEFAULT_THRESHOLD}")
            continue
        won    = trades["effective_won"]
        fill_p = trades["pm_fill_price"].fillna(trades["pm_price_signal"])
        pnl    = _pnl_per_trade(fill_p, won)
        n_win  = valid["window_ts"].nunique()
        summary_lines.append(
            f"- **{asset}**: AUC={r['auc']:.3f} — "
            f"{len(trades)}/{n_win} windows ({len(trades)/n_win*100:.0f}%) — "
            f"win={won.mean()*100:.1f}% — avg_pnl={pnl.mean():+.4f} — total={pnl.sum():+.4f}"
        )

    n_base_feats  = len(NN_FEATURES)
    n_cross_feats = len(results[0].get("features", NN_FEATURES)) - n_base_feats if results else 0
    feat_desc = (
        f"{n_base_feats} base + {n_cross_feats} cross-asset move_sigmas"
        if n_cross_feats else f"{n_base_feats}"
    )

    lines = [
        "# Flat MLP Model Report: Edge-Based Entry Strategy",
        f"_Generated {generated_at}_",
        "",
        f"**Architecture:** MLP — BatchNorm → "
        f"{' → '.join(str(h) for h in MLP_HIDDEN)} → sigmoid",
        f"**Features:** {feat_desc} — instantaneous signals + path-dependent + PM order flow + cross-asset momentum",
        "**Training:** AdamW + cosine LR + elapsed-weighted BCE (early-window rows weighted "
        "1/(elapsed+1)). PM windows only. Temporal val split (last 15% of train windows).",
        "",
        f"### At-a-glance (threshold = {DEFAULT_THRESHOLD})",
        "",
        "\n".join(summary_lines),
        "",
        "---",
        "",
        "## 1. Inference Speed",
        "",
        "Flat MLP — each row scored independently, no sequence state.",
        "",
        benchmark_inference(results),
        "",
        "---",
        "",
        "## 2. Calibration",
        "",
        "BCELoss directly optimises calibrated probabilities. "
        "No Platt scaling — raw sigmoid output is used.",
        "",
        section_calibration(results),
        "",
        "---",
        "",
        "## 3. EV Sweep — What Threshold Should I Use?",
        "",
        "`avg_pnl` includes CLOB fee. Best side (UP / DOWN) at first trigger per window.",
        "",
        section_threshold_ev(results, pm_lookup, dn),
        "",
        "---",
        "",
        "## 4. Execution Slippage — 1s vs 2s Fill Lag",
        "",
        f"Measured at actual trade moments (threshold={DEFAULT_THRESHOLD}). Does NOT include fee.",
        "",
        section_slippage(results, pm_lookup, dn),
        "",
        "---",
        "",
        "## 5. Entry Timing",
        "",
        section_entry_timing(results, pm_lookup, dn),
        "",
        "---",
        "",
        "## 6. Feature Importance — What Does the GRU Learn?",
        "",
        "Permutation importance shuffles one feature globally across all windows, "
        "then re-runs sequential GRU inference. Measures each feature's contribution "
        "to the GRU's sequence-level prediction.",
        "",
        section_nn_feature_importance(results),
        "",
        "---",
        "",
        "## 7. Model Quality Metrics",
        "",
        "_AUC-ROC: 0.5 = random, 1.0 = perfect. "
        "Brier: raw sigmoid output — lower = better probability accuracy._",
        "",
        section_nn_metrics(results),
        "",
        "---",
        "",
        "## 8. Edge by Model Confidence Decile",
        "",
        section_edge_by_decile(results, pm_lookup, dn),
        "",
        "---",
        "",
        "## 9. Hour of Day",
        "",
        section_hour_of_day(results, pm_lookup, dn),
        "",
        "---",
        "",
        "## 9b. Day of Week",
        "",
        section_day_of_week(results, pm_lookup, dn),
        "",
        "---",
        "",
        "## 10. Recent Performance — Last 24h and 48h",
        "",
        section_recent_windows(results, pm_lookup, dn),
        "",
        "---",
        "",
        "## 11. Orderbook Imbalance",
        "",
        section_orderbook_imbalance(
            results, pm_lookup, dn,
            up_imb_lookup or {}, dn_imb_lookup or {},
        ),
        "",
        "---",
        "",
        "## Methodology",
        "",
        "### Architecture",
        "",
        f"```",
        f"Input({feat_desc} features)",
        f"  └─ BatchNorm1d",
        *[f"  └─ Linear(→{h}) → GELU → Dropout({MLP_DROPOUT})" for h in MLP_HIDDEN],
        f"  └─ Linear(→1) → Sigmoid",
        f"```",
        "",
        "**Why flat MLP:** Each row (second) is scored independently. "
        "`elapsed_second` is included as a feature so the model learns timing-dependent "
        "signal without needing sequence state. This fires confidently at second 1 — "
        "unlike a GRU which needs warmup before producing stable hidden states.",
        "",
        "### Features",
        "",
        f"**{feat_desc} features** — instantaneous + path-dependent + PM order flow + cross-asset:\n",
        "| group | feature | captures |",
        "|---|---|---|",
        "| Directional | `move_sigmas`, `move_x_elapsed` | cumulative σ-move and its interaction with time |",
        "| Timing | `elapsed_second`, `hour_sin/cos` | window position and time-of-day |",
        "| Momentum | `vel_5s`, `vel_10s`, `vel_20s`, `vel_decay` | short/medium momentum across timescales |",
        "| Range | `dist_low_30`, `dist_high_30` | σ-distance from 30s rolling extremes |",
        "| MA deviation | `ma_dev_10s`, `ma_dev_30s` | price vs recent rolling mean — position in range |",
        "| Volume | `vol_10s_log`, `signed_vol_imb` | participation and directional imbalance |",
        "| Trend | `trend_str_30`, `dir_consistency_10` | trend quality signals |",
        "| PM flow | `up_imbalance`, `dn_imbalance` | order-book buying/selling pressure not yet priced in |",
        "| **Path** | `vwap_dev` | deviation from volume-weighted avg price in window |",
        "| **Path** | `chan_pos` | position in running high-low channel [0=low, 1=high] |",
        "| **Path** | `max_up_excursion`, `max_dn_excursion` | max move from open in each direction |",
        "| **Path** | `dir_consistency_window` | running fraction of up-ticks since window open |",
        "| **Cross-asset** | `move_sigmas_<OTHER>` | each other asset's running σ-move at this second — macro crypto momentum |",
        "",
        "### Training protocol",
        "",
        f"- **Optimiser:** AdamW (lr={MLP_LR}, weight_decay={MLP_WEIGHT_DECAY})",
        f"- **Loss:** elapsed-weighted BCE — w=1/(elapsed+1) focuses on early-window entries",
        f"- **Gradient clipping:** max norm {MLP_GRAD_CLIP}",
        f"- **Early stopping:** patience={MLP_PATIENCE} epochs, val = last {int(MLP_VAL_WIN_FRAC*100)}% of PM windows (time-ordered)",
        f"- **Batch size:** {MLP_BATCH_SIZE} rows",
        "- **Imputer/scaler:** fit on fit-split only; applied to val, test, and inference",
        "- **Device:** MPS (Apple Silicon GPU) → CUDA → CPU",
        "",
        "### Edge formula",
        "",
        "- `edge_up = predicted_prob − (up_ask + 0.072 × up_ask × (1 − up_ask))`",
        "- `edge_dn = (1 − predicted_prob) − (dn_ask + 0.072 × dn_ask × (1 − dn_ask))`",
        f"- CLOB fee: C × {FEE_GAMMA} × p × (1 − p) in all PnL tables",
    ]
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train per-asset GRU+MLP for P(resolved_up)")
    p.add_argument("--prices-dir",  default="data/prices")
    p.add_argument("--coin-dir",    default="data/coin_prices")
    p.add_argument("--out-report",  default="data/reports/nn_report.md")
    p.add_argument("--out-models",  default="data/models")
    p.add_argument("--assets",      nargs="+", default=None)
    p.add_argument("--ewma-lambda", type=float, default=EWMA_LAMBDA)
    p.add_argument("--no-save",     action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_models, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_report), exist_ok=True)

    assets = [a.upper() for a in args.assets] if args.assets else list(ASSET_TO_SYMBOL.keys())
    log.info("Assets: %s  |  Device: %s", assets,
             str(_get_device()) if _HAS_TORCH else "cpu (no torch)")

    results:       list[dict]                   = []
    pm_lookup:     dict[tuple[str, int], float] = {}
    dn_lookup:     dict[tuple[str, int], float] = {}
    up_imb_lookup: dict[tuple[str, int], float] = {}
    dn_imb_lookup: dict[tuple[str, int], float] = {}

    # ── Phase 1: load data + build per-asset datasets ─────────────────────────
    # We need all assets' move_sigmas before training so each model can receive
    # the other assets' running σ-move as cross-asset momentum features.
    asset_dfs:   dict[str, pd.DataFrame] = {}
    asset_sigma: dict[str, float]        = {}
    # cross_move[A] = slim df with columns [window_ts, elapsed_second, move_sigmas_A]
    cross_move:  dict[str, pd.DataFrame] = {}

    for asset in assets:
        log.info("=== %s  (loading) ===", asset)
        loaded = _load_asset(args, asset)
        if loaded is None:
            continue
        pm_df, close_series, volume_series, sigma = loaded

        for ts, ask in zip(pm_df.loc[pm_df["up_ask"].notna(), "ts"],
                           pm_df.loc[pm_df["up_ask"].notna(), "up_ask"]):
            pm_lookup[(asset, int(ts))] = float(ask)
        for ts, ask in zip(pm_df.loc[pm_df["dn_ask"].notna(), "ts"],
                           pm_df.loc[pm_df["dn_ask"].notna(), "dn_ask"]):
            dn_lookup[(asset, int(ts))] = float(ask)
        if "up_imbalance" in pm_df.columns:
            for ts, v in zip(pm_df.loc[pm_df["up_imbalance"].notna(), "ts"],
                             pm_df.loc[pm_df["up_imbalance"].notna(), "up_imbalance"]):
                up_imb_lookup[(asset, int(ts))] = float(v)
        if "dn_imbalance" in pm_df.columns:
            for ts, v in zip(pm_df.loc[pm_df["dn_imbalance"].notna(), "ts"],
                             pm_df.loc[pm_df["dn_imbalance"].notna(), "dn_imbalance"]):
                dn_imb_lookup[(asset, int(ts))] = float(v)

        df = build_market_features_dataset(
            asset, pm_df, close_series, volume_series, sigma,
            ewma_lambda=args.ewma_lambda,
        )
        del pm_df, close_series, volume_series
        gc.collect()

        if df.empty:
            continue

        asset_dfs[asset]   = df
        asset_sigma[asset] = sigma
        cross_move[asset]  = (
            df[["window_ts", "elapsed_second", "move_sigmas"]]
            .rename(columns={"move_sigmas": f"move_sigmas_{asset}"})
        )

    # ── Phase 2: join cross-asset move_sigmas, then train each model ──────────
    for asset in assets:
        if asset not in asset_dfs:
            continue

        df    = asset_dfs.pop(asset)
        sigma = asset_sigma[asset]

        # Join other assets' move_sigmas on (window_ts, elapsed_second).
        # Windows are UTC-aligned so timestamps match across assets.
        # Missing rows get NaN — handled by the median imputer inside train_asset_nn.
        other_assets = [a for a in assets if a != asset and a in cross_move]
        for other in other_assets:
            df = df.merge(
                cross_move[other],
                on=["window_ts", "elapsed_second"],
                how="left",
            )
        cross_features = [f"move_sigmas_{a}" for a in other_assets]
        features       = NN_FEATURES + cross_features

        log.info("=== %s  (training, %d features) ===", asset, len(features))
        result = train_asset_nn(asset, df, features=features)
        del df
        gc.collect()

        if result is None:
            continue

        result.pop("train_df", None)
        result["sigma"]    = sigma
        result["features"] = features
        results.append(result)

        if not args.no_save:
            model_path = os.path.join(args.out_models, f"{asset.lower()}_nn.joblib")
            joblib.dump({
                "type":     "mlp",
                "pipe":     result["pipe"],
                "features": features,
                "sigma":    sigma,
            }, model_path)
            log.info("%s: MLP model → %s", asset, model_path)

    if not results:
        log.error("No results produced.")
        sys.exit(1)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = build_nn_report(
        results, pm_lookup, dn_lookup, generated_at,
        up_imb_lookup=up_imb_lookup,
        dn_imb_lookup=dn_imb_lookup,
    )

    with open(args.out_report, "w") as f:
        f.write(report)
    log.info("Report → %s", args.out_report)
    print(report)


if __name__ == "__main__":
    main()
