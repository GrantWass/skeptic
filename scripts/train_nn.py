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
    BUY_FEE_RATE,
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
# 17 instantaneous signals (no interaction terms — GRU learns them) +
# 8 path-dependent quant features.
# Dropped from the original 22-feature MLP set based on permutation importance:
# vel_ratio, time_since_flip, zscore_20, vol_z_30, acc_4s, vel_2s.
NN_FEATURES = [
    # Core directional signal
    "move_sigmas",
    "elapsed_second",
    "move_x_elapsed",           # explicit top-2 interaction — strong prior signal
    "hour_sin",
    "hour_cos",
    # Momentum
    "vel_5s",
    "vel_10s",
    "vel_decay",
    "mom_slope",
    # Curvature
    "acc_10s",
    # Range / local structure
    "dist_low_30",
    "dist_high_30",
    # Volume / participation
    "vol_10s_log",
    "signed_vol_imb",
    # Trend structure
    "trend_str_30",
    "vol_expansion",
    "dir_consistency_10",
    # ── Quant path features ──────────────────────────────────────────────────
    # These encode WHERE price has been in the window — what the GRU uses
    # to build "memory" of the path taken, not just the current state.
    "vwap_dev",               # above/below volume-weighted avg price in window
    "chan_pos",               # position in running high-low channel [0, 1]
    "max_up_excursion",       # max upside from open (trend persistence evidence)
    "max_dn_excursion",       # max downside from open (drawdown / shakeout depth)
    "move_efficiency",        # |move| / range — clean trend vs choppy retracement
    "dir_consistency_window", # running fraction of up-ticks since window open
    "pv_corr_10",             # 10s price-change / volume correlation
    "vol_accel",              # 5s / 20s volume ratio (expanding vs fading volume)
]

# ── GRU hyperparameters ───────────────────────────────────────────────────────
GRU_HIDDEN        = 64    # hidden units per layer
GRU_LAYERS        = 2     # stacked GRU depth
GRU_DROPOUT       = 0.20  # applied between GRU layers and in MLP head
MLP_HEAD          = (128, 64)  # MLP head dims after GRU
GRU_LR            = 3e-4  # lower LR — 1e-3 caused val divergence
GRU_WEIGHT_DECAY  = 1e-4
GRU_BATCH_WINDOWS = 256   # large batches — all data pre-padded, no collation overhead
GRU_MAX_EPOCHS    = 100
GRU_PATIENCE      = 15
GRU_GRAD_CLIP     = 1.0
GRU_VAL_WIN_FRAC  = 0.15  # last 15% of train windows (time-ordered) for early stopping
GRU_MAX_TRAIN_WINS = 6000 # cap recent windows — older data less predictive, speeds up epoch


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

class _GRUModel(nn.Module):
    """
    2-layer GRU → LayerNorm → MLP head → sigmoid.

    forward(x) processes a full batch of padded sequences.
    step(x_t, h)  advances one second — used at inference to avoid
                  reprocessing the full sequence.
    """

    def __init__(
        self,
        n_in:    int,
        hidden:  int             = GRU_HIDDEN,
        n_layers: int            = GRU_LAYERS,
        head:    tuple[int, ...] = MLP_HEAD,
        dropout: float           = GRU_DROPOUT,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.hidden   = hidden

        # Normalise inputs before GRU — stabilises training on mixed-scale features
        self.input_norm = nn.LayerNorm(n_in)

        self.gru = nn.GRU(
            input_size  = n_in,
            hidden_size = hidden,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )

        # MLP head: GRU output → probability
        layers: list[nn.Module] = [nn.LayerNorm(hidden)]
        prev = hidden
        for h in head:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        x: [batch, seq_len, n_features]
        returns: [batch, seq_len]  — probability at every second
        """
        h, _ = self.gru(self.input_norm(x))
        return torch.sigmoid(self.head(h)).squeeze(-1)

    def step(
        self,
        x_t:   "torch.Tensor",   # [1, n_features]  — one second's features
        h_prev: "torch.Tensor",  # [n_layers, 1, hidden]
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Single-step inference — O(1), maintains running hidden state."""
        # input_norm operates on feature dimension regardless of batch/seq dims
        x_in = self.input_norm(x_t.unsqueeze(0))   # [1, 1, n_features]
        h_out, h_new = self.gru(x_in, h_prev)       # h_out: [1, 1, hidden]
        prob = torch.sigmoid(self.head(h_out)).squeeze()
        return prob, h_new


# ── Inference wrapper (serializable) ─────────────────────────────────────────

class TorchGRUModel:
    """
    Sklearn-compatible predict_proba around a trained _GRUModel.

    Weights stored as a CPU state-dict for safe joblib serialization.
    Device is chosen lazily on first call (MPS → CUDA → CPU).

    predict_proba(X) accepts either:
      - pd.DataFrame with feature columns (+ optionally window_ts for
        correct sequential grouping)
      - np.ndarray of shape [T, n_features] (treated as one sequence)
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
            net = _GRUModel(self._n_in).to(self._device)
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
        """
        If X is a DataFrame with 'window_ts', process each window as its own
        sequence (correct causal behaviour).  Otherwise treat all rows as a
        single sequence (used by permutation importance with shuffled features).
        """
        self._ensure_net()

        if isinstance(X, pd.DataFrame):
            feat_cols = [f for f in self._features if f in X.columns]
        else:
            X = pd.DataFrame(X, columns=self._features)
            feat_cols = self._features

        if "window_ts" in X.columns:
            out = np.zeros(len(X), dtype=np.float32)
            iloc_map = {orig: pos for pos, orig in enumerate(X.index)}
            for _, grp in X.groupby("window_ts"):
                grp_sorted = grp.sort_values("elapsed_second") if "elapsed_second" in grp.columns else grp
                X_f32 = self._preprocess(grp_sorted[feat_cols].to_numpy())
                with torch.no_grad():
                    t = torch.from_numpy(X_f32).unsqueeze(0).to(self._device)
                    probs = self._net(t).squeeze(0).cpu().numpy()
                for orig_idx, p in zip(grp_sorted.index, probs):
                    out[iloc_map[orig_idx]] = p
        else:
            X_f32 = self._preprocess(X[feat_cols].to_numpy())
            with torch.no_grad():
                t = torch.from_numpy(X_f32).unsqueeze(0).to(self._device)
                probs = self._net(t).squeeze(0).cpu().numpy()
            out = probs

        return np.column_stack([1 - out, out])


# ── Sequence helpers ──────────────────────────────────────────────────────────

def _build_window_sequences(
    df:       pd.DataFrame,
    features: list[str],
    imp:      SimpleImputer,
    sc:       StandardScaler,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Group df by window_ts, sort by elapsed_second, preprocess features.
    Returns list of (X_preprocessed[T, F], y[T]) tuples — one per window.
    """
    windows = []
    for _, grp in df.groupby("window_ts"):
        grp   = grp.sort_values("elapsed_second")
        X_raw = grp[features].to_numpy(dtype=np.float64)
        X_f32 = sc.transform(imp.transform(X_raw)).astype(np.float32)
        y     = grp["resolved_up"].to_numpy(dtype=np.float32)
        windows.append((X_f32, y))
    return windows


def _pad_windows(
    windows: list[tuple[np.ndarray, np.ndarray]],
    max_T:   int,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Pre-pad all windows to max_T once at build time — eliminates per-batch
    CPU collation overhead during training.
    Returns GPU-ready tensors: X[N, max_T, F], y[N, max_T], mask[N, max_T].
    """
    N = len(windows)
    F = windows[0][0].shape[1]
    X_pad = np.zeros((N, max_T, F), dtype=np.float32)
    y_pad = np.zeros((N, max_T),    dtype=np.float32)
    mask  = np.zeros((N, max_T),    dtype=np.float32)
    for i, (x, y) in enumerate(windows):
        T            = x.shape[0]
        X_pad[i, :T] = x
        y_pad[i, :T] = y
        mask[i, :T]  = 1.0
    return (
        torch.from_numpy(X_pad),
        torch.from_numpy(y_pad),
        torch.from_numpy(mask),
    )


# ── GRU training loop ─────────────────────────────────────────────────────────

def _train_gru(
    windows_tr: list[tuple[np.ndarray, np.ndarray]],
    windows_vl: list[tuple[np.ndarray, np.ndarray]],
    n_in:       int,
    device:     "torch.device",
) -> tuple[dict, int]:
    """
    Train _GRUModel with AdamW + cosine LR schedule + gradient clipping.
    Early stopping on validation BCE loss (temporal split — last 15% of windows).

    All windows are pre-padded to a fixed length once before training begins.
    No per-batch collation — batches are pure index slices into GPU tensors.
    Best weights are copied to CPU immediately; all GPU tensors freed before return.
    """
    max_T = max(x.shape[0] for x, _ in windows_tr + windows_vl)

    # Pre-pad and move to device once — no per-batch CPU work
    X_tr, y_tr, mask_tr = _pad_windows(windows_tr, max_T)
    X_vl, y_vl, mask_vl = _pad_windows(windows_vl, max_T)
    X_tr    = X_tr.to(device);    y_tr    = y_tr.to(device);    mask_tr = mask_tr.to(device)
    X_vl    = X_vl.to(device);    y_vl    = y_vl.to(device);    mask_vl = mask_vl.to(device)

    N_tr = X_tr.shape[0]
    B    = GRU_BATCH_WINDOWS

    net   = _GRUModel(n_in).to(device)
    opt   = torch.optim.AdamW(net.parameters(), lr=GRU_LR, weight_decay=GRU_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=GRU_MAX_EPOCHS)
    crit  = nn.BCELoss()

    best_loss:  float     = float("inf")
    best_state: dict | None = None
    no_improve: int       = 0
    epoch:      int       = 0
    rng = torch.Generator(device="cpu")
    rng.manual_seed(42)

    for epoch in range(GRU_MAX_EPOCHS):
        # ── Training: random index batches, no CPU collation ───────────────
        net.train()
        perm = torch.randperm(N_tr, generator=rng)
        for start in range(0, N_tr, B):
            idx    = perm[start: start + B]
            X_b    = X_tr[idx]
            y_b    = y_tr[idx]
            mask_b = mask_tr[idx]
            opt.zero_grad(set_to_none=True)
            loss   = crit(net(X_b)[mask_b.bool()], y_b[mask_b.bool()])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), GRU_GRAD_CLIP)
            opt.step()

        sched.step()

        # ── Validation (single pass — val set fits in memory) ─────────────
        net.eval()
        with torch.no_grad():
            probs_vl = net(X_vl)[mask_vl.bool()]
            vl_loss  = crit(probs_vl, y_vl[mask_vl.bool()]).item()

        if vl_loss < best_loss - 1e-6:
            best_loss  = vl_loss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            log.info("  epoch %3d  val_bce=%.5f  no_improve=%d",
                     epoch + 1, vl_loss, no_improve)

        if no_improve >= GRU_PATIENCE:
            log.info("  early stop at epoch %d", epoch + 1)
            break

    # Free GPU tensors before returning
    del net, X_tr, y_tr, mask_tr, X_vl, y_vl, mask_vl
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
    Train GRU + MLP head on this asset's per-second window data.

    Memory strategy:
    - Extract numpy arrays and delete the DataFrame before GPU allocation.
    - Fit imputer/scaler on fit split only, apply to val/test.
    - GRU processes windows as sequences; DataLoader handles variable lengths.
    - Best weights copied to CPU immediately after training; GPU tensors freed.
    """
    if not _HAS_TORCH:
        log.error("PyTorch not installed — cannot train. pip install torch")
        return None

    device = _get_device()
    log.info("%s GRU: device=%s  features=%d", asset, device, len(features))

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

    # ── Temporal val split: last GRU_VAL_WIN_FRAC of training windows ─────
    all_train_wins = sorted(train_df["window_ts"].unique())
    n_val_wins     = max(1, int(len(all_train_wins) * GRU_VAL_WIN_FRAC))
    val_set        = set(all_train_wins[-n_val_wins:])
    fit_set        = set(all_train_wins[:-n_val_wins])

    fit_df = train_df[train_df["window_ts"].isin(fit_set)]
    val_df = train_df[train_df["window_ts"].isin(val_set)]

    n_train_wins = len(all_train_wins)
    n_train_rows = len(train_df)
    baseline_wr  = float(train_df["resolved_up"].mean())

    # Fit preprocessors on fit split only (no data leakage into val/test)
    X_fit_np = fit_df[features].to_numpy(dtype=np.float64)
    imp, sc, _ = _fit_preprocessors(X_fit_np)
    del X_fit_np

    del train_df, test_df  # release full DataFrames before GPU work
    gc.collect()

    # ── Build window sequences ─────────────────────────────────────────────
    log.info("%s GRU: building sequences — fit=%d wins, val=%d wins",
             asset, len(fit_set), len(val_set))
    windows_fit = _build_window_sequences(fit_df, features, imp, sc)
    windows_val = _build_window_sequences(val_df, features, imp, sc)
    del fit_df, val_df
    gc.collect()

    # ── Train ──────────────────────────────────────────────────────────────
    log.info("%s GRU: training on %s …", asset, device)
    best_state, n_epochs = _train_gru(windows_fit, windows_val, len(features), device)
    del windows_fit, windows_val
    gc.collect()

    log.info("%s GRU: converged in %d epochs", asset, n_epochs)
    model = TorchGRUModel(imp, sc, best_state, len(features), features)

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

    # ── Evaluate on test set ───────────────────────────────────────────────
    # predict_proba groups by window_ts — correct sequential inference
    probs = model.predict_proba(test_slim)[:, 1]
    y_test = test_slim["resolved_up"].to_numpy(dtype=np.int32)

    try:
        auc = float(roc_auc_score(y_test, probs))
    except ValueError as e:
        log.warning("%s GRU: roc_auc_score failed: %s", asset, e)
        return None

    brier = float(brier_score_loss(y_test, probs))
    test_slim = test_slim.copy()
    test_slim["predicted_prob"] = probs

    log.info(
        "%s GRU: AUC=%.4f  Brier=%.4f  baseline=%.1f%%  "
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
    Time a full-window GRU pass (one 5-minute window ≈ 300 rows) per asset.
    Also times single-step inference (what production uses each second).
    """
    device_str = str(_get_device()) if _HAS_TORCH else "cpu (no torch)"
    out = [
        f"_Device: **{device_str}**.  "
        f"Full-window = process all 300 rows at once (report mode). "
        f"Single-step = one GRU step per new second (production mode)._\n",
        "| asset | features | rows | full-window ms | single-step ms | pass? |",
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

        # Preprocess one window
        batch_df = mid.sort_values("elapsed_second")
        X_f32    = model._preprocess(batch_df[feats].to_numpy())
        rows     = len(X_f32)

        # ── Full-window timing ─────────────────────────────────────────
        t_in = torch.from_numpy(X_f32).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = net(t_in)  # warm up

        full_times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = net(t_in)
            full_times.append((time.perf_counter() - t0) * 1000)

        # ── Single-step timing (one GRU step) ─────────────────────────
        h0   = torch.zeros(GRU_LAYERS, 1, GRU_HIDDEN, device=device)
        x1   = t_in[:, 0, :]   # one second: [1, F]
        with torch.no_grad():
            _, _ = net.step(x1, h0)

        step_times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            with torch.no_grad():
                _, _ = net.step(x1, h0)
            step_times.append((time.perf_counter() - t0) * 1000)

        full_med = float(np.median(full_times))
        step_med = float(np.median(step_times))
        flag     = "✓" if full_med < 250 else "✗ SLOW"

        out.append(
            f"| {r['asset']} | {len(feats)} | {rows} "
            f"| {full_med:.1f} | {step_med:.2f} | {flag} |"
        )

    out.append(
        "\n_Full-window = entire 300-row sequence in one forward pass. "
        "Single-step = one GRU hidden-state update (production per-second cost). "
        "Median over 100 repeats. Does not include feature engineering time._"
    )
    return "\n".join(out)


# ── Feature importance (permutation) ─────────────────────────────────────────

def compute_permutation_importance(
    result:     dict,
    n_repeats:  int = 5,
    max_windows: int = 100,
) -> dict[str, float]:
    """
    Permutation importance on the test set using AUC as the scoring metric.

    Shuffles one feature column across all rows (globally), then re-runs
    sequential GRU inference grouped by window_ts.  ΔAUC measures how much
    the GRU relies on that feature's cross-window variation.

    Capped at max_windows for speed (GRU inference is more expensive than MLP).
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

    lines = [
        "# GRU + MLP Model Report: Edge-Based Entry Strategy",
        f"_Generated {generated_at}_",
        "",
        f"**Architecture:** 2-layer GRU (hidden={GRU_HIDDEN}) → "
        f"MLP head ({' → '.join(str(h) for h in MLP_HEAD)}) → sigmoid",
        f"**Features:** {len(NN_FEATURES)} — 17 instantaneous + 8 path-dependent quant signals",
        "**Training:** AdamW + cosine LR schedule + gradient clipping. "
        "Temporal val split (last 15% of train windows). No post-hoc calibration.",
        "",
        f"### At-a-glance (threshold = {DEFAULT_THRESHOLD})",
        "",
        "\n".join(summary_lines),
        "",
        "---",
        "",
        "## 1. Inference Speed",
        "",
        "Full-window = process all 300 rows as one sequence (how predictions are "
        "made for the report). Single-step = one GRU hidden-state update per second "
        "(production mode — the model maintains state between calls).",
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
        "`avg_pnl` includes a 1.5% buy fee. Best side (UP / DOWN) at first trigger per window.",
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
        f"Input({len(NN_FEATURES)} features)",
        f"  └─ LayerNorm",
        f"  └─ GRU(hidden={GRU_HIDDEN}, layers={GRU_LAYERS}, dropout={GRU_DROPOUT})",
        f"  └─ LayerNorm",
        *[f"  └─ Linear({a}→{b}) → GELU → Dropout({GRU_DROPOUT})"
          for a, b in zip((GRU_HIDDEN,) + MLP_HEAD[:-1], MLP_HEAD)],
        f"  └─ Linear({MLP_HEAD[-1]}→1) → Sigmoid",
        f"```",
        "",
        "**Why GRU over Transformer:** At inference we receive one second at a time. "
        "The GRU updates its hidden state in O(1) per step — just one matrix multiply. "
        "A Transformer would need to reprocess the full sequence each second "
        "(O(t²) attention) unless KV-caching is implemented. For ≤300-step sequences "
        "a GRU matches Transformer quality at ~5× lower inference cost.",
        "",
        "### Features",
        "",
        f"**{len(NN_FEATURES)} features** — 17 instantaneous + 8 path-dependent:\n",
        "| group | feature | captures |",
        "|---|---|---|",
        "| Directional | `move_sigmas`, `move_x_elapsed` | cumulative σ-move and its interaction with time |",
        "| Timing | `elapsed_second`, `hour_sin/cos` | window position and time-of-day |",
        "| Momentum | `vel_5s`, `vel_10s`, `vel_decay`, `mom_slope` | short/medium momentum and shape |",
        "| Curvature | `acc_10s` | second derivative — accelerating vs decelerating |",
        "| Range | `dist_low_30`, `dist_high_30` | σ-distance from 30s rolling extremes |",
        "| Volume | `vol_10s_log`, `signed_vol_imb` | participation and directional imbalance |",
        "| Trend | `trend_str_30`, `vol_expansion`, `dir_consistency_10` | trend quality signals |",
        "| **Path** | `vwap_dev` | deviation from volume-weighted avg price in window |",
        "| **Path** | `chan_pos` | position in running high-low channel [0=low, 1=high] |",
        "| **Path** | `max_up_excursion`, `max_dn_excursion` | max move from open in each direction |",
        "| **Path** | `move_efficiency` | \\|move\\| / range — trend cleanliness vs chop |",
        "| **Path** | `dir_consistency_window` | running fraction of up-ticks since window open |",
        "| **Path** | `pv_corr_10` | 10s price-change / volume correlation |",
        "| **Path** | `vol_accel` | 5s / 20s volume ratio — volume picking up or fading |",
        "",
        "### Training protocol",
        "",
        f"- **Optimiser:** AdamW (lr={GRU_LR}, weight_decay={GRU_WEIGHT_DECAY})",
        f"- **LR schedule:** CosineAnnealingLR over {GRU_MAX_EPOCHS} epochs",
        f"- **Gradient clipping:** max norm {GRU_GRAD_CLIP}",
        f"- **Early stopping:** patience={GRU_PATIENCE} epochs, val = last {int(GRU_VAL_WIN_FRAC*100)}% of train windows (time-ordered)",
        f"- **Batch:** {GRU_BATCH_WINDOWS} windows padded to max sequence length in batch",
        "- **Imputer/scaler:** fit on fit-split only; applied to val, test, and inference",
        "- **Device:** MPS (Apple Silicon GPU) → CUDA → CPU",
        "",
        "### Edge formula",
        "",
        "- `edge_up = predicted_prob − up_ask × 1.015`",
        "- `edge_dn = (1 − predicted_prob) − dn_ask × 1.015`",
        f"- Buy fee: {BUY_FEE_RATE*100:.1f}% in all PnL tables",
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

    for asset in assets:
        log.info("=== %s ===", asset)
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

        df = build_asset_dataset(
            asset, pm_df, close_series, volume_series, sigma,
            ewma_lambda=args.ewma_lambda,
        )
        if df.empty:
            del pm_df, close_series, volume_series
            gc.collect()
            continue

        # Align test split to market model's temporal split
        mkt_df = build_market_features_dataset(
            asset, pm_df, close_series, volume_series, sigma,
            ewma_lambda=args.ewma_lambda,
        )
        if not mkt_df.empty:
            mkt_test_wins = set(mkt_df[mkt_df["split"] == "test"]["window_ts"].unique())
            df = df.copy()
            df["split"] = df["window_ts"].apply(
                lambda w: "test" if w in mkt_test_wins else "train"
            )
        del mkt_df

        result = train_asset_nn(asset, df, features=NN_FEATURES)
        if result is None:
            del pm_df, close_series, volume_series, df
            gc.collect()
            continue

        result.pop("train_df", None)
        result["sigma"] = sigma
        results.append(result)

        if not args.no_save:
            model_path = os.path.join(args.out_models, f"{asset.lower()}_gru.joblib")
            joblib.dump({
                "type":     "gru_mlp",
                "pipe":     result["pipe"],
                "features": NN_FEATURES,
                "sigma":    sigma,
            }, model_path)
            log.info("%s: GRU model → %s", asset, model_path)

        del pm_df, close_series, volume_series, df
        gc.collect()

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
