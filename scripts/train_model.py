#!/usr/bin/env python3
"""
Per-asset logistic regression model: P(resolved_up) at any second of a window.

Features computed at each second t in each window:
  move_sigmas, elapsed_second, hour_utc,
  vel_2s, vel_5s, vel_10s, acc_4s, acc_10s,
  vel_ratio, vel_decay, acc_positive, vol_10s_log

Resolution: last up_price >= 0.95 → UP, <= 0.05 → DOWN, else fallback to coin direction.
Train/test split: chronological 80/20 at the window level per asset.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --assets BTC ETH
    python scripts/train_model.py --assets BTC --out-report data/reports/model_report.md
"""
import argparse
import joblib
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

WINDOW_SECS   = 300
TRAIN_FRAC    = 0.80
MIN_COIN_ROWS = 280
VOL_LOOKBACK  = 10

ASSET_TO_SYMBOL = {
    "BTC":  "BTCUSDT",
    "ETH":  "ETHUSDT",
    "SOL":  "SOLUSDT",
    "DOGE": "DOGEUSDT",
    "XRP":  "XRPUSDT",
    "BNB":  "BNBUSDT",
    "HYPE": "HYPEUSDT",
}

BASE_FEATURES = [
    "move_sigmas",
    "elapsed_second",
    "hour_utc",
    "vel_2s",
    "vel_5s",
    "vel_10s",
    "acc_4s",
    "acc_10s",
    "vel_ratio",
    "vel_decay",
    "vol_10s_log",
]

# Interaction features added on top of BASE_FEATURES
INTERACTION_FEATURES = [
    "move_x_elapsed",  # position × time: early moves can reverse, late moves hold
    "move_x_vol",      # position × volume: volume-confirmed move vs thin-air move
]

FEATURES = BASE_FEATURES + INTERACTION_FEATURES


# ── Data loading ───────────────────────────────────────────────────────────────

def load_pm_windows(prices_dir: str, asset: str) -> pd.DataFrame:
    """Load all prices_*.csv files, filter to asset. Returns ts, window_ts, up_price, down_price."""
    files = sorted(Path(prices_dir).glob("prices_*.csv"))
    if not files:
        raise FileNotFoundError(f"No prices_*.csv in {prices_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f, usecols=["ts", "window_ts", "asset", "up_price", "down_price"])
        frames.append(df[df["asset"] == asset])
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["ts", "window_ts"]).sort_values(["window_ts", "ts"])
    log.info("%s: loaded %d PM rows across %d windows", asset, len(out), out["window_ts"].nunique())
    return out.reset_index(drop=True)


def load_coin_series(coin_dir: str, asset: str) -> tuple[pd.Series | None, pd.Series | None]:
    """Load {SYMBOL}_1s.csv. Returns (close_series, volume_series) indexed by ts, or (None, None)."""
    symbol = ASSET_TO_SYMBOL.get(asset.upper())
    if symbol is None:
        log.warning("%s: no symbol mapping", asset)
        return None, None
    path = os.path.join(coin_dir, f"{symbol}_1s.csv")
    if not os.path.exists(path):
        log.warning("%s: coin file missing: %s", asset, path)
        return None, None
    df = pd.read_csv(path, usecols=["ts", "close", "volume"])
    if df.empty:
        log.warning("%s: coin file is empty", asset)
        return None, None
    df = df.drop_duplicates("ts").sort_values("ts")
    ts_idx = df["ts"].values
    close_series  = pd.Series(df["close"].values.astype(float),  index=ts_idx)
    volume_series = pd.Series(df["volume"].values.astype(float), index=ts_idx)
    log.info("%s: loaded %d coin rows", asset, len(close_series))
    return close_series, volume_series


# ── Sigma ──────────────────────────────────────────────────────────────────────

def compute_sigma(windows: list[int], close_series: pd.Series) -> float:
    """Global std dev of window close-minus-open moves (>=280 coin rows required)."""
    moves  = []
    ts_idx = close_series.index.values
    vals   = close_series.values
    for wts in windows:
        lo = int(np.searchsorted(ts_idx, wts,               side="left"))
        hi = int(np.searchsorted(ts_idx, wts + WINDOW_SECS, side="left"))
        if (hi - lo) < MIN_COIN_ROWS:
            continue
        moves.append(float(vals[hi - 1]) - float(vals[lo]))
    if len(moves) < 10:
        raise ValueError(f"Only {len(moves)} valid windows for sigma — not enough data")
    return float(np.std(moves))


# ── Resolution ─────────────────────────────────────────────────────────────────

def resolve_window(pm_window: pd.DataFrame, window_move: float) -> bool | None:
    """
    0.95/0.05 PM threshold, fallback to coin direction.
    Returns None if window_move == 0 and PM is ambiguous.
    """
    if not pm_window.empty and "up_price" in pm_window.columns:
        last_up = pm_window.sort_values("ts")["up_price"].dropna()
        if not last_up.empty:
            v = float(last_up.iloc[-1])
            if v >= 0.95:
                return True
            if v <= 0.05:
                return False
    if window_move > 0:
        return True
    if window_move < 0:
        return False
    return None


# ── Per-window feature computation ────────────────────────────────────────────

def build_window_features(
    window_ts:     int,
    close_series:  pd.Series,
    volume_series: pd.Series,
    sigma:         float,
    hour_utc:      int,
) -> pd.DataFrame:
    """
    Vectorized per-window feature computation.
    Returns a DataFrame with one row per second where coin data exists.
    """
    ts_idx = close_series.index.values
    prices = close_series.values
    vols   = volume_series.values

    lo = int(np.searchsorted(ts_idx, window_ts,               side="left"))
    hi = int(np.searchsorted(ts_idx, window_ts + WINDOW_SECS, side="left"))

    if (hi - lo) < MIN_COIN_ROWS:
        return pd.DataFrame()

    win_ts    = ts_idx[lo:hi]
    win_price = prices[lo:hi]
    win_vol   = vols[lo:hi]
    n         = len(win_ts)

    open_price = win_price[0]
    move_sigmas = (win_price - open_price) / sigma
    elapsed    = (win_ts - window_ts).astype(float)

    def _vel(lb: int) -> np.ndarray:
        out = np.full(n, np.nan)
        if lb < n:
            out[lb:] = (win_price[lb:] - win_price[:n - lb]) / sigma
        return out

    vel_2s  = _vel(2)
    vel_5s  = _vel(5)
    vel_10s = _vel(10)

    def _acc(half: int) -> np.ndarray:
        lb  = half * 2
        out = np.full(n, np.nan)
        if lb < n:
            out[lb:] = (win_price[lb:] - 2 * win_price[half:n - half] + win_price[:n - lb]) / sigma
        return out

    acc_4s  = _acc(2)
    acc_10s = _acc(5)

    with np.errstate(divide="ignore", invalid="ignore"):
        vel_ratio = np.where(np.abs(vel_10s) > 1e-12, np.abs(vel_2s) / np.abs(vel_10s), np.nan)

    vel_decay = np.abs(vel_10s) - np.abs(vel_2s)

    # vol_10s_log: sum of volume in [t-VOL_LOOKBACK, t) via cumsum
    cum_vol = np.concatenate([[0.0], np.cumsum(win_vol)])
    vol_10s = np.full(n, np.nan)
    for i in range(n):
        t_lo        = win_ts[i] - VOL_LOOKBACK
        idx_lo      = int(np.searchsorted(win_ts, t_lo, side="left"))
        vol_10s[i]  = cum_vol[i] - cum_vol[idx_lo]
    vol_10s_log = np.log1p(vol_10s)

    df = pd.DataFrame({
        "ts":             win_ts,
        "elapsed_second": elapsed,
        "hour_utc":       float(hour_utc),
        "move_sigmas":     move_sigmas,
        "vel_2s":         vel_2s,
        "vel_5s":         vel_5s,
        "vel_10s":        vel_10s,
        "acc_4s":         acc_4s,
        "acc_10s":        acc_10s,
        "vel_ratio":      vel_ratio,
        "vel_decay":      vel_decay,
        "vol_10s_log":    vol_10s_log,
        "move_x_elapsed": move_sigmas * elapsed,
        "move_x_vol":     move_sigmas * vol_10s_log,
    })

    return df


# ── Asset dataset builder ─────────────────────────────────────────────────────

def build_asset_dataset(
    asset:         str,
    pm_df:         pd.DataFrame,
    close_series:  pd.Series,
    volume_series: pd.Series,
    sigma:         float,
) -> pd.DataFrame:
    """Build the full per-second dataset for one asset across all resolvable windows."""
    windows      = sorted(pm_df["window_ts"].unique())
    split_idx    = int(len(windows) * TRAIN_FRAC)
    train_set    = set(windows[:split_idx])

    ts_idx = close_series.index.values
    vals   = close_series.values

    all_frames         = []
    skipped_coin       = 0
    skipped_resolution = 0

    for wts in windows:
        lo = int(np.searchsorted(ts_idx, wts,               side="left"))
        hi = int(np.searchsorted(ts_idx, wts + WINDOW_SECS, side="left"))
        if (hi - lo) < MIN_COIN_ROWS:
            skipped_coin += 1
            continue

        window_move = float(vals[hi - 1]) - float(vals[lo])
        pm_window   = pm_df[pm_df["window_ts"] == wts]
        label       = resolve_window(pm_window, window_move)
        if label is None:
            skipped_resolution += 1
            continue

        hour_utc = datetime.fromtimestamp(wts, tz=timezone.utc).hour
        feat_df  = build_window_features(wts, close_series, volume_series, sigma, hour_utc)
        if feat_df.empty:
            skipped_coin += 1
            continue

        feat_df["window_ts"]   = wts
        feat_df["resolved_up"] = int(label)
        feat_df["split"]       = "train" if wts in train_set else "test"
        all_frames.append(feat_df)

    if not all_frames:
        log.warning("%s: no windows produced features", asset)
        return pd.DataFrame()

    out = pd.concat(all_frames, ignore_index=True)
    log.info(
        "%s: %d rows from %d windows  (skipped: %d no-coin, %d unresolved)  "
        "train/test windows: %d/%d",
        asset, len(out),
        len(windows) - skipped_coin - skipped_resolution,
        skipped_coin, skipped_resolution,
        out[out["split"] == "train"]["window_ts"].nunique(),
        out[out["split"] == "test"]["window_ts"].nunique(),
    )
    return out


# ── Model ──────────────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LogisticRegression(max_iter=1000)),
    ])


def train_asset(asset: str, df: pd.DataFrame, features: list[str] = FEATURES) -> dict | None:
    train_df = df[df["split"] == "train"].copy()
    test_df  = df[df["split"] == "test"].copy()

    production = len(test_df) == 0  # production mode: all data in train
    if len(train_df) < 100 or (not production and len(test_df) < 50):
        log.warning("%s: too few rows (train=%d, test=%d) — skipping", asset, len(train_df), len(test_df))
        return None

    X_train = train_df[features].values
    y_train = train_df["resolved_up"].values

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    coefs = pipe.named_steps["model"].coef_[0]

    if production:
        log.info("%s: trained on all %d rows (%d windows)",
                 asset, len(train_df), train_df["window_ts"].nunique())
        return {
            "asset":           asset,
            "pipe":            pipe,
            "features":        features,
            "train_df":        train_df,
            "test_df":         test_df,
            "coefs":           coefs,
            "auc":             None,
            "brier":           None,
            "baseline_wr":     float(y_train.mean()),
            "n_train":         len(train_df),
            "n_test":          0,
            "n_windows_train": train_df["window_ts"].nunique(),
            "n_windows_test":  0,
        }

    X_test = test_df[features].values
    y_test = test_df["resolved_up"].values
    probs  = pipe.predict_proba(X_test)[:, 1]

    try:
        auc = float(roc_auc_score(y_test, probs))
    except ValueError as e:
        log.warning("%s: roc_auc_score failed (%s) — skipping", asset, e)
        return None

    brier       = float(brier_score_loss(y_test, probs))
    baseline_wr = float(y_test.mean())

    test_df = test_df.copy()
    test_df["predicted_prob"] = probs

    log.info("%s: AUC=%.4f  Brier=%.4f  baseline=%.1f%%  train=%d  test=%d",
             asset, auc, brier, baseline_wr * 100, len(train_df), len(test_df))

    return {
        "asset":           asset,
        "pipe":            pipe,
        "features":        features,
        "train_df":        train_df,
        "test_df":         test_df,
        "coefs":           coefs,
        "auc":             auc,
        "brier":           brier,
        "baseline_wr":     baseline_wr,
        "n_train":         len(train_df),
        "n_test":          len(test_df),
        "n_windows_train": train_df["window_ts"].nunique(),
        "n_windows_test":  test_df["window_ts"].nunique(),
    }


# ── Report sections ────────────────────────────────────────────────────────────

def section_metrics(results: list[dict]) -> str:
    rows = []
    for r in results:
        rows.append({
            "asset":          r["asset"],
            "win_train":      r["n_windows_train"],
            "win_test":       r["n_windows_test"],
            "n_train":        r["n_train"],
            "n_test":         r["n_test"],
            "AUC-ROC":        round(r["auc"], 4),
            "Brier":          round(r["brier"], 4),
            "baseline_win%":  f"{r['baseline_wr']*100:.1f}%",
        })
    return pd.DataFrame(rows).to_markdown(index=False)


def section_feature_importance(results: list[dict]) -> str:
    out = []
    for r in results:
        out.append(f"### {r['asset']}\n")
        pairs = sorted(zip(r["features"], r["coefs"]), key=lambda x: x[1], reverse=True)
        top5  = pairs[:5]
        bot5  = pairs[-5:]

        out.append("**Top 5 positive (associated with P(UP) ↑):**\n")
        out.append("| feature | coef |")
        out.append("|---|---:|")
        for feat, coef in top5:
            out.append(f"| {feat} | {coef:+.4f} |")

        out.append("\n**Top 5 negative (associated with P(UP) ↓):**\n")
        out.append("| feature | coef |")
        out.append("|---|---:|")
        for feat, coef in bot5:
            out.append(f"| {feat} | {coef:+.4f} |")
        out.append("")
    return "\n".join(out)


def section_edge_by_decile(results: list[dict], pm_lookup: dict) -> str:
    out = []
    for r in results:
        asset   = r["asset"]
        test_df = r["test_df"].copy()
        test_df["pm_price_up_equiv"] = test_df["ts"].apply(
            lambda t: pm_lookup.get((asset, int(t)))
        )
        valid = test_df[test_df["pm_price_up_equiv"].notna()].copy()
        if valid.empty:
            continue

        out.append(f"### {asset}\n")
        out.append("| decile | avg_pred_prob | avg_mkt_p_up | edge | n |")
        out.append("|---|---:|---:|---:|---:|")

        try:
            valid["decile"] = pd.qcut(valid["predicted_prob"], q=10, labels=False, duplicates="drop")
        except ValueError:
            out.append("_Not enough distinct probability values for decile binning._\n")
            continue

        for decile, grp in valid.groupby("decile"):
            avg_pred = grp["predicted_prob"].mean()
            avg_mkt  = grp["pm_price_up_equiv"].mean()
            edge     = avg_pred - avg_mkt
            out.append(f"| {int(decile)+1} | {avg_pred:.3f} | {avg_mkt:.3f} | {edge:+.3f} | {len(grp)} |")

        out.append("")
    return "\n".join(out)


def section_calibration(results: list[dict]) -> str:
    """
    Calibration table: split predicted_prob into 10 equal-width buckets (0–0.1, 0.1–0.2, …).
    For each bucket show avg predicted prob, actual win rate, and n.
    A well-calibrated model has avg_pred ≈ actual_win_rate (falls on the diagonal).
    """
    BINS = [i / 10 for i in range(11)]  # 0.0, 0.1, ..., 1.0
    CHART_W = 40  # chars wide for the bar chart axis

    out = []
    for r in results:
        test_df = r["test_df"].copy()
        out.append(f"### {r['asset']}\n")

        # collect bucket stats first
        test_df["bucket"] = pd.cut(test_df["predicted_prob"], bins=BINS, include_lowest=True)
        bucket_stats = []
        for bucket, grp in test_df.groupby("bucket", observed=True):
            if grp.empty:
                continue
            avg_pred   = grp["predicted_prob"].mean()
            actual_win = grp["resolved_up"].mean()
            bucket_stats.append((bucket, avg_pred, actual_win, len(grp)))

        # ASCII calibration chart
        # X axis = predicted prob bucket (0.0–1.0)
        # Each row: show where actual win% lands vs the diagonal (perfect calibration)
        out.append("```")
        out.append("Calibration plot  (· = perfect, █ = actual win rate, gap shows miscalibration)")
        out.append(f"{'pred':>6}  {'0%':4}{'':>{CHART_W//2 - 4}}{'50%':3}{'':>{CHART_W//2 - 4}}{'100%'}")
        out.append(f"{'':>6}  {'|'}{'-' * (CHART_W - 1)}|")
        for bucket, avg_pred, actual_win, n in bucket_stats:
            perfect_pos = int(avg_pred * CHART_W)          # where diagonal would be
            actual_pos  = int(actual_win * CHART_W)        # where actual is
            flag        = " ⚠" if abs(avg_pred - actual_win) > 0.05 else ""

            # build the row: place · at perfect_pos, █ at actual_pos
            row = [" "] * CHART_W
            row[min(perfect_pos, CHART_W - 1)] = "·"
            row[min(actual_pos,  CHART_W - 1)] = "█"
            # if both land in same cell, █ wins
            out.append(f"{avg_pred:5.2f}  |{''.join(row)}|  {actual_win*100:.0f}%{flag}")
        out.append(f"{'':>6}  {'|'}{'-' * (CHART_W - 1)}|")
        out.append("```\n")

        # table
        out.append("| prob bucket | avg_pred | actual_win% | n | calibration gap |")
        out.append("|---|---:|---:|---:|---:|")
        for bucket, avg_pred, actual_win, n in bucket_stats:
            gap  = avg_pred - actual_win
            flag = " ⚠" if abs(gap) > 0.05 else ""
            out.append(
                f"| {bucket} | {avg_pred:.3f} | {actual_win*100:.1f}% | {n} | {gap:+.3f}{flag} |"
            )
        out.append("")

    # Aggregate calibration across all assets
    all_df = pd.concat([r["test_df"][["predicted_prob", "resolved_up"]] for r in results], ignore_index=True)
    all_df["bucket"] = pd.cut(all_df["predicted_prob"], bins=BINS, include_lowest=True)
    out.append("### All Assets (aggregated)\n")
    out.append("| prob bucket | avg_pred | actual_win% | n | calibration gap |")
    out.append("|---|---:|---:|---:|---:|")
    for bucket, grp in all_df.groupby("bucket", observed=True):
        if grp.empty:
            continue
        avg_pred   = grp["predicted_prob"].mean()
        actual_win = grp["resolved_up"].mean()
        gap        = avg_pred - actual_win
        flag       = " ⚠" if abs(gap) > 0.05 else ""
        out.append(
            f"| {bucket} | {avg_pred:.3f} | {actual_win*100:.1f}% | {len(grp)} | {gap:+.3f}{flag} |"
        )
    out.append("")

    out.append(
        "_Calibration gap = avg predicted prob − actual win rate. "
        "Near zero = well calibrated. ⚠ = gap > 5% — probabilities in that bucket are misleading._"
    )
    return "\n".join(out)


def section_edge_by_elapsed(results: list[dict]) -> str:
    out = []
    for r in results:
        test_df = r["test_df"].copy()
        out.append(f"### {r['asset']}\n")
        out.append("| window second | avg_pred | win_rate | abs_error | n |")
        out.append("|---|---:|---:|---:|---:|")

        for b in range(0, WINDOW_SECS, 30):
            grp = test_df[(test_df["elapsed_second"] >= b) & (test_df["elapsed_second"] < b + 30)]
            if grp.empty:
                continue
            out.append(
                f"| {b}–{b+30}s | {grp['predicted_prob'].mean():.3f} | "
                f"{grp['resolved_up'].mean():.3f} | "
                f"{abs(grp['predicted_prob'].mean() - grp['resolved_up'].mean()):.3f} | {len(grp)} |"
            )
        out.append("")
    return "\n".join(out)


# ── Trading-focused sections ──────────────────────────────────────────────────

def section_threshold_ev(results: list[dict], pm_lookup: dict) -> str:
    """
    For each asset: sweep edge thresholds 0.05–0.30.
    One trade per window = first second where edge >= threshold.
    Shows n_trades, win%, avg_fill, avg_pnl, total_pnl.
    """
    THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    out = []

    for r in results:
        asset   = r["asset"]
        test_df = r["test_df"].copy()
        test_df["pm_price_up_equiv"] = test_df["ts"].apply(
            lambda t: pm_lookup.get((asset, int(t)))
        )
        test_df["edge"] = test_df["predicted_prob"] - test_df["pm_price_up_equiv"]
        valid = test_df.dropna(subset=["pm_price_up_equiv"]).copy()
        if valid.empty:
            continue

        out.append(f"### {asset}\n")
        out.append("| threshold | n_trades | fill_rate% | win% | avg_fill | avg_pnl | total_pnl |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")

        n_windows = valid["window_ts"].nunique()
        for t in THRESHOLDS:
            # first second in each window where edge >= t
            cands  = valid[valid["edge"] >= t].sort_values("elapsed_second")
            trades = cands.groupby("window_ts").first().reset_index()
            if trades.empty:
                out.append(f"| {t:.2f} | 0 | 0% | — | — | — | — |")
                continue
            fills     = len(trades) / n_windows * 100
            won       = trades["resolved_up"].astype(int)
            fill_p    = trades["pm_price_up_equiv"]
            pnl       = (1.0 - fill_p) * won + (-fill_p) * (1 - won)
            best_flag = ""
            out.append(
                f"| {t:.2f} | {len(trades)} | {fills:.0f}% | {won.mean()*100:.1f}% |"
                f" {fill_p.mean():.3f} | {pnl.mean():+.4f} | {pnl.sum():+.4f}{best_flag} |"
            )

        out.append("")

    # Aggregate EV sweep across all assets
    # Pool all test rows, attach pm_price_up_equiv per asset
    frames = []
    for r in results:
        asset   = r["asset"]
        tdf     = r["test_df"].copy()
        tdf["asset"] = asset
        tdf["pm_price_up_equiv"] = tdf["ts"].apply(lambda t: pm_lookup.get((asset, int(t))))
        tdf["edge"] = tdf["predicted_prob"] - tdf["pm_price_up_equiv"]
        frames.append(tdf.dropna(subset=["pm_price_up_equiv"]))

    if frames:
        all_valid    = pd.concat(frames, ignore_index=True)
        n_windows_total = all_valid.groupby("asset")["window_ts"].nunique().sum()
        out.append("### All Assets (aggregated)\n")
        out.append("| threshold | n_trades | fill_rate% | win% | avg_fill | avg_pnl | total_pnl |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for t in THRESHOLDS:
            cands  = all_valid[all_valid["edge"] >= t].sort_values("elapsed_second")
            trades = cands.groupby(["asset", "window_ts"]).first().reset_index()
            if trades.empty:
                out.append(f"| {t:.2f} | 0 | 0% | — | — | — | — |")
                continue
            fills  = len(trades) / n_windows_total * 100
            won    = trades["resolved_up"].astype(int)
            fill_p = trades["pm_price_up_equiv"]
            pnl    = (1.0 - fill_p) * won + (-fill_p) * (1 - won)
            out.append(
                f"| {t:.2f} | {len(trades)} | {fills:.0f}% | {won.mean()*100:.1f}% |"
                f" {fill_p.mean():.3f} | {pnl.mean():+.4f} | {pnl.sum():+.4f} |"
            )
        out.append("")

    out.append(
        "_fill_rate% = % of test windows where a trade fires at this threshold. "
        "avg_fill = avg Polymarket price paid. avg_pnl per trade assumes $1 stake._"
    )
    return "\n".join(out)


def _timing_summary(trades: "pd.DataFrame") -> str:
    """Return '| n | win% | avg_fill | avg_pnl |' cells for a trades DataFrame."""
    if trades.empty:
        return " — | — | — | — |"
    won    = trades["resolved_up"].astype(int)
    fill_p = trades["pm_price_up_equiv"]
    pnl    = (1.0 - fill_p) * won + (-fill_p) * (1 - won)
    return f" {len(trades)} | {won.mean()*100:.1f}% | {fill_p.mean():.3f} | {pnl.mean():+.4f} |"


def _entry_timing_for_threshold(results: list[dict], pm_lookup: dict, threshold: float) -> list[str]:
    """Return lines for the entry-timing table at a single threshold."""
    out = []
    all_trades_frames = []

    for r in results:
        asset   = r["asset"]
        test_df = r["test_df"].copy()
        test_df["pm_price_up_equiv"] = test_df["ts"].apply(
            lambda t: pm_lookup.get((asset, int(t)))
        )
        test_df["edge"] = test_df["predicted_prob"] - test_df["pm_price_up_equiv"]
        valid = test_df.dropna(subset=["pm_price_up_equiv"])

        cands  = valid[valid["edge"] >= threshold].sort_values("elapsed_second")
        trades = cands.groupby("window_ts").first().reset_index()
        if trades.empty:
            continue

        all_trades_frames.append(trades)

        cands_30  = valid[(valid["edge"] >= threshold) & (valid["elapsed_second"] >= 30)].sort_values("elapsed_second")
        trades_30 = cands_30.groupby("window_ts").first().reset_index()

        out.append(f"#### {asset}\n")
        out.append("| entry window | n (all) | win% | avg_fill | avg_pnl | n (skip <30s) | win% | avg_fill | avg_pnl |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

        for b in range(0, WINDOW_SECS, 30):
            grp    = trades[   (trades["elapsed_second"]    >= b) & (trades["elapsed_second"]    < b + 30)]
            grp_30 = trades_30[(trades_30["elapsed_second"] >= b) & (trades_30["elapsed_second"] < b + 30)]
            if grp.empty and grp_30.empty:
                continue
            out.append(f"| {b}–{b+30}s |{_timing_summary(grp)}{_timing_summary(grp_30)}")

        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        out.append(f"| **all windows** |{_timing_summary(trades)}{_timing_summary(trades_30)}")
        out.append("")

    # Aggregate across all assets
    if all_trades_frames:
        agg_trades_30_frames = []
        for r in results:
            asset   = r["asset"]
            test_df = r["test_df"].copy()
            test_df["pm_price_up_equiv"] = test_df["ts"].apply(
                lambda t: pm_lookup.get((asset, int(t)))
            )
            test_df["edge"] = test_df["predicted_prob"] - test_df["pm_price_up_equiv"]
            valid2 = test_df.dropna(subset=["pm_price_up_equiv"])
            cands_30 = valid2[(valid2["edge"] >= threshold) & (valid2["elapsed_second"] >= 30)].sort_values("elapsed_second")
            t30 = cands_30.groupby("window_ts").first().reset_index()
            if not t30.empty:
                t30["asset"] = asset
                agg_trades_30_frames.append(t30)

        all_trades = pd.concat(all_trades_frames, ignore_index=True)
        all_t30    = pd.concat(agg_trades_30_frames, ignore_index=True) if agg_trades_30_frames else pd.DataFrame()
        out.append("#### All Assets (aggregated)\n")
        out.append("| entry window | n (all) | win% | avg_fill | avg_pnl | n (skip <30s) | win% | avg_fill | avg_pnl |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

        for b in range(0, WINDOW_SECS, 30):
            grp    = all_trades[(all_trades["elapsed_second"] >= b) & (all_trades["elapsed_second"] < b + 30)]
            grp_30 = all_t30[  (all_t30["elapsed_second"]    >= b) & (all_t30["elapsed_second"]    < b + 30)] if not all_t30.empty else pd.DataFrame()
            if grp.empty and grp_30.empty:
                continue
            out.append(f"| {b}–{b+30}s |{_timing_summary(grp)}{_timing_summary(grp_30)}")

        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        out.append(f"| **all windows** |{_timing_summary(all_trades)}{_timing_summary(all_t30)}")
        out.append("")

    return out


def section_entry_timing(results: list[dict], pm_lookup: dict) -> str:
    """
    Entry timing tables at thresholds 0.20, 0.25, 0.30 — how win% and avg_pnl vary
    by when in the window the edge threshold is first crossed.

    'skip <30s' columns re-run first-trade logic restricted to elapsed_second >= 30.
    """
    THRESHOLDS = [0.20, 0.25, 0.30, 0.35]
    out = []

    for t in THRESHOLDS:
        out.append(f"### threshold = {t:.2f}\n")
        out.extend(_entry_timing_for_threshold(results, pm_lookup, t))

    out.append(
        "_'skip <30s' columns don't trade when the threshold was crossed before second 30._"
    )
    return "\n".join(out)


# ── Predictions DataFrame ─────────────────────────────────────────────────────

def build_predictions_df(results: list[dict], pm_lookup: dict) -> pd.DataFrame:
    frames = []
    for r in results:
        asset   = r["asset"]
        test_df = r["test_df"].copy()
        test_df["asset"] = asset
        test_df["pm_price_up_equiv"] = test_df["ts"].apply(
            lambda t: pm_lookup.get((asset, int(t)))
        )
        test_df["edge"] = test_df["predicted_prob"] - test_df["pm_price_up_equiv"]
        cols = ["asset", "window_ts", "ts", "elapsed_second", "resolved_up",
                "predicted_prob", "pm_price_up_equiv", "edge"]
        frames.append(test_df[[c for c in cols if c in test_df.columns]])

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["asset", "window_ts", "ts"])


# ── Interaction feature comparison ────────────────────────────────────────────

def section_interaction_comparison(results_with: list[dict], results_without: list[dict]) -> str:
    """
    Side-by-side AUC, Brier, and calibration gap with vs without interaction features.
    ΔAUC > 0 and ΔBrier < 0 = interactions help.
    """
    without_map = {r["asset"]: r for r in results_without}
    rows = []
    for r in results_with:
        asset = r["asset"]
        rb    = without_map.get(asset)
        rows.append({
            "asset":           asset,
            "AUC (base)":      f"{rb['auc']:.4f}"   if rb else "—",
            "AUC (+ interact)": f"{r['auc']:.4f}",
            "ΔAUC":            f"{r['auc'] - rb['auc']:+.4f}" if rb else "—",
            "Brier (base)":    f"{rb['brier']:.4f}" if rb else "—",
            "Brier (+ interact)": f"{r['brier']:.4f}",
            "ΔBrier":          f"{r['brier'] - rb['brier']:+.4f}" if rb else "—",
        })
    return pd.DataFrame(rows).to_markdown(index=False)


# ── Report builder ─────────────────────────────────────────────────────────────

def build_report(results: list[dict], pm_lookup: dict, generated_at: str,
                 results_base: "list[dict] | None" = None) -> str:
    # top-line summary per asset for the overview
    summary_lines = []
    for r in results:
        test_df = r["test_df"].copy()
        test_df["pm_price_up_equiv"] = test_df["ts"].apply(
            lambda t: pm_lookup.get((r["asset"], int(t)))
        )
        test_df["edge"] = test_df["predicted_prob"] - test_df["pm_price_up_equiv"]
        valid = test_df.dropna(subset=["pm_price_up_equiv"])
        cands = valid[valid["edge"] >= 0.10].sort_values("elapsed_second")
        trades = cands.groupby("window_ts").first().reset_index() if not cands.empty else pd.DataFrame()

        if trades.empty:
            summary_lines.append(f"- **{r['asset']}**: AUC={r['auc']:.3f} — no trades at 0.10 threshold")
            continue

        won    = trades["resolved_up"].astype(int)
        fill_p = trades["pm_price_up_equiv"]
        pnl    = (1.0 - fill_p) * won + (-fill_p) * (1 - won)
        n_win  = valid["window_ts"].nunique()
        summary_lines.append(
            f"- **{r['asset']}**: AUC={r['auc']:.3f} — "
            f"{len(trades)}/{n_win} windows triggered ({len(trades)/n_win*100:.0f}%) — "
            f"win={won.mean()*100:.1f}% — avg_pnl={pnl.mean():+.4f} — total_pnl={pnl.sum():+.4f}"
        )

    lines = [
        "# Model Report: Edge-Based Entry Strategy",
        f"_Generated {generated_at}_",
        "",
        "**Strategy:** At each second of every 5-minute window, compute `edge = predicted_P(UP) − market_P(UP)`.",
        "When `edge ≥ threshold`, buy UP on Polymarket. One entry per window (first qualifying second).",
        "",
        "**Test set only** — final 20% of windows chronologically per asset. No training data used.",
        "",
        "### At-a-glance (threshold = 0.10)",
        "",
        "\n".join(summary_lines),
        "",
        "---",
        "",
        "## 1. Is the Model Trustworthy? — Calibration",
        "",
        "Before using model probabilities as prices, verify they mean what they say.",
        "If the model outputs 0.75, the window should resolve UP ~75% of the time.",
        "**⚠ flagged buckets are unreliable — do not trade based on those probabilities.**",
        "",
        section_calibration(results),
        "",
        "---",
        "",
        "## 2. What Edge Threshold Should I Use? — EV Sweep",
        "",
        "Sweeps thresholds from 5c to 30c. `fill_rate%` = how often a trade fires per session.",
        "`avg_pnl` = expected value per $1 stake. Use this to pick your entry threshold.",
        "",
        section_threshold_ev(results, pm_lookup),
        "",
        "---",
        "",
        "## 3. When in the Window Should I Enter? — Entry Timing",
        "",
        "At thresholds 0.20 / 0.25 / 0.30: does entry timing affect outcome?",
        "Earlier entries have lower fill prices but less model conviction.",
        "",
        section_entry_timing(results, pm_lookup),
        "",
        "---",
        "",
        "## 4. What Drives the Edge? — Feature Importance",
        "",
        "_Standardised logistic regression coefficients. Magnitude = importance, sign = direction._",
        "_Positive coefficient → feature increases P(UP). Strongest features are the most actionable signals._",
        "",
        section_feature_importance(results),
        "",
        "---",
        "",
        "## 5. Model Quality Metrics",
        "",
        "_AUC measures ranking quality (0.5=random, 1.0=perfect). Brier measures probability accuracy (lower=better)._",
        "_Baseline win% ≈ 50% because the model is trained on all window-seconds, not just trigger moments._",
        "",
        section_metrics(results),
        "",
        "---",
        "",
        "## 6. Interaction Feature Value",
        "",
        "_Do `move_x_elapsed` and `move_x_vol` improve predictions over base features alone?_",
        "_ΔAUC > 0 and ΔBrier < 0 = interactions help. If both are near zero, drop them._",
        "",
        section_interaction_comparison(results, results_base) if results_base else "_Baseline not available._",
        "",
        "---",
        "",
        "## Methodology",
        "",
        f"- Features: " + ", ".join(f"`{f}`" for f in FEATURES),
        "- Train/test: chronological 80/20 split at window level (not row level — no leakage)",
        "- `pm_price_up_equiv` = Polymarket `up_price` — market's implied P(UP). Not a feature.",
        "- NaN features (early seconds, missing lookbacks) imputed with training-set median",
        f"- Volume: log1p(sum of last {VOL_LOOKBACK}s coin volume before each second)",
        "- Resolution: last `up_price >= 0.95` → UP, `<= 0.05` → DOWN, else coin direction",
    ]
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train per-asset logistic regression for P(resolved_up)")
    p.add_argument("--prices-dir", default="data/prices",
                   help="Directory containing prices_*.csv (default: %(default)s)")
    p.add_argument("--coin-dir",   default="data/coin_prices",
                   help="Directory containing {SYMBOL}_1s.csv (default: %(default)s)")
    p.add_argument("--out-report", default="data/reports/model_report.md",
                   help="Output markdown report (default: %(default)s)")
    p.add_argument("--out-models", default="data/models",
                   help="Directory to save fitted per-asset pipelines (default: data/models)")
    p.add_argument("--out-csv",    default="data/reports/model_predictions.csv",
                   help="Output predictions CSV (default: %(default)s)")
    p.add_argument("--assets",     nargs="+", default=None,
                   help="Assets to train (default: all)")
    p.add_argument("--production", action="store_true",
                   help="Train on ALL data (no train/test split), save models only — no report or CSV")
    return p.parse_args()


def _load_asset(args: argparse.Namespace, asset: str) -> "tuple | None":
    """Load coin + PM data and compute sigma. Returns (pm_df, close_series, volume_series, sigma) or None."""
    close_series, volume_series = load_coin_series(args.coin_dir, asset)
    if close_series is None:
        log.warning("%s: skipping — no coin data", asset)
        return None
    try:
        pm_df = load_pm_windows(args.prices_dir, asset)
    except FileNotFoundError as e:
        log.error("%s: %s", asset, e)
        return None
    if pm_df.empty:
        log.warning("%s: no PM data — skipping", asset)
        return None
    windows = sorted(pm_df["window_ts"].unique())
    try:
        sigma = compute_sigma(windows, close_series)
    except ValueError as e:
        log.warning("%s: %s — skipping", asset, e)
        return None
    log.info("%s: sigma=%.6g", asset, sigma)
    return pm_df, close_series, volume_series, sigma



def main() -> None:
    args = parse_args()
    os.makedirs(args.out_models, exist_ok=True)

    assets = [a.upper() for a in args.assets] if args.assets else list(ASSET_TO_SYMBOL.keys())
    log.info("Mode: %s  |  Assets: %s", "production" if args.production else "report", assets)

    if args.production:
        # Train on ALL data — no split, no report, no CSV
        for asset in assets:
            log.info("=== %s ===", asset)
            loaded = _load_asset(args, asset)
            if loaded is None:
                continue
            pm_df, close_series, volume_series, sigma = loaded
            df = build_asset_dataset(asset, pm_df, close_series, volume_series, sigma)
            if df.empty:
                continue
            # Fit on everything (override split — mark all rows as train)
            df = df.copy()
            df["split"] = "train"
            result = train_asset(asset, df)
            if result is not None:
                model_path = os.path.join(args.out_models, f"{asset.lower()}.joblib")
                joblib.dump({"pipe": result["pipe"], "features": FEATURES, "sigma": sigma}, model_path)
                log.info("Model → %s  (n=%d)", model_path, result["n_train"])
        return

    # Report mode — train/test split, generate report and predictions CSV
    os.makedirs(os.path.dirname(args.out_report), exist_ok=True)

    results:      list[dict]                   = []
    results_base: list[dict]                   = []
    pm_lookup:    dict[tuple[str, int], float] = {}

    for asset in assets:
        log.info("=== %s ===", asset)
        loaded = _load_asset(args, asset)
        if loaded is None:
            continue
        pm_df, close_series, volume_series, sigma = loaded

        for _, row in pm_df[pm_df["up_price"].notna()].iterrows():
            pm_lookup[(asset, int(row["ts"]))] = float(row["up_price"])

        df = build_asset_dataset(asset, pm_df, close_series, volume_series, sigma)
        if df.empty:
            continue

        result = train_asset(asset, df)
        if result is not None:
            results.append(result)
            model_path = os.path.join(args.out_models, f"{asset.lower()}.joblib")
            joblib.dump({"pipe": result["pipe"], "features": FEATURES, "sigma": sigma}, model_path)
            log.info("Model → %s", model_path)

        # Base model: same data, no interaction features — for section 6 comparison
        result_base = train_asset(asset, df, features=BASE_FEATURES)
        if result_base is not None:
            results_base.append(result_base)

    if not results:
        log.error("No results produced.")
        sys.exit(1)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report       = build_report(results, pm_lookup, generated_at, results_base or None)

    with open(args.out_report, "w") as f:
        f.write(report)
    log.info("Report → %s", args.out_report)

    preds_df = build_predictions_df(results, pm_lookup)
    preds_df.to_csv(args.out_csv, index=False)
    log.info("Predictions → %s  (%d rows)", args.out_csv, len(preds_df))

    print(report)


if __name__ == "__main__":
    main()
