#!/usr/bin/env python3
"""
Per-asset logistic regression model: P(resolved_up) at any second of a window.

Features computed at each second t in each window:
  move_sigmas, elapsed_second, hour_sin, hour_cos,
  vel_2s, vel_5s, vel_10s, acc_4s, acc_10s,
  vel_ratio, vel_decay, acc_positive, vol_10s_log

Resolution: coin-price move direction only (positive=UP, negative=DOWN).
Train/test split: train on coin-only windows, test on windows with market data.

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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

WINDOW_SECS   = 300
MIN_COIN_ROWS = 280
VOL_LOOKBACK  = 10


DEFAULT_THRESHOLD = 0.25
BUY_FEE_RATE = 0.015

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
    "hour_sin",
    "hour_cos",
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

# Extended feature set — structural / regime features added on top of FEATURES
NEW_FEATURES = [
    "dist_high_30",       # distance below 30s rolling high (σ-norm)
    "dist_low_30",        # distance above 30s rolling low (σ-norm)
    "zscore_20",          # price z-score relative to 20s mean/std
    "vol_z_30",           # volume z-score relative to 30s mean/std
    "signed_vol_imb",     # net signed volume imbalance over 10s (range ≈ [-1, 1])
    "trend_str_30",       # net_move / total_path over 30s (1 = clean trend, 0 = chop)
    "vol_expansion",      # rv_10 / rv_30 (>1 = expanding vol / breakout)
    "mom_slope",          # vel_2s − vel_10s: positive = accelerating, negative = fading
    "dir_consistency_10", # fraction of up-ticks in last 10s (range [-1, 1])
    "time_since_flip",    # seconds since last direction flip (trend persistence)
]

EXTENDED_FEATURES = FEATURES + NEW_FEATURES


from skeptic.models.calibration import PlattScaledModel  # noqa: F401 — re-exported for joblib


# ── Data loading ───────────────────────────────────────────────────────────────

def load_pm_windows(prices_dir: str, asset: str) -> pd.DataFrame:
    """Load all prices_*.csv files, filter to asset. Returns ts, window_ts, up_ask, dn_ask."""
    files = sorted(Path(prices_dir).glob("prices_*.csv"))
    if not files:
        raise FileNotFoundError(f"No prices_*.csv in {prices_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f, usecols=["ts", "window_ts", "asset", "up_ask", "dn_ask"])
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

EWMA_LAMBDA  = 0.95   # decay factor — must match executor and threshold_edge
EWMA_WARMUP  = 20     # windows to skip before the estimate is considered reliable


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


def compute_ewma_sigmas(
    windows_sorted: list[int],
    close_series: pd.Series,
    lambda_: float = EWMA_LAMBDA,
) -> dict[int, float]:
    """
    Walk-forward EWMA variance estimator (RiskMetrics-style).

    For window t, returns sigma_t estimated ONLY from windows 0…t-1 —
    no lookahead. The first EWMA_WARMUP windows after the seed are excluded
    so the estimate has time to stabilise.

        ewma_var_t = λ·ewma_var_{t-1} + (1-λ)·move_{t-1}²
        sigma_t    = sqrt(ewma_var_t)

    Returns {window_ts: sigma} for windows that have a valid estimate.
    """
    ts_idx   = close_series.index.values
    vals     = close_series.values
    ewma_var: float | None = None
    result: dict[int, float] = {}

    for wts in windows_sorted:
        lo = int(np.searchsorted(ts_idx, wts,               side="left"))
        hi = int(np.searchsorted(ts_idx, wts + WINDOW_SECS, side="left"))
        if (hi - lo) < MIN_COIN_ROWS:
            continue
        move = float(vals[hi - 1]) - float(vals[lo])

        # Sigma available to USE this window = what was known before it started
        if ewma_var is not None:
            result[wts] = float(np.sqrt(ewma_var))

        # Update EWMA with this window's realised move
        if ewma_var is None:
            ewma_var = move ** 2
        else:
            ewma_var = lambda_ * ewma_var + (1.0 - lambda_) * move ** 2

    # Drop the first EWMA_WARMUP windows (estimate not yet stable)
    keys = list(result.keys())
    for wts in keys[:EWMA_WARMUP]:
        del result[wts]

    return result


# ── Resolution ─────────────────────────────────────────────────────────────────

def resolve_window(window_move: float) -> bool | None:
    """Resolve window direction from coin price move only."""
    if window_move > 0:
        return True
    if window_move < 0:
        return False
    return None


def build_coin_windows(close_series: pd.Series) -> list[int]:
    """Build window start timestamps aligned to real 5-minute UTC boundaries."""
    ts_idx = close_series.index.values
    if len(ts_idx) == 0:
        return []
    first_ts = int(ts_idx[0])
    last_ts  = int(ts_idx[-1])

    # Align to real 5-minute boundaries (epoch multiples of WINDOW_SECS)
    start = (first_ts // WINDOW_SECS) * WINDOW_SECS
    if first_ts % WINDOW_SECS != 0:
        start += WINDOW_SECS
    end = (last_ts // WINDOW_SECS) * WINDOW_SECS

    if start > end:
        return []
    return list(range(start, end + 1, WINDOW_SECS))


# ── Per-window feature computation ────────────────────────────────────────────

def build_window_features(
    window_ts:     int,
    close_series:  pd.Series,
    volume_series: pd.Series,
    sigma:         float,
    hour_utc:      int,  # kept as param so callers pass the same value; converted internally
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

    # ── Extended features ─────────────────────────────────────────────────────
    _sw = np.lib.stride_tricks.sliding_window_view  # alias

    # Tick-level direction arrays (shared across several features)
    price_diffs    = np.empty(n)
    price_diffs[0] = 0.0
    price_diffs[1:] = np.diff(win_price)
    tick_sign = np.sign(price_diffs)
    abs_diffs = np.abs(price_diffs)

    # 1 & 2: Distance from 30s rolling high / low (σ-normalised; ≤0 and ≥0 resp.)
    dist_high_30 = np.full(n, np.nan)
    dist_low_30  = np.full(n, np.nan)
    if n > 30:
        sw_p30 = _sw(win_price, 30)                              # (n-29, 30)
        dist_high_30[30:n] = (win_price[30:n] - sw_p30[:n-30].max(axis=1)) / sigma
        dist_low_30[30:n]  = (win_price[30:n] - sw_p30[:n-30].min(axis=1)) / sigma
    else:
        sw_p30 = None

    # 3: Z-score vs 20s local mean/std
    zscore_20 = np.full(n, np.nan)
    if n > 20:
        sw_p20 = _sw(win_price, 20)                              # (n-19, 20)
        mu20   = sw_p20[:n-20].mean(axis=1)
        sd20   = sw_p20[:n-20].std(axis=1) + 1e-8
        zscore_20[20:n] = (win_price[20:n] - mu20) / sd20

    # 4: Volume z-score vs 30s baseline
    vol_z_30 = np.full(n, np.nan)
    if n > 30:
        sw_v30  = _sw(win_vol, 30)                               # (n-29, 30)
        mu_v30  = sw_v30[:n-30].mean(axis=1)
        sd_v30  = sw_v30[:n-30].std(axis=1) + 1e-8
        vol_z_30[30:n] = (win_vol[30:n] - mu_v30) / sd_v30

    # 5 & 9: Shared 10s sliding windows (volume + sign)
    signed_vol_imb     = np.full(n, np.nan)
    dir_consistency_10 = np.full(n, np.nan)
    if n > 9:
        sw_v10   = _sw(win_vol,   10)                            # (n-9, 10)
        sw_sgn10 = _sw(tick_sign, 10)                            # (n-9, 10)
        denom_v  = sw_v10.sum(axis=1) + 1e-8
        signed_vol_imb[9:n]     = ((sw_v10 * sw_sgn10).sum(axis=1) / denom_v)[:n-9]
        dir_consistency_10[9:n] = sw_sgn10[:n-9].sum(axis=1) / 10.0

    # 6: Trend strength — net displacement / total path over 30s
    trend_str_30 = np.full(n, np.nan)
    if n > 30:
        sw_abs30     = _sw(abs_diffs, 30)                        # (n-29, 30)
        net_moves    = np.abs(win_price[30:n] - win_price[:n-30])
        total_path   = sw_abs30[1:n-29].sum(axis=1) + 1e-8      # abs_diffs[i-29:i+1]
        trend_str_30[30:n] = net_moves / total_path

    # 7: Vol expansion — short-term rv vs longer-term rv
    vol_expansion = np.full(n, np.nan)
    if n > 30:
        sw_p10 = _sw(win_price, 10)                              # (n-9, 10)
        rv10   = sw_p10[:n-10].std(axis=1)                       # shape (n-10,)
        rv30   = (sw_p30 if sw_p30 is not None else _sw(win_price, 30))[:n-30].std(axis=1)  # (n-30,)
        vol_expansion[30:n] = rv10[20:n-10] / (rv30[:n-30] + 1e-8)

    # 8: Momentum slope — recent acceleration in σ units
    mom_slope = vel_2s - vel_10s   # NaN propagates where either is NaN

    # 10: Time since last direction flip (seconds; trend persistence signal)
    time_since_flip = np.empty(n)
    time_since_flip[0] = 0.0
    last_flip = 0
    prev_s    = 0
    for i in range(1, n):
        s = int(tick_sign[i])
        if s != 0:
            if prev_s != 0 and s != prev_s:
                last_flip = i
            prev_s = s
        time_since_flip[i] = float(i - last_flip)

    _hour_rad = hour_utc * (2 * np.pi / 24)
    df = pd.DataFrame({
        "ts":               win_ts,
        "elapsed_second":   elapsed,
        "hour_sin":         np.sin(_hour_rad),
        "hour_cos":         np.cos(_hour_rad),
        "move_sigmas":      move_sigmas,
        "vel_2s":           vel_2s,
        "vel_5s":           vel_5s,
        "vel_10s":          vel_10s,
        "acc_4s":           acc_4s,
        "acc_10s":          acc_10s,
        "vel_ratio":        vel_ratio,
        "vel_decay":        vel_decay,
        "vol_10s_log":      vol_10s_log,
        "move_x_elapsed":   move_sigmas * elapsed,
        "move_x_vol":       move_sigmas * vol_10s_log,
        # Extended features
        "dist_high_30":       dist_high_30,
        "dist_low_30":        dist_low_30,
        "zscore_20":          zscore_20,
        "vol_z_30":           vol_z_30,
        "signed_vol_imb":     signed_vol_imb,
        "trend_str_30":       trend_str_30,
        "vol_expansion":      vol_expansion,
        "mom_slope":          mom_slope,
        "dir_consistency_10": dir_consistency_10,
        "time_since_flip":    time_since_flip,
    })

    return df


# ── Asset dataset builder ─────────────────────────────────────────────────────

def build_asset_dataset(
    asset:         str,
    pm_df:         pd.DataFrame,
    close_series:  pd.Series,
    volume_series: pd.Series,
    sigma:         float,  # kept for vol_ratio / report sections; features use EWMA sigma
) -> pd.DataFrame:
    """
    Build the full per-second dataset for one asset across all resolvable windows.

    Features are normalised by a walk-forward EWMA sigma (λ=0.95), matching what
    the live executor does.  Windows without a valid EWMA estimate (warmup period)
    are excluded.  `sigma` (global static) is still used for vol-ratio sections.
    """
    windows = build_coin_windows(close_series)
    market_windows = set(pm_df["window_ts"].unique()) if not pm_df.empty else set()

    # Prevent time leakage: keep test windows as market windows, and only train
    # on windows strictly before the first test window.
    if market_windows:
        first_test_window = min(market_windows)
        test_set = {w for w in windows if w in market_windows}
        train_set = {w for w in windows if w < first_test_window and w not in market_windows}
    else:
        test_set = set()
        train_set = set(windows)

    # Walk-forward EWMA sigma: computed over ALL coin windows (dense signal)
    ewma_sigmas = compute_ewma_sigmas(windows, close_series)
    log.info(
        "%s: EWMA σ available for %d/%d windows  (warmup excluded: %d)",
        asset, len(ewma_sigmas), len(windows), len(windows) - len(ewma_sigmas),
    )

    ts_idx = close_series.index.values
    vals   = close_series.values

    all_frames         = []
    skipped_coin       = 0
    skipped_resolution = 0
    skipped_ewma       = 0

    for wts in windows:
        lo = int(np.searchsorted(ts_idx, wts,               side="left"))
        hi = int(np.searchsorted(ts_idx, wts + WINDOW_SECS, side="left"))
        if (hi - lo) < MIN_COIN_ROWS:
            skipped_coin += 1
            continue

        window_move = float(vals[hi - 1]) - float(vals[lo])
        label       = resolve_window(window_move)
        if label is None:
            skipped_resolution += 1
            continue

        win_sigma = ewma_sigmas.get(wts)
        if win_sigma is None or win_sigma <= 0:
            skipped_ewma += 1
            continue

        hour_utc = datetime.fromtimestamp(wts, tz=timezone.utc).hour
        feat_df  = build_window_features(wts, close_series, volume_series, win_sigma, hour_utc)
        if feat_df.empty:
            skipped_coin += 1
            continue

        feat_df["window_ts"]   = wts
        feat_df["ewma_sigma"]  = win_sigma
        feat_df["resolved_up"] = int(label)
        if wts in test_set:
            feat_df["split"] = "test"
        elif wts in train_set:
            feat_df["split"] = "train"
        else:
            # Exclude windows that would leak future information into training.
            continue
        all_frames.append(feat_df)

    if not all_frames:
        log.warning("%s: no windows produced features", asset)
        return pd.DataFrame()

    out = pd.concat(all_frames, ignore_index=True)
    log.info(
        "%s: %d rows from %d windows  "
        "(skipped: %d no-coin, %d unresolved, %d ewma-warmup)  "
        "train/test windows: %d/%d",
        asset, len(out),
        len(windows) - skipped_coin - skipped_resolution - skipped_ewma,
        skipped_coin, skipped_resolution, skipped_ewma,
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


def train_asset_logreg(asset: str, df: pd.DataFrame, features: list[str] = FEATURES) -> dict | None:
    train_df = df[df["split"] == "train"].copy()
    test_df  = df[df["split"] == "test"].copy()

    production = len(test_df) == 0  # production mode: all data in train
    if len(train_df) < 100 or (not production and len(test_df) < 50):
        log.warning("%s: too few rows (train=%d, test=%d) — skipping", asset, len(train_df), len(test_df))
        return None

    # Split train into fit (first 80% of windows) + calibration (last 20%)
    # Uses window-level time order so calibration windows are always later than fit windows.
    train_windows = sorted(train_df["window_ts"].unique())
    n_cal_windows = max(1, int(len(train_windows) * 0.20))
    cal_window_set = set(train_windows[-n_cal_windows:])
    fit_window_set = set(train_windows[:-n_cal_windows])

    fit_df = train_df[train_df["window_ts"].isin(fit_window_set)]
    cal_df = train_df[train_df["window_ts"].isin(cal_window_set)]

    X_fit = fit_df[features].copy()
    y_fit = fit_df["resolved_up"].values
    X_cal = cal_df[features].copy()
    y_cal = cal_df["resolved_up"].values

    pipe = build_pipeline()
    pipe.fit(X_fit, y_fit)
    coefs = pipe.named_steps["model"].coef_[0]

    # Platt scaling: fit logistic regression on raw probabilities from the cal set
    p_cal = pipe.predict_proba(X_cal)[:, 1].reshape(-1, 1)
    platt = LogisticRegression()
    platt.fit(p_cal, np.asarray(y_cal, dtype=int))
    calibrated = PlattScaledModel(pipe, platt)
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(p_cal.ravel(), np.asarray(y_cal, dtype=int))

    if production:
        log.info("%s: trained on %d rows (%d fit windows, %d cal windows)",
                 asset, len(train_df), len(fit_window_set), len(cal_window_set))
        return {
            "asset":           asset,
            "pipe":            calibrated,
            "features":        features,
            "train_df":        train_df,
            "test_df":         test_df,
            "coefs":           coefs,
            "auc":             None,
            "brier":           None,
            "isotonic_auc":    None,
            "isotonic_brier":  None,
            "baseline_wr":     float(y_fit.mean()),
            "n_train":         len(train_df),
            "n_test":          0,
            "n_windows_train": train_df["window_ts"].nunique(),
            "n_windows_test":  0,
        }

    X_test = test_df[features].copy()
    y_test = test_df["resolved_up"].values
    probs_raw  = pipe.predict_proba(X_test)[:, 1]
    probs      = calibrated.predict_proba(X_test)[:, 1]
    probs_iso  = isotonic.predict(probs_raw)

    try:
        auc = float(roc_auc_score(y_test, probs))
    except ValueError as e:
        log.warning("%s: roc_auc_score failed (%s) — skipping", asset, e)
        return None

    brier_raw   = float(brier_score_loss(y_test, probs_raw))
    brier       = float(brier_score_loss(y_test, probs))
    brier_iso   = float(brier_score_loss(y_test, probs_iso))
    auc_iso     = float(roc_auc_score(y_test, probs_iso))
    baseline_wr = float(y_test.mean())

    test_df = test_df.copy()
    test_df["predicted_prob"] = probs

    log.info(
        "%s: AUC=%.4f  Brier=%.4f→%.4f (Platt)  baseline=%.1f%%  fit=%d  cal=%d  test=%d",
        asset, auc, brier_raw, brier, baseline_wr * 100,
        len(fit_df), len(cal_df), len(test_df),
    )

    return {
        "asset":           asset,
        "pipe":            calibrated,
        "features":        features,
        "train_df":        train_df,
        "test_df":         test_df,
        "coefs":           coefs,
        "auc":             auc,
        "brier":           brier,
        "isotonic_auc":    auc_iso,
        "isotonic_brier":  brier_iso,
        "baseline_wr":     baseline_wr,
        "n_train":         len(train_df),
        "n_test":          len(test_df),
        "n_windows_train": train_df["window_ts"].nunique(),
        "n_windows_test":  test_df["window_ts"].nunique(),
    }

# ── LightGBM primary model ────────────────────────────────────────────────────

def train_asset_lgb(asset: str, df: pd.DataFrame, features: list[str] = FEATURES) -> dict | None:
    """
    LightGBM + Platt calibration — primary production model.
    Returns the same dict shape as train_asset_logreg so all report sections work unchanged.
    """
    if not _HAS_LGB:
        log.warning("%s: LightGBM not installed — cannot train LGB model", asset)
        return None
    assert _HAS_LGB

    train_df = df[df["split"] == "train"].copy()
    test_df  = df[df["split"] == "test"].copy()

    production = len(test_df) == 0
    if len(train_df) < 100 or (not production and len(test_df) < 50):
        log.warning("%s: too few rows (train=%d, test=%d) — skipping", asset, len(train_df), len(test_df))
        return None

    train_windows = sorted(train_df["window_ts"].unique())
    n_cal_windows = max(1, int(len(train_windows) * 0.20))
    cal_window_set = set(train_windows[-n_cal_windows:])
    fit_window_set = set(train_windows[:-n_cal_windows])

    fit_df = train_df[train_df["window_ts"].isin(fit_window_set)]
    cal_df = train_df[train_df["window_ts"].isin(cal_window_set)]

    X_fit = fit_df[features].copy()
    y_fit = np.asarray(fit_df["resolved_up"], dtype=int)
    X_cal = cal_df[features].copy()
    y_cal = np.asarray(cal_df["resolved_up"], dtype=int)

    pipe = _make_gbm_pipeline(lgb.LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    ))
    pipe.fit(X_fit, y_fit)

    # Platt calibration on held-out cal windows
    p_cal = pipe.predict_proba(X_cal)[:, 1].reshape(-1, 1)
    platt = LogisticRegression()
    platt.fit(p_cal, np.asarray(y_cal, dtype=int))
    calibrated = PlattScaledModel(pipe, platt)
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(p_cal.ravel(), np.asarray(y_cal, dtype=int))

    # Feature importances (gain-based — more interpretable than split count)
    estimator = pipe.named_steps["model"]
    try:
        imp_vals = estimator.booster_.feature_importance(importance_type="gain")
    except Exception:
        imp_vals = estimator.feature_importances_
    feature_importances = dict(zip(features, imp_vals.tolist()))

    if production:
        log.info("%s LGB: trained on %d rows (%d fit windows, %d cal windows)",
                 asset, len(train_df), len(fit_window_set), len(cal_window_set))
        return {
            "asset":              asset,
            "pipe":               calibrated,
            "features":           features,
            "train_df":           train_df,
            "test_df":            test_df,
            "feature_importances": feature_importances,
            "auc":                None,
            "brier":              None,
            "brier_raw":          None,
            "isotonic_auc":       None,
            "isotonic_brier":     None,
            "baseline_wr":        float(y_fit.mean()),
            "n_train":            len(train_df),
            "n_test":             0,
            "n_windows_train":    train_df["window_ts"].nunique(),
            "n_windows_test":     0,
        }

    X_test = test_df[features].copy()
    y_test = np.asarray(test_df["resolved_up"], dtype=int)
    probs_raw = pipe.predict_proba(X_test)[:, 1]
    probs     = calibrated.predict_proba(X_test)[:, 1]
    probs_iso = isotonic.predict(probs_raw)

    try:
        auc = float(roc_auc_score(y_test, probs))
    except ValueError as e:
        log.warning("%s LGB: roc_auc_score failed (%s) — skipping", asset, e)
        return None

    brier_raw   = float(brier_score_loss(y_test, probs_raw))
    brier       = float(brier_score_loss(y_test, probs))
    brier_iso   = float(brier_score_loss(y_test, probs_iso))
    auc_iso     = float(roc_auc_score(y_test, probs_iso))
    baseline_wr = float(y_test.mean())

    test_df = test_df.copy()
    test_df["predicted_prob"] = probs

    log.info(
        "%s LGB: AUC=%.4f  Brier=%.4f→%.4f (Platt)  baseline=%.1f%%  fit=%d  cal=%d  test=%d",
        asset, auc, brier_raw, brier, baseline_wr * 100,
        len(fit_df), len(cal_df), len(test_df),
    )

    return {
        "asset":              asset,
        "pipe":               calibrated,
        "features":           features,
        "train_df":           train_df,
        "test_df":            test_df,
        "feature_importances": feature_importances,
        "auc":                auc,
        "brier":              brier,
        "brier_raw":          brier_raw,
        "isotonic_auc":       auc_iso,
        "isotonic_brier":     brier_iso,
        "baseline_wr":        baseline_wr,
        "n_train":            len(train_df),
        "n_test":             len(test_df),
        "n_windows_train":    train_df["window_ts"].nunique(),
        "n_windows_test":     test_df["window_ts"].nunique(),
    }


# ── GBM pipeline helper ───────────────────────────────────────────────────────

def _make_gbm_pipeline(model) -> Pipeline:
    """Wrap a GBM estimator in an imputer pipeline (no scaling needed for trees)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   model),
    ])


def section_extended_features(
    base_results: list[dict],
    extended_results: "dict[str, dict]",
    pm_lookup: dict,
    dn_lookup: dict,
    threshold: float = DEFAULT_THRESHOLD,
) -> str:
    """
    Compare base LGB (FEATURES) vs extended LGB (EXTENDED_FEATURES) on AUC, Brier, and trading EV.
    Shows whether the 10 new structural/regime features add predictive value.
    """
    out = [
        f"Base LGB uses {len(FEATURES)} features. Extended LGB adds {len(NEW_FEATURES)} structural/regime features "
        f"({len(EXTENDED_FEATURES)} total). Both use identical train/test splits and Platt calibration. "
        f"threshold={threshold}.\n",
        "### AUC & Calibration",
        "",
        "| asset | Base AUC | Ext AUC | ΔAUC | Base Brier | Ext Brier | ΔBrier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for r in base_results:
        asset = r["asset"]
        ext   = extended_results.get(asset)
        if not ext:
            out.append(f"| {asset} | {r['auc']:.4f} | — | — | {r['brier']:.4f} | — | — |")
            continue
        d_auc   = ext["auc"]   - r["auc"]
        d_brier = ext["brier"] - r["brier"]
        flag_auc   = " ✓" if d_auc   > 0    else (" ✗" if d_auc   < 0    else "")
        flag_brier = " ✓" if d_brier < 0    else (" ✗" if d_brier > 0    else "")
        out.append(
            f"| {asset} | {r['auc']:.4f} | {ext['auc']:.4f} | {d_auc:+.4f}{flag_auc} "
            f"| {r['brier']:.4f} | {ext['brier']:.4f} | {d_brier:+.4f}{flag_brier} |"
        )

    out.append("")
    out.append("_✓ = extended is better (AUC higher, Brier lower). ✗ = worse._")
    out.append("")
    out.append("### Trading EV (avg_pnl at threshold)")
    out.append("")
    out.append("| asset | model | n_trades | win% | avg_pnl | total_pnl |")
    out.append("|---|---|---:|---:|---:|---:|")

    def _ev_row(label: str, asset: str, test_df: pd.DataFrame) -> str:
        valid  = _augment_both_sides(test_df.copy(), asset, pm_lookup, dn_lookup)
        cands  = valid[valid["edge"] >= threshold].sort_values("elapsed_second")
        trades = cands.groupby("window_ts").first().reset_index() if not cands.empty else pd.DataFrame()
        if trades.empty:
            return f"| {asset} | {label} | 0 | — | — | — |"
        won    = trades["effective_won"]
        fill_p = trades["pm_fill_price"].fillna(trades["pm_price_signal"])
        pnl    = _pnl_per_trade(fill_p, won)
        return (
            f"| {asset} | {label} | {len(trades)} | {won.mean()*100:.1f}% "
            f"| {pnl.mean():+.4f} | {pnl.sum():+.4f} |"
        )

    for r in base_results:
        asset = r["asset"]
        out.append(_ev_row("Base LGB", asset, r["test_df"]))
        ext = extended_results.get(asset)
        if ext and not ext["test_df"].empty:
            out.append(_ev_row("Extended LGB", asset, ext["test_df"]))

    out.append("")
    return "\n".join(out)


# ── Feature ablation ──────────────────────────────────────────────────────────

def _lgb_auc_fast(df: pd.DataFrame, features: list[str]) -> float | None:
    """
    Train LGB (100 trees, no Platt) on a feature subset, return test AUC.
    Used for ablation comparisons where speed matters and Platt doesn't affect AUC.
    """
    if not _HAS_LGB or not features:
        return None
    assert _HAS_LGB  # narrows type for linter
    train_df = df[df["split"] == "train"]
    test_df  = df[df["split"] == "test"]
    if len(train_df) < 100 or len(test_df) < 50:
        return None

    # Use all train windows for fitting (no cal split needed — no Platt)
    X_fit  = train_df[features].to_numpy(dtype=float)
    y_fit  = train_df["resolved_up"].to_numpy(dtype=int)
    X_test = test_df[features].to_numpy(dtype=float)
    y_test = test_df["resolved_up"].to_numpy(dtype=int)

    try:
        imp      = SimpleImputer(strategy="median")
        X_fit_i  = imp.fit_transform(X_fit)
        X_test_i = imp.transform(X_test)
        model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        model.fit(X_fit_i, y_fit)
        probs = np.asarray(model.predict_proba(X_test_i))[:, 1]
        return float(roc_auc_score(y_test, probs))
    except Exception as e:
        log.debug("Ablation LGB failed: %s", e)
        return None


def section_feature_ablation(asset: str, df: pd.DataFrame,
                              all_features: list[str] | None = None) -> str:
    """
    Three-pass feature ablation:
      1. Single-feature AUC — which features have standalone signal
      2. Leave-one-out AUC — which features contribute given all others
      3. Forward greedy selection — optimal feature-addition ordering
    Uses LGB 100 trees (fast; AUC is comparable, Platt doesn't change ranking).
    """
    if all_features is None:
        all_features = EXTENDED_FEATURES
    if not _HAS_LGB:
        return "_LightGBM not installed — ablation skipped._"

    n = len(all_features)
    out: list[str] = [
        f"**Asset: {asset}** | {n} features | LGB 100 trees (fast mode, no Platt)\n"
    ]

    log.info("%s ablation: baseline (%d features) …", asset, n)
    baseline = _lgb_auc_fast(df, all_features)
    if baseline is None:
        return "_Ablation failed: could not train baseline model._"
    out.append(f"**Full-model baseline AUC ({n} features): {baseline:.4f}**\n")

    # ── 1. Single-feature AUC ──────────────────────────────────────────────
    log.info("%s ablation: single-feature pass (%d models) …", asset, n)
    out += [
        "### 1. Single-Feature AUC",
        "_Each feature trained alone. Reveals standalone predictive signal._\n",
        "| feature | AUC | Δ vs baseline |",
        "|---|---:|---:|",
    ]
    single: list[tuple[str, float]] = []
    for f in all_features:
        auc = _lgb_auc_fast(df, [f])
        if auc is not None:
            single.append((f, auc))
    for f, auc in sorted(single, key=lambda x: x[1], reverse=True):
        out.append(f"| `{f}` | {auc:.4f} | {auc - baseline:+.4f} |")
    out.append("")

    # ── 2. Leave-one-out ──────────────────────────────────────────────────
    log.info("%s ablation: leave-one-out pass (%d models) …", asset, n)
    out += [
        "### 2. Leave-One-Out AUC",
        "_Remove one feature, train on the rest. "
        "ΔAUC < 0 = feature was helping (keep). ΔAUC > 0 = feature was hurting (consider dropping)._\n",
        "| feature removed | AUC | ΔAUC | verdict |",
        "|---|---:|---:|---|",
    ]
    loo: list[tuple[str, float, float]] = []
    for f in all_features:
        auc = _lgb_auc_fast(df, [x for x in all_features if x != f])
        if auc is not None:
            loo.append((f, auc, auc - baseline))
    for f, auc, d in sorted(loo, key=lambda x: x[2]):
        verdict = "keep (hurts when removed)" if d < -0.001 else ("drop (helps when removed)" if d > 0.001 else "neutral")
        out.append(f"| `{f}` | {auc:.4f} | {d:+.4f} | {verdict} |")
    out.append("")

    # ── 3. Forward greedy selection ───────────────────────────────────────
    MAX_ROUNDS = min(n, 12)
    log.info("%s ablation: forward greedy (up to %d rounds) …", asset, MAX_ROUNDS)
    out += [
        "### 3. Forward Greedy Selection",
        f"_Start empty, add the feature that most improves AUC each round "
        f"(up to {MAX_ROUNDS} rounds). Stops early if AUC gain < 0.0002._\n",
        "| step | feature added | AUC | Δ from prev |",
        "|---|---|---:|---:|",
    ]
    remaining = list(all_features)
    selected:  list[str] = []
    prev_auc = 0.5
    for step in range(MAX_ROUNDS):
        best_f, best_auc = None, -1.0
        for f in remaining:
            auc = _lgb_auc_fast(df, selected + [f])
            if auc is not None and auc > best_auc:
                best_auc, best_f = auc, f
        if best_f is None:
            break
        d = best_auc - prev_auc
        selected.append(best_f)
        remaining.remove(best_f)
        out.append(f"| {step + 1} | `{best_f}` | {best_auc:.4f} | {d:+.4f} |")
        prev_auc = best_auc
        if d < 0.0002 and step >= 3:
            out.append("| — | _(converged — no further gain)_ | — | — |")
            break

    out.append("")
    out.append(
        f"_Greedy-optimal {len(selected)}-feature set: "
        + ", ".join(f"`{f}`" for f in selected) + "_"
    )
    out.append("")
    return "\n".join(out)


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
    """Feature importance from LightGBM (gain) or LogReg coefficients, depending on what's in the result."""
    out = []
    for r in results:
        out.append(f"### {r['asset']}\n")

        if "feature_importances" in r and r["feature_importances"]:
            # LightGBM: gain-based importance
            imp   = r["feature_importances"]
            pairs = sorted(imp.items(), key=lambda x: x[1], reverse=True)
            top8  = pairs[:8]
            out.append("**Top 8 by gain (higher = more predictive):**\n")
            out.append("| feature | gain |")
            out.append("|---|---:|")
            for feat, val in top8:
                out.append(f"| {feat} | {val:.0f} |")
        elif "coefs" in r and r["coefs"] is not None:
            # LogReg: signed coefficients
            pairs = sorted(zip(r["features"], r["coefs"]), key=lambda x: x[1], reverse=True)
            top5  = pairs[:5]
            bot5  = pairs[-5:]
            out.append("**Top 5 positive (P(UP) ↑):**\n")
            out.append("| feature | coef |")
            out.append("|---|---:|")
            for feat, coef in top5:
                out.append(f"| {feat} | {coef:+.4f} |")
            out.append("\n**Top 5 negative (P(UP) ↓):**\n")
            out.append("| feature | coef |")
            out.append("|---|---:|")
            for feat, coef in bot5:
                out.append(f"| {feat} | {coef:+.4f} |")

        out.append("")
    return "\n".join(out)



def section_calibration(results: list[dict]) -> str:
    """
    Calibration table: split predicted_prob into 10 equal-width buckets (0–0.1, 0.1–0.2, …).
    For each bucket show avg predicted prob, actual win rate, and n.
    A well-calibrated model has avg_pred ≈ actual_win_rate (falls on the diagonal).
    """
    BINS = [i / 10 for i in range(11)]  # 0.0, 0.1, ..., 1.0
    out = []

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


def section_calibration_method_compare(results: list[dict]) -> str:
    """Compare Platt (active) vs isotonic (evaluation-only) calibration on held-out test data."""
    rows = []
    for r in results:
        iso_auc = r.get("isotonic_auc")
        iso_brier = r.get("isotonic_brier")
        if iso_auc is None or iso_brier is None:
            continue
        platt_auc = float(r["auc"])
        platt_brier = float(r["brier"])
        rows.append({
            "asset": r["asset"],
            "Platt AUC": f"{platt_auc:.4f}",
            "Iso AUC": f"{float(iso_auc):.4f}",
            "ΔAUC (Iso-Platt)": f"{float(iso_auc) - platt_auc:+.4f}",
            "Platt Brier": f"{platt_brier:.4f}",
            "Iso Brier": f"{float(iso_brier):.4f}",
            "ΔBrier (Iso-Platt)": f"{float(iso_brier) - platt_brier:+.4f}",
        })

    if not rows:
        return "_Calibration method comparison unavailable (no held-out test rows)._"
    return pd.DataFrame(rows).to_markdown(index=False)


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

def _fill_with_fee(fill_price: pd.Series) -> pd.Series:
    """Effective entry cost per share after applying the buy fee."""
    return fill_price * (1.0 + BUY_FEE_RATE)


def _pnl_per_trade(fill_price: pd.Series, won: pd.Series) -> pd.Series:
    """Net PnL per trade (per share), accounting for buy fee in entry cost."""
    return won.astype(float) - _fill_with_fee(fill_price)

def section_threshold_ev(results: list[dict], pm_lookup: dict, dn_lookup: dict) -> str:
    """
    For each asset: sweep edge thresholds 0.05–0.30.
    One trade per window = first second where best-side edge >= threshold.
    Shows n_trades, win%, avg_fill, avg_pnl, total_pnl.
    """
    THRESHOLDS = [0.15, 0.20, 0.25]
    out = []

    for r in results:
        asset   = r["asset"]
        valid   = _augment_both_sides(r["test_df"].copy(), asset, pm_lookup, dn_lookup)
        if valid.empty:
            continue

        out.append(f"### {asset}\n")
        out.append("| threshold | n_trades | fill_rate% | win% | avg_fill | avg_pnl | total_pnl |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")

        n_windows = valid["window_ts"].nunique()
        for t in THRESHOLDS:
            cands  = valid[valid["edge"] >= t].sort_values("elapsed_second")
            trades = cands.groupby("window_ts").first().reset_index()
            if trades.empty:
                out.append(f"| {t:.2f} | 0 | 0% | — | — | — | — |")
                continue
            fills  = len(trades) / n_windows * 100
            won    = trades["effective_won"]
            fill_p = trades["pm_fill_price"].fillna(trades["pm_price_signal"])
            pnl    = _pnl_per_trade(fill_p, won)
            out.append(
                f"| {t:.2f} | {len(trades)} | {fills:.0f}% | {won.mean()*100:.1f}% |"
                f" {fill_p.mean():.3f} | {pnl.mean():+.4f} | {pnl.sum():+.4f} |"
            )

        out.append("")

    frames = []
    for r in results:
        asset = r["asset"]
        tdf   = _augment_both_sides(r["test_df"].copy(), asset, pm_lookup, dn_lookup)
        tdf["asset"] = asset
        frames.append(tdf)

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
            won    = trades["effective_won"]
            fill_p = trades["pm_fill_price"].fillna(trades["pm_price_signal"])
            pnl    = _pnl_per_trade(fill_p, won)
            out.append(
                f"| {t:.2f} | {len(trades)} | {fills:.0f}% | {won.mean()*100:.1f}% |"
                f" {fill_p.mean():.3f} | {pnl.mean():+.4f} | {pnl.sum():+.4f} |"
            )
        out.append("")

    out.append(
        "_fill_rate% = % of test windows where a trade fires at this threshold. "
        "avg_fill = avg ask price 1s after trigger (network latency). "
        "avg_pnl per trade assumes a 1.5% buy fee on entry. "
        "Best side (UP or DOWN) taken per window._"
    )
    return "\n".join(out)


def _multi_trade_window_both(window_df: pd.DataFrame, threshold: float, cooldown: int = 5) -> pd.DataFrame:
    """
    Greedy multi-trade picker for a single window — UP and DOWN tracked independently.
    At each second, UP fires if edge_up >= threshold and cooldown since last UP trade.
    DOWN fires if edge_dn >= threshold and cooldown since last DOWN trade.
    Both can fire in the same second.
    """
    selected = []
    last_up = -cooldown - 1
    last_dn = -cooldown - 1
    for _, row in window_df.sort_values("elapsed_second").iterrows():
        elapsed = row["elapsed_second"]
        if row["edge_up"] >= threshold and elapsed - last_up >= cooldown:
            r = row.copy()
            r["side_taken"]      = "up"
            r["edge"]            = row["edge_up"]
            r["pm_price_signal"] = row["up_ask"]
            r["pm_fill_price"]   = row["up_ask_fill"]
            r["effective_won"]   = int(bool(row["resolved_up"]))
            selected.append(r)
            last_up = elapsed
        if row["edge_dn"] >= threshold and elapsed - last_dn >= cooldown:
            r = row.copy()
            r["side_taken"]      = "dn"
            r["edge"]            = row["edge_dn"]
            r["pm_price_signal"] = row["dn_ask"]
            r["pm_fill_price"]   = row["dn_ask_fill"]
            r["effective_won"]   = int(not bool(row["resolved_up"]))
            selected.append(r)
            last_dn = elapsed
    return pd.DataFrame(selected)


def section_multi_trade(results: list[dict], pm_lookup: dict, dn_lookup: dict) -> str:
    """
    Compares three strategies at each threshold:
      A) First-only        — one trade per window (first second EITHER side edge >= threshold)
      B) Multi 5s cooldown  — UP and DOWN re-enter independently every 5s
      C) Multi 10s cooldown — UP and DOWN re-enter independently every 10s

    avg_pnl and total_pnl are per-trade so they are directly comparable across strategies.
    """
    THRESHOLDS = [0.15, 0.20, 0.25]
    COOLDOWNS  = [5, 10]
    out = []

    def _build_multi_both(valid: pd.DataFrame, group_cols: list, t: float, cd: int) -> pd.DataFrame:
        parts = [
            _multi_trade_window_both(grp, t, cd)
            for _, grp in valid.groupby(group_cols)
        ]
        non_empty = [p for p in parts if not p.empty]
        return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()

    def _fmt(trades: pd.DataFrame, n_win: int) -> str:
        if trades.empty:
            return "0 | — | — | — | —"
        fill_p = trades["pm_fill_price"].fillna(trades["pm_price_signal"])
        won    = trades["effective_won"]
        pnl    = _pnl_per_trade(fill_p, won)
        return (
            f"{len(trades)} |"
            f" {len(trades)/n_win:.2f} |"
            f" {won.mean()*100:.1f}% |"
            f" {pnl.mean():+.4f} |"
            f" {pnl.sum():+.4f}"
        )

    header = (
        "| threshold"
        " | A: n | A: t/win | A: win% | A: avg_pnl | A: total_pnl"
        " | B(5s): n | B: t/win | B: win% | B: avg_pnl | B: total_pnl"
        " | C(10s): n | C: t/win | C: win% | C: avg_pnl | C: total_pnl |"
    )
    sep = "|---|" + "---:|" * 15

    all_frames = []
    for r in results:
        asset = r["asset"]
        valid = _augment_both_sides(r["test_df"].copy(), asset, pm_lookup, dn_lookup)
        if valid.empty:
            continue
        valid = valid.copy()
        valid["asset"] = asset
        all_frames.append(valid)

    if all_frames:
        all_valid       = pd.concat(all_frames, ignore_index=True)
        n_windows_total = all_valid.groupby("asset")["window_ts"].nunique().sum()
        out.append("### All Assets (aggregated)\n")
        out.append(header)
        out.append(sep)

        for t in THRESHOLDS:
            cands_a  = all_valid[all_valid["edge"] >= t].sort_values("elapsed_second")
            a_trades = cands_a.groupby(["asset", "window_ts"]).first().reset_index()
            b_trades = _build_multi_both(all_valid, ["asset", "window_ts"], t, COOLDOWNS[0])
            c_trades = _build_multi_both(all_valid, ["asset", "window_ts"], t, COOLDOWNS[1])
            out.append(
                f"| {t:.2f} | {_fmt(a_trades, n_windows_total)}"
                f" | {_fmt(b_trades, n_windows_total)}"
                f" | {_fmt(c_trades, n_windows_total)} |"
            )
        out.append("")

    out.append(
        "_A = first qualifying second per window (best side at that instant). "
        "B = UP and DOWN re-enter independently with 5s cooldown per side. "
        "C = UP and DOWN re-enter independently with 10s cooldown per side. "
        "t/win = avg trades per window. avg_pnl and total_pnl are per-trade._"
    )
    return "\n".join(out)


def _timing_summary(trades: "pd.DataFrame") -> str:
    """Return '| n | win% | avg_fill | avg_pnl |' cells for a trades DataFrame."""
    if trades.empty:
        return " — | — | — | — |"
    won    = trades["effective_won"]
    fill_p = trades["pm_fill_price"].fillna(trades["pm_price_signal"])
    pnl    = _pnl_per_trade(fill_p, won)
    return f" {len(trades)} | {won.mean()*100:.1f}% | {fill_p.mean():.3f} | {pnl.mean():+.4f} |"


def _entry_timing_for_threshold(results: list[dict], pm_lookup: dict, dn_lookup: dict,
                                threshold: float) -> list[str]:
    """Return lines for the entry-timing table at a single threshold."""
    out = []
    all_trades_frames = []

    for r in results:
        asset = r["asset"]
        valid = _augment_both_sides(r["test_df"].copy(), asset, pm_lookup, dn_lookup)
        if valid.empty:
            continue

        cands  = valid[valid["edge"] >= threshold].sort_values("elapsed_second")
        trades = cands.groupby("window_ts").first().reset_index()
        if trades.empty:
            continue

        all_trades_frames.append(trades)

    if all_trades_frames:
        all_trades = pd.concat(all_trades_frames, ignore_index=True)
        out.append("#### All Assets (aggregated)\n")
        out.append("| entry window | n | win% | avg_fill | avg_pnl |")
        out.append("|---|---:|---:|---:|---:|")

        for b in range(0, WINDOW_SECS, 30):
            grp    = all_trades[(all_trades["elapsed_second"] >= b) & (all_trades["elapsed_second"] < b + 30)]
            if grp.empty:
                continue
            out.append(f"| {b}–{b+30}s |{_timing_summary(grp)}")

        out.append("|---|---:|---:|---:|---:|")
        out.append(f"| **all windows** |{_timing_summary(all_trades)}")
        out.append("")

    return out


def section_entry_timing(results: list[dict], pm_lookup: dict, dn_lookup: dict) -> str:
    """
    Entry timing tables at thresholds 0.20, 0.25, 0.30 — how win% and avg_pnl vary
    by when in the window the edge threshold is first crossed.

    'skip <30s' columns re-run first-trade logic restricted to elapsed_second >= 30.
    """
    THRESHOLDS = [0.15, 0.20, 0.25]
    out = []

    for t in THRESHOLDS:
        out.append(f"### threshold = {t:.2f}\n")
        out.extend(_entry_timing_for_threshold(results, pm_lookup, dn_lookup, t))

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
        test_df["pm_fill_price"] = test_df["ts"].apply(
            lambda t: pm_lookup.get((asset, int(t) + 1))
        )
        test_df["edge"] = test_df["predicted_prob"] - test_df["pm_price_up_equiv"] * (1.0 + BUY_FEE_RATE)
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


def section_hour_of_day(results: list[dict], pm_lookup: dict, dn_lookup: dict,
                         threshold: float = DEFAULT_THRESHOLD) -> str:
    """Aggregated win rate and avg_pnl by UTC hour at the given threshold."""
    all_trades = []
    for r in results:
        asset  = r["asset"]
        valid  = _augment_both_sides(r["test_df"].copy(), asset, pm_lookup, dn_lookup)
        cands  = valid[valid["edge"] >= threshold].sort_values("elapsed_second")
        trades = cands.groupby("window_ts").first().reset_index()
        if trades.empty:
            continue
        trades["asset"] = asset
        all_trades.append(trades)

    if not all_trades:
        return "_No trades available for hour-of-day aggregation._"

    agg = pd.concat(all_trades, ignore_index=True)
    agg["hour_utc"] = pd.to_datetime(agg["window_ts"], unit="s", utc=True).dt.hour

    # Compute per-hour avg_pnl for bar chart
    hour_pnl: dict[int, tuple[float, int]] = {}
    for hour in range(24):
        grp = agg[agg["hour_utc"] == hour]
        if grp.empty:
            continue
        won    = grp["effective_won"]
        fill_p = grp["pm_fill_price"].fillna(grp["pm_price_signal"])
        pnl    = _pnl_per_trade(fill_p, won)
        hour_pnl[hour] = (float(pnl.mean()), len(grp))

    out = [
        "### All Assets (aggregated)",
        "",
        "**avg_pnl by hour (UTC)** — each █ ≈ 1% avg_pnl:\n",
        "```",
    ]
    for hour in range(24):
        if hour not in hour_pnl:
            continue
        e, n    = hour_pnl[hour]
        bar_len = max(0, int(abs(e) * 100))
        bar     = ("█" * bar_len) if e >= 0 else ("░" * bar_len)
        sign    = "+" if e >= 0 else "-"
        flag    = " ⚠" if e < -0.05 else ""
        out.append(f"  {hour:02d}h UTC  {sign}{abs(e):.3f}  {bar}  (n={n}){flag}")
    out.append("```")

    if hour_pnl:
        sorted_hours = sorted(hour_pnl.items(), key=lambda x: x[1][0], reverse=True)
        top3    = sorted_hours[:3]
        bottom3 = sorted_hours[-3:]
        out.append(
            "\n**Best hours:**  " +
            "  ".join(f"{h:02d}h UTC ({e:+.3f})" for h, (e, _) in top3)
        )
        out.append(
            "**Worst hours:** " +
            "  ".join(f"{h:02d}h UTC ({e:+.3f})" for h, (e, _) in bottom3)
        )

    out.append("")
    out.append("| hour UTC | n | win% | avg_fill | avg_pnl |")
    out.append("|---:|---:|---:|---:|---:|")

    for hour in range(24):
        grp = agg[agg["hour_utc"] == hour]
        if grp.empty:
            continue
        won    = grp["effective_won"]
        fill_p = grp["pm_fill_price"].fillna(grp["pm_price_signal"])
        pnl    = _pnl_per_trade(fill_p, won)
        flag   = " ⚠" if won.mean() < 0.45 else ""
        out.append(
            f"| {hour:02d}:00 | {len(grp)} | {won.mean()*100:.1f}%{flag}"
            f" | {fill_p.mean():.3f} | {pnl.mean():+.4f} |"
        )

    out.append("")
    return "\n".join(out)


_DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def section_day_of_week(results: list[dict], pm_lookup: dict, dn_lookup: dict,
                         threshold: float = DEFAULT_THRESHOLD) -> str:
    """Aggregated win rate and avg_pnl by day of week (UTC) at the given threshold."""
    all_trades = []
    for r in results:
        asset  = r["asset"]
        valid  = _augment_both_sides(r["test_df"].copy(), asset, pm_lookup, dn_lookup)
        cands  = valid[valid["edge"] >= threshold].sort_values("elapsed_second")
        trades = cands.groupby("window_ts").first().reset_index()
        if trades.empty:
            continue
        trades["asset"] = asset
        all_trades.append(trades)

    if not all_trades:
        return "_No trades available for day-of-week aggregation._"

    agg = pd.concat(all_trades, ignore_index=True)
    agg["dow"] = pd.to_datetime(agg["window_ts"], unit="s", utc=True).dt.dayofweek  # 0=Mon

    dow_stats: dict[int, tuple[float, float, int]] = {}
    for dow in range(7):
        grp = agg[agg["dow"] == dow]
        if grp.empty:
            continue
        won    = grp["effective_won"]
        fill_p = grp["pm_fill_price"].fillna(grp["pm_price_signal"])
        pnl    = _pnl_per_trade(fill_p, won)
        dow_stats[dow] = (float(pnl.mean()), float(won.mean()), len(grp))

    out = [
        "### All Assets (aggregated)",
        "",
        "**avg_pnl by day of week (UTC)** — each █ ≈ 1% avg_pnl:\n",
        "```",
    ]
    for dow in range(7):
        if dow not in dow_stats:
            continue
        pnl, wr, n = dow_stats[dow]
        bar  = ("█" if pnl >= 0 else "░") * max(0, int(abs(pnl) * 100))
        sign    = "+" if pnl >= 0 else "-"
        flag    = " ⚠" if pnl < -0.05 else ""
        out.append(f"  {_DOW_NAMES[dow]}  {sign}{abs(pnl):.3f}  {bar}  win={wr:.1%}  (n={n}){flag}")
    out.append("```")

    if dow_stats:
        sorted_dow = sorted(dow_stats.items(), key=lambda x: x[1][0], reverse=True)
        out.append(
            "\n**Best days:**  " +
            "  ".join(f"{_DOW_NAMES[d]} ({p:+.3f})" for d, (p, _, _) in sorted_dow[:3])
        )
        out.append(
            "**Worst days:** " +
            "  ".join(f"{_DOW_NAMES[d]} ({p:+.3f})" for d, (p, _, _) in sorted_dow[-3:])
        )

    out.append("")
    out.append("| day | n | win% | avg_fill | avg_pnl |")
    out.append("|---:|---:|---:|---:|---:|")
    for dow in range(7):
        grp = agg[agg["dow"] == dow]
        if grp.empty:
            continue
        won    = grp["effective_won"]
        fill_p = grp["pm_fill_price"].fillna(grp["pm_price_signal"])
        pnl    = _pnl_per_trade(fill_p, won)
        flag   = " ⚠" if won.mean() < 0.45 else ""
        out.append(
            f"| {_DOW_NAMES[dow]} | {len(grp)} | {won.mean()*100:.1f}%{flag}"
            f" | {fill_p.mean():.3f} | {pnl.mean():+.4f} |"
        )

    out.append("")
    return "\n".join(out)


def section_recent_windows(results: list[dict], pm_lookup: dict, dn_lookup: dict,
                            threshold: float = DEFAULT_THRESHOLD) -> str:
    """Performance over the last 24h and 48h of test windows at the given threshold."""
    HORIZONS = [("24h", 288), ("48h", 576)]  # 288 × 5min = 24h, 576 × 5min = 48h
    out = []

    for r in results:
        asset = r["asset"]
        valid = _augment_both_sides(r["test_df"].copy(), asset, pm_lookup, dn_lookup)

        all_windows = sorted(valid["window_ts"].unique())
        out.append(f"### {asset}  _(total test windows: {len(all_windows)})_\n")
        out.append("| horizon | windows | trades | fill% | win% | avg_fill | avg_pnl | total_pnl |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|")

        for label, n_win in HORIZONS:
            recent_wins = set(all_windows[-n_win:])
            recent = valid[valid["window_ts"].isin(recent_wins)]
            cands  = recent[recent["edge"] >= threshold].sort_values("elapsed_second")
            trades = cands.groupby("window_ts").first().reset_index()
            n_w    = len(recent_wins)
            if trades.empty:
                out.append(f"| {label} | {n_w} | 0 | 0% | — | — | — | — |")
                continue
            won    = trades["effective_won"]
            fill_p = trades["pm_fill_price"].fillna(trades["pm_price_signal"])
            pnl    = _pnl_per_trade(fill_p, won)
            out.append(
                f"| {label} | {n_w} | {len(trades)} | {len(trades)/n_w*100:.0f}% |"
                f" {won.mean()*100:.1f}% | {fill_p.mean():.3f} | {pnl.mean():+.4f} | {pnl.sum():+.4f} |"
            )
        out.append("")

    return "\n".join(out)


def section_edge_by_decile(results: list[dict], pm_lookup: dict, dn_lookup: dict,
                            threshold: float = DEFAULT_THRESHOLD) -> str:
    """
    Pool first-trade rows (edge >= threshold) across all assets, bucket by
    predicted_prob decile, show win%, avg_fill, avg_pnl per decile.
    Reveals whether higher model confidence actually delivers better outcomes.
    """
    frames = []
    for r in results:
        asset = r["asset"]
        tdf   = _augment_both_sides(r["test_df"].copy(), asset, pm_lookup, dn_lookup)
        tdf["asset"] = asset
        frames.append(tdf)

    if not frames:
        return "_No data available._"

    all_valid = pd.concat(frames, ignore_index=True)
    cands  = all_valid[all_valid["edge"] >= threshold].sort_values("elapsed_second")
    trades = cands.groupby(["asset", "window_ts"]).first().reset_index()

    if trades.empty:
        return f"_No trades at threshold={threshold}._"

    trades["decile"] = pd.qcut(trades["predicted_prob"], q=10, labels=False, duplicates="drop")

    out = [
        "| decile | prob range | n | win% | avg_fill | avg_pnl |",
        "|---:|---|---:|---:|---:|---:|",
    ]

    for d, grp in trades.groupby("decile"):
        won    = grp["effective_won"]
        fill_p = grp["pm_fill_price"].fillna(grp["pm_price_signal"])
        pnl    = _pnl_per_trade(fill_p, won)
        lo     = grp["predicted_prob"].min()
        hi     = grp["predicted_prob"].max()
        out.append(
            f"| {int(d)+1} | {lo:.3f}–{hi:.3f} | {len(grp)} |"
            f" {won.mean()*100:.1f}% | {fill_p.mean():.3f} | {pnl.mean():+.4f} |"
        )

    out.append(
        f"\n_Decile 1 = lowest predicted_prob, decile 10 = highest. "
        f"Trades: first second per window where edge ≥ {threshold}, pooled across all assets. "
        f"avg_fill = ask price 1s after trigger._"
    )
    return "\n".join(out)


# ── Execution slippage ────────────────────────────────────────────────────────

def section_slippage(results: list[dict], pm_lookup: dict, dn_lookup: dict,
                     threshold: float = DEFAULT_THRESHOLD) -> str:
    """
    1-second and 2-second execution slippage at actual trade moments.
    Slippage = ask at fill second (t+N) − ask at signal second (t), for the side taken.
    Positive = PM repriced against us; negative = moved in our favour.
    Does NOT include the 1.5% fee.

    1s slippage is already baked into all PnL numbers (fills are simulated at t+1).
    2s comparison shows the cost of a slower execution path.
    """
    frames = []
    for r in results:
        asset  = r["asset"]
        valid  = _augment_both_sides(r["test_df"].copy(), asset, pm_lookup, dn_lookup)
        valid  = valid.dropna(subset=["pm_fill_price"])

        # 2-second fill: look up t+2 ask for the taken side
        def _ask_t2(row):
            t2 = int(row["ts"]) + 2
            if row["side_taken"] == "dn":
                return dn_lookup.get((asset, t2))
            return pm_lookup.get((asset, t2))

        valid["pm_fill_price_2s"] = valid.apply(_ask_t2, axis=1)

        cands  = valid[valid["edge"] >= threshold].sort_values("elapsed_second")
        trades = cands.groupby("window_ts").first().reset_index()
        if trades.empty:
            continue
        trades["asset"]      = asset
        trades["slippage"]   = trades["pm_fill_price"]    - trades["pm_price_signal"]
        trades["slippage_2s"] = trades["pm_fill_price_2s"] - trades["pm_price_signal"]
        frames.append(trades)

    if not frames:
        return "_No trades available._"

    all_trades = pd.concat(frames, ignore_index=True)

    def _slip_row(label: str, grp: pd.DataFrame, col: str) -> str:
        slip = grp[col].dropna()
        if slip.empty:
            return f"| {label} | — | — | — | — | — | — | — |"
        signal_col = "pm_price_signal"
        fill_col   = "pm_fill_price" if col == "slippage" else "pm_fill_price_2s"
        return (
            f"| {label} | {len(slip)} |"
            f" {grp[signal_col].mean():.4f} |"
            f" {grp[fill_col].dropna().mean():.4f} |"
            f" {slip.mean():+.5f} |"
            f" {slip.median():+.5f} |"
            f" {slip.quantile(0.75):+.5f} |"
            f" {(slip > 0).mean()*100:.0f}% |"
        )

    out = []
    for lag, col in [("1s (actual)", "slippage"), ("2s (slower exec)", "slippage_2s")]:
        out.append(f"**Fill lag: {lag}**\n")
        out.append("| asset | n | ask@signal | ask@fill | avg_slip | median_slip | p75_slip | adverse% |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for asset, grp in all_trades.groupby("asset"):
            out.append(_slip_row(asset, grp, col))
        out.append(_slip_row("**ALL**", all_trades, col))
        out.append("")

    # Extra: avg additional slippage cost of 2s vs 1s
    extra = (all_trades["slippage_2s"] - all_trades["slippage"]).dropna()
    out.append(
        f"_Avg additional slippage from 2s vs 1s fill: {extra.mean():+.5f} "
        f"(median {extra.median():+.5f}). "
        f"1s slippage is already baked into all PnL numbers — fills are simulated at t+1. "
        f"adverse = PM ask moved up before fill. Does NOT include the 1.5% buy fee._"
    )
    return "\n".join(out)


# ── Both-sides helper ────────────────────────────────────────────────────────

def _augment_both_sides(test_df: pd.DataFrame, asset: str,
                        pm_lookup: dict, dn_lookup: dict) -> pd.DataFrame:
    """
    For each second in test_df, compute edge for both UP and DOWN sides.
    Takes the better side (tie goes to UP).

    Adds columns:
      edge            — best side edge (used for threshold comparisons)
      pm_fill_price   — ask at t+1 for the taken side
      effective_won   — 1 if trade won, accounting for which side was taken
      side_taken      — "up" or "dn"
      pm_price_signal — ask at t for the taken side (for slippage)
    """
    df = test_df.copy()
    df["up_ask"]      = df["ts"].apply(lambda t: pm_lookup.get((asset, int(t))))
    df["up_ask_fill"] = df["ts"].apply(lambda t: pm_lookup.get((asset, int(t) + 1)))
    df["dn_ask"]      = df["ts"].apply(lambda t: dn_lookup.get((asset, int(t))))
    df["dn_ask_fill"] = df["ts"].apply(lambda t: dn_lookup.get((asset, int(t) + 1)))
    df["edge_up"]     = df["predicted_prob"]         - df["up_ask"] * (1.0 + BUY_FEE_RATE)
    df["edge_dn"]     = (1.0 - df["predicted_prob"]) - df["dn_ask"] * (1.0 + BUY_FEE_RATE)
    # Best side per second; UP wins ties
    take_dn           = df["edge_dn"] > df["edge_up"]
    df["edge"]        = np.where(take_dn, df["edge_dn"], df["edge_up"])
    df["side_taken"]  = np.where(take_dn, "dn", "up")
    df["pm_price_signal"] = np.where(take_dn, df["dn_ask"],      df["up_ask"])
    df["pm_fill_price"]   = np.where(take_dn, df["dn_ask_fill"], df["up_ask_fill"])
    df["effective_won"]   = np.where(
        take_dn,
        (~df["resolved_up"].astype(bool)).astype(int),
        df["resolved_up"].astype(int),
    )
    return df.dropna(subset=["up_ask"])


# ── Report builder ─────────────────────────────────────────────────────────────

def build_report(results: list[dict], pm_lookup: dict, generated_at: str,
                 dn_lookup: "dict | None" = None,
                 ensemble_results: "dict | None" = None,
                 extended_lgb_results: "dict | None" = None,
                 include_extended_features: bool = False,
                 ablation_sections: "dict[str, str] | None" = None) -> str:
    dn = dn_lookup or {}
    # top-line summary per asset for the overview
    summary_lines = []
    for r in results:
        asset  = r["asset"]
        valid  = _augment_both_sides(r["test_df"].copy(), asset, pm_lookup, dn)
        cands  = valid[valid["edge"] >= DEFAULT_THRESHOLD].sort_values("elapsed_second")
        trades = cands.groupby("window_ts").first().reset_index() if not cands.empty else pd.DataFrame()

        if trades.empty:
            summary_lines.append(f"- **{asset}**: AUC={r['auc']:.3f} — no trades at {DEFAULT_THRESHOLD} threshold")
            continue

        won    = trades["effective_won"]
        fill_p = trades["pm_fill_price"].fillna(trades["pm_price_signal"])
        pnl    = _pnl_per_trade(fill_p, won)
        n_win  = valid["window_ts"].nunique()
        summary_lines.append(
            f"- **{asset}**: AUC={r['auc']:.3f} — "
            f"{len(trades)}/{n_win} windows triggered ({len(trades)/n_win*100:.0f}%) — "
            f"win={won.mean()*100:.1f}% — avg_pnl={pnl.mean():+.4f} — total_pnl={pnl.sum():+.4f}"
        )

    lines = [
        "# Model Report: Edge-Based Entry Strategy",
        f"_Generated {generated_at}_",
        "",
        "**Strategy:** At each second of every 5-minute window, compute edge for both UP and DOWN sides.",
        "The first second where either side's edge ≥ threshold triggers a trade on that side (higher edge wins ties).",
        "",
        "**Test set only** — windows with market data. Training uses earlier coin-only windows.",
        "",
        f"### At-a-glance (threshold = {DEFAULT_THRESHOLD})",
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
        # "",
        # "## 1b. Calibration Method Comparison — Platt vs Isotonic",
        # "",
        # "Platt remains the active calibration used for trading and saved models.",
        # "Isotonic is evaluation-only here (side-by-side metrics on the same held-out test set).",
        # "",
        # section_calibration_method_compare(results),
        # "",
        # "---",
        "",
        "## 2. What Edge Threshold Should I Use? — EV Sweep",
        "",
        "Sweeps thresholds from 5c to 30c. `fill_rate%` = how often a trade fires per session.",
        "`avg_pnl` includes a 1.5% buy fee on entry. Best side (UP or DOWN) taken at first trigger each window.",
        "",
        section_threshold_ev(results, pm_lookup, dn),
        "",
        "---",
        "",
        "## 3. Multi-Trade vs First-Only Comparison",
        "",
        "**A**: one trade per window (first second either side edge ≥ threshold). "
        "**B/C**: UP and DOWN re-enter independently with 5s / 10s cooldown per side. "
        "`avg_pnl` is per-trade across all columns so they are directly comparable.",
        "",
        section_multi_trade(results, pm_lookup, dn),
        "",
        "---",
        "",
        "## 4. Execution Slippage — 1s vs 2s Fill Lag",
        "",
        "How much does the Polymarket ask move between signal and fill? "
        "1s = current execution (already baked into all PnL numbers). "
        "2s = cost of a slower execution path, shown for comparison.",
        f"Measured at actual trade moments (first qualifying second per window at threshold={DEFAULT_THRESHOLD}).",
        "Does NOT include the 1.5% fee.",
        "",
        section_slippage(results, pm_lookup, dn),
        "",
        "---",
        "",
        "## 5. When in the Window Should I Enter? — Entry Timing",
        "",
        "At thresholds 0.20 / 0.25 / 0.30: does entry timing affect outcome?",
        "Earlier entries have lower fill prices but less model conviction.",
        "",
        section_entry_timing(results, pm_lookup, dn),
        "",
        "---",
        "",
        "## 6. What Drives the Edge? — Feature Importance",
        "",
        "_LightGBM gain-based importance — how much each feature reduces loss across all trees. Higher = more predictive._",
        "",
        section_feature_importance(results),
        "",
        "---",
        "",
        "## 7. Model Quality Metrics",
        "",
        "_AUC measures ranking quality (0.5=random, 1.0=perfect). Brier measures probability accuracy (lower=better)._",
        "_Baseline win% ≈ 50% because the model is trained on all window-seconds, not just trigger moments._",
        "",
        section_metrics(results),
        "",
        "---",
        "",
        "## 8. Edge by Model Confidence Decile",
        "",
        f"All assets pooled. Trades: first qualifying second per window at threshold={DEFAULT_THRESHOLD}. "
        "Does higher predicted_prob actually mean better outcomes?",
        "",
        section_edge_by_decile(results, pm_lookup, dn),
        "",
        "---",
        "",
        "## 9. Hour of Day — When Is the Edge Largest?",
        "",
        f"Win rate and avg_pnl by UTC hour at threshold={DEFAULT_THRESHOLD}. "
        "⚠ = win rate below 45%.",
        "",
        section_hour_of_day(results, pm_lookup, dn),
        "",
        "---",
        "",
        "## 9b. Day of Week — Does the Day Matter?",
        "",
        f"Aggregated across all assets at threshold={DEFAULT_THRESHOLD}. ⚠ = win rate below 45%.",
        "",
        section_day_of_week(results, pm_lookup, dn),
        "",
        "---",
        "",
        "## 10. Recent Performance — Last 24h and 48h",
        "",
        f"Last 288 / 576 test windows (24h / 48h at one window per 5 min) at threshold={DEFAULT_THRESHOLD}. "
        "Checks whether the model is still working on recent data.",
        "",
        section_recent_windows(results, pm_lookup, dn),
        "",
        "---",
        "",
    ]

    if include_extended_features:
        lines += [
            "",
            "## 13. Extended Features — Do 10 New Signals Help?",
            "",
            f"Trains a second LightGBM on {len(EXTENDED_FEATURES)} features (base {len(FEATURES)} + {len(NEW_FEATURES)} new structural/regime features). "
            "Same split and Platt calibration as the primary model. "
            "ΔAUC > 0 or ΔBrier < 0 means the extended feature set adds value.",
            "",
            section_extended_features(results, extended_lgb_results or {}, pm_lookup, dn),
            "",
            "---",
        ]

    # Section 14: Feature ablation (optional — only if --ablation-assets was set)
    if ablation_sections:
        for ab_asset, ab_text in ablation_sections.items():
            lines += [
                "",
                f"## 14. Feature Ablation — {ab_asset}",
                "",
                f"Single-feature AUC, leave-one-out, and forward greedy selection for {ab_asset}. "
                "Uses LGB 100 trees (no Platt) — fast enough to sweep all feature subsets. "
                "Run with `--ablation-assets` to regenerate.",
                "",
                ab_text,
                "",
                "---",
            ]

    lines += [
        "",
        "## Methodology",
        "",
        "### Features",
        "",
        "Each feature is computed per-second within a 5-minute window. `σ` is the walk-forward EWMA sigma"
        f" (λ={EWMA_LAMBDA}) seeded from `assets.yaml` and updated each window.",
        "",
        "| feature | formula | what it captures |",
        "|---|---|---|",
        "| `move_sigmas` | `(price[t] − price[open]) / σ` | cumulative move from window open, in σ units; the primary directional signal |",
        "| `elapsed_second` | `t − window_start` (0–299) | how far into the window we are; late moves have less time to reverse |",
        "| `hour_sin` | `sin(hour_utc × 2π / 24)` | time-of-day encoded cyclically so midnight wraps correctly |",
        "| `hour_cos` | `cos(hour_utc × 2π / 24)` | paired with `hour_sin` to fully represent hour as a point on the unit circle |",
        "| `vel_2s` | `(price[t] − price[t−2]) / σ` | very short-term momentum; captures tick-level acceleration |",
        "| `vel_5s` | `(price[t] − price[t−5]) / σ` | short-term momentum over a 5-second lookback |",
        "| `vel_10s` | `(price[t] − price[t−10]) / σ` | medium short-term momentum; smoother than vel_2s |",
        "| `acc_4s` | `(price[t] − 2·price[t−2] + price[t−4]) / σ` | second derivative over 4s; positive = accelerating up, negative = decelerating |",
        "| `acc_10s` | `(price[t] − 2·price[t−5] + price[t−10]) / σ` | second derivative over 10s; slower-frequency curvature |",
        "| `vel_ratio` | `|vel_2s| / |vel_10s|` | recent speed vs recent trend speed; > 1 = burst, < 1 = fading |",
        "| `vel_decay` | `|vel_10s| − |vel_2s|` | positive = move is slowing (mean-reversion signal), negative = accelerating |",
        f"| `vol_10s_log` | `log1p(Σ volume[t−{VOL_LOOKBACK}…t))` | log of coin volume in the last {VOL_LOOKBACK}s; high volume = more conviction |",
        "| `move_x_elapsed` | `move_sigmas × elapsed_second` | interaction: a large move late in the window is more likely to hold |",
        "| `move_x_vol` | `move_sigmas × vol_10s_log` | interaction: a large move on high volume is more reliable than a thin-book move |",
        "",
        "**Extended features** (tested in section 13, not used by primary model unless promoted):\n",
        "| feature | formula | what it captures |",
        "|---|---|---|",
        "| `dist_high_30` | `(price[t] − max(price[t−30:t])) / σ` | near 0 = breakout zone; very negative = far below recent highs (exhaustion) |",
        "| `dist_low_30` | `(price[t] − min(price[t−30:t])) / σ` | near 0 = breakdown zone; very positive = overextended upward |",
        "| `zscore_20` | `(price[t] − mean(price[t−20:t])) / std(price[t−20:t])` | local mean-reversion pressure; large values indicate stretch |",
        "| `vol_z_30` | `(vol[t] − mean(vol[t−30:t])) / std(vol[t−30:t])` | abnormal participation; high z = unusual conviction |",
        "| `signed_vol_imb` | `Σ vol[i]·sign(Δprice[i]) / Σ vol[i]` over 10s | net buy vs sell pressure weighted by volume (range ≈ [−1, 1]) |",
        "| `trend_str_30` | `|price[t] − price[t−30]| / Σ|Δprice[i]|` over 30s | ≈1 = clean directional trend; ≈0 = choppy noise |",
        "| `vol_expansion` | `std(price[t−10:t]) / std(price[t−30:t])` | >1 = vol expanding / breakout; <1 = compression / mean reversion |",
        "| `mom_slope` | `vel_2s − vel_10s` | positive = accelerating; negative = fading / decelerating |",
        "| `dir_consistency_10` | `Σ sign(Δprice[i]) / 10` over 10s | +1 = all upticks; 0 = chop; −1 = all downticks |",
        "| `time_since_flip` | seconds since `sign(Δprice)` last changed | long runs tend to revert; captures trend persistence |",
        "",
        "### Model & training",
        "",
        "- **Primary model**: LightGBM (300 trees, max_depth=4, lr=0.05, subsample=0.8) + Platt calibration",
        "- **Calibration split**: last 20% of train windows (time-ordered) held out for Platt scaling",
        "- **Train/test split**: test = windows with market (PM) data; train = earlier coin-only windows — window-level, no row leakage",
        "- **EWMA sigma**: walk-forward, λ=0.95, first 20 windows excluded (warmup). Matches live executor.",
        "- **NaN imputation**: early-window seconds (missing lookbacks) imputed with training-set median",
        "- **Resolution label**: `close > open` in the 5-min coin window ⇒ UP, `close < open` ⇒ DOWN",
        "",
        "### Edge & trading",
        "",
        "- `edge_up = predicted_prob − up_ask × 1.015`",
        "- `edge_dn = (1 − predicted_prob) − dn_ask × 1.015`",
        "- Trade side: whichever side has higher edge at the first qualifying second (tie → UP)",
        f"- Buy fee: {BUY_FEE_RATE*100:.1f}% applied to fill price in all PnL tables",
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
    p.add_argument("--ablation-assets", nargs="+", default=None, metavar="ASSET",
                   help="Run feature ablation (single, LOO, forward greedy) for these assets (slow)")
    p.add_argument("--test-extended-features", action="store_true",
                   help="Train/report extended-feature LightGBM comparison (off by default)")
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
        if args.production:
            log.warning("%s: %s — proceeding without PM data in production mode", asset, e)
            pm_df = pd.DataFrame(columns=["ts", "window_ts", "up_ask", "dn_ask"])
        else:
            log.error("%s: %s", asset, e)
            return None

    if pm_df.empty and not args.production:
        log.warning("%s: no PM data — skipping in report mode", asset)
        return None

    # Sigma is computed over test windows only (the PM data windows), matching
    # the sigma used in production where only recent data is available.
    # In production mode (no PM data) fall back to all coin windows.
    if not pm_df.empty:
        test_windows = sorted(pm_df["window_ts"].unique().tolist())
    else:
        test_windows = build_coin_windows(close_series)

    try:
        sigma = compute_sigma(test_windows, close_series)
    except ValueError as e:
        log.warning("%s: %s — skipping", asset, e)
        return None
    log.info("%s: sigma=%.6g  (from %d test windows)", asset, sigma, len(test_windows))

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
            result = train_asset_lgb(asset, df)
            if result is not None:
                model_path = os.path.join(args.out_models, f"{asset.lower()}.joblib")
                joblib.dump({"pipe": result["pipe"], "features": FEATURES, "sigma": sigma}, model_path)
                log.info("Model → %s  (n=%d)", model_path, result["n_train"])
        return

    # Report mode — train/test split, generate report and predictions CSV
    os.makedirs(os.path.dirname(args.out_report), exist_ok=True)

    results:              list[dict]                   = []
    pm_lookup:            dict[tuple[str, int], float] = {}
    dn_lookup:            dict[tuple[str, int], float] = {}
    extended_lgb_results: dict[str, dict]              = {}
    ablation_sections:    dict[str, str]               = {}
    ablation_assets = {a.upper() for a in args.ablation_assets} if args.ablation_assets else set()

    for asset in assets:
        log.info("=== %s ===", asset)
        loaded = _load_asset(args, asset)
        if loaded is None:
            continue
        pm_df, close_series, volume_series, sigma = loaded

        up_rows = pm_df[pm_df["up_ask"].notna()]
        pm_lookup.update({(asset, int(ts)): float(ask) for ts, ask in zip(up_rows["ts"], up_rows["up_ask"])})
        dn_rows = pm_df[pm_df["dn_ask"].notna()]
        dn_lookup.update({(asset, int(ts)): float(ask) for ts, ask in zip(dn_rows["ts"], dn_rows["dn_ask"])})

        df = build_asset_dataset(asset, pm_df, close_series, volume_series, sigma)
        if df.empty:
            continue

        # Primary model: LightGBM + Platt calibration (base features)
        result = train_asset_lgb(asset, df)
        if result is not None:
            result["close_series"] = close_series
            result["sigma"]        = sigma
            results.append(result)
            model_path = os.path.join(args.out_models, f"{asset.lower()}.joblib")
            joblib.dump({"pipe": result["pipe"], "features": FEATURES, "sigma": sigma}, model_path)
            log.info("Model → %s", model_path)

        # Comparison: LightGBM with extended feature set (report only, not saved)
        if args.test_extended_features:
            log.info("=== %s Extended LGB comparison ===", asset)
            ext_result = train_asset_lgb(asset, df, features=EXTENDED_FEATURES)
            if ext_result is not None:
                extended_lgb_results[asset] = ext_result

        # Feature ablation (optional — only when --ablation-assets includes this asset)
        if asset in ablation_assets:
            log.info("=== %s feature ablation ===", asset)
            ablation_sections[asset] = section_feature_ablation(asset, df)

    if not results:
        log.error("No results produced.")
        sys.exit(1)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report       = build_report(
        results,
        pm_lookup,
        generated_at,
        dn_lookup=dn_lookup,
        extended_lgb_results=extended_lgb_results,
        include_extended_features=args.test_extended_features,
        ablation_sections=ablation_sections or None,
    )

    with open(args.out_report, "w") as f:
        f.write(report)
    log.info("Report → %s", args.out_report)

    preds_df = build_predictions_df(results, pm_lookup)
    preds_df.to_csv(args.out_csv, index=False)
    log.info("Predictions → %s  (%d rows)", args.out_csv, len(preds_df))

    print(report)


if __name__ == "__main__":
    main()
