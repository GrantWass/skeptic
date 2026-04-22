#!/usr/bin/env python3
"""
Threshold edge analysis + report.

For each 5-minute Polymarket window:
  - Finds the FIRST second the coin crosses window_open ± N*sigma (either direction)
  - Records the corresponding Polymarket price at that trigger moment
  - Determines whether that bet won (up crossing → bet on UP; down crossing → bet on DOWN)

Report sections:
  1. Summary — edge by asset / sigma
  2. Trigger timing — edge by when in the window the trigger fires (early/mid/late)
  3. Hour of day — edge by UTC hour
  4. Cascade rate — if coin hits 0.5σ, how often does it go on to hit 1σ / 1.5σ / 2σ?
  5. Time to reprice — how many seconds after trigger does PM price catch up?

Usage:
    python scripts/threshold_edge.py
    python scripts/threshold_edge.py --assets BTC ETH --sigma 1.0
    python scripts/threshold_edge.py --assets BTC DOGE --sigma 0.5 1.0 1.5 2.0
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%M:%S")
log = logging.getLogger(__name__)

WINDOW_SECS = 300

ASSET_TO_SYMBOL = {
    "BTC":  "BTCUSDT",
    "ETH":  "ETHUSDT",
    "SOL":  "SOLUSDT",
    "DOGE": "DOGEUSDT",
    "XRP":  "XRPUSDT",
    "BNB":  "BNBUSDT",
    "HYPE": "HYPEUSDT",
}


HALF_DAY_MIN_WINDOWS = 50
BUY_FEE_RATE = 0.015


def _price_with_fee(price):
    """Return effective entry cost after buy fee."""
    return price * (1.0 + BUY_FEE_RATE)


def _edge_from_win_and_pm(win_rate: float, avg_pm_price: float) -> float:
    """Net edge per fill after buy fee."""
    return float(win_rate - _price_with_fee(avg_pm_price))


def _price_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where pm_price × 1.015 < per-(asset, sigma) win rate.
    Win rate is computed from the full df so the hurdle rate is stable.
    """
    win_rates = (
        df.groupby(["asset", "sigma"])["won"]
        .mean()
        .rename("_win_rate")
        .reset_index()
    )
    merged = df.merge(win_rates, on=["asset", "sigma"], how="left")
    merged.index = df.index
    return df[_price_with_fee(merged["pm_price"]) < merged["_win_rate"]]


# ── data loading ──────────────────────────────────────────────────────────────

def load_prices(prices_dir: str) -> pd.DataFrame:
    files = sorted(Path(prices_dir).glob("prices_*.csv"))
    if not files:
        raise FileNotFoundError(f"No prices_*.csv in {prices_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset=["ts", "window_ts", "asset"])
    df = df.sort_values(["asset", "window_ts", "ts"]).reset_index(drop=True)
    return df


def load_prices_for_asset(prices_dir: str, asset: str) -> pd.DataFrame:
    files = sorted(Path(prices_dir).glob("prices_*.csv"))
    if not files:
        raise FileNotFoundError(f"No prices_*.csv in {prices_dir}")
    frames = []
    for f in files:
        chunk = pd.read_csv(f)
        chunk = chunk[chunk["asset"] == asset.upper()]
        if not chunk.empty:
            frames.append(chunk)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["ts", "window_ts", "asset"])
    df = df.sort_values(["window_ts", "ts"]).reset_index(drop=True)
    return df


def load_coin_prices(coin_dir: str, asset: str) -> pd.Series | None:
    symbol = ASSET_TO_SYMBOL.get(asset.upper())
    if symbol is None:
        log.warning("No symbol mapping for %s", asset)
        return None
    path = os.path.join(coin_dir, f"{symbol}_1s.csv")
    if not os.path.exists(path):
        log.warning("No coin price file: %s", path)
        return None
    df = pd.read_csv(path, usecols=["ts", "close"])
    df = df.drop_duplicates("ts").set_index("ts")["close"].astype(float)
    if df.empty:
        log.warning("No coin price data for %s — skipping", asset)
        return None
    log.info("Loaded %d coin price rows for %s", len(df), asset)
    return df


# ── resolution ────────────────────────────────────────────────────────────────

def _resolve_window(pm_window: pd.DataFrame) -> bool | None:
    """Last up_price >= 0.95 → UP won, <= 0.05 → DOWN won. Returns None if ambiguous."""
    if pm_window.empty or "up_price" not in pm_window.columns:
        return None
    last_up = pm_window.sort_values("ts")["up_price"].iloc[-1]
    if pd.isna(last_up):
        return None
    if last_up >= 0.95:
        return True
    if last_up <= 0.05:
        return False
    return None


# ── per-asset analysis ────────────────────────────────────────────────────────

def analyze_asset(
    asset: str,
    pm_df: pd.DataFrame,
    coin_series: pd.Series,
    sigma_multiples: list[float],
    min_elapsed_secs: int = 0,
) -> tuple[pd.DataFrame, int]:
    """
    Returns (records, n_unresolved).
    records: one record per (window, sigma) — the FIRST threshold crossing.
    Columns:
      asset, window_ts, hour_utc, sigma, trigger_dir,
      trigger_ts, trigger_second, pm_price, resolved_up, won
    """
    asset_pm = pm_df[pm_df["asset"] == asset].copy()
    windows = sorted(asset_pm["window_ts"].unique())

    # Extract numpy arrays once — avoids O(N) boolean mask per window
    ts_idx = coin_series.index.values
    vals   = coin_series.values

    # sigma = std dev of window close-minus-open moves
    window_moves = []
    for wts in windows:
        lo = int(np.searchsorted(ts_idx, wts,               side="left"))
        hi = int(np.searchsorted(ts_idx, wts + WINDOW_SECS, side="left"))
        if (hi - lo) < 280:
            continue
        window_moves.append(float(vals[hi - 1]) - float(vals[lo]))

    if len(window_moves) < 10:
        log.warning("%s: only %d windows — skipping", asset, len(window_moves))
        return pd.DataFrame(), 0

    sigma = float(np.std(window_moves))
    log.info("%s: sigma=%.8g  windows=%d", asset, sigma, len(window_moves))

    vol_ratios    = compute_rolling_vol_ratios(windows, coin_series, sigma)
    vol_ratios_3h = compute_rolling_vol_ratios(windows, coin_series, sigma, lookback_secs=3 * 3600)
    vol_ratios_1h  = compute_rolling_vol_ratios(windows, coin_series, sigma, lookback_secs=1 * 3600)

    # Pre-group PM data by window and sort index once — avoids O(N_pm) scan + re-sort per window
    pm_by_window: dict[int, pd.DataFrame] = {
        int(wts): grp.set_index("ts").sort_index()
        for wts, grp in asset_pm.groupby("window_ts")
    }

    records = []
    n_unresolved = 0

    for wts in windows:
        lo = int(np.searchsorted(ts_idx, wts,               side="left"))
        hi = int(np.searchsorted(ts_idx, wts + WINDOW_SECS, side="left"))
        if (hi - lo) < 280:
            continue

        win_ts_arr = ts_idx[lo:hi]
        win_pr_arr = vals[lo:hi]

        open_price   = float(win_pr_arr[0])
        window_move  = float(win_pr_arr[-1]) - open_price
        hour_utc = datetime.fromtimestamp(wts, tz=timezone.utc).hour

        pm_window_idx = pm_by_window.get(wts)
        if pm_window_idx is None or len(pm_window_idx) < 280:
            continue
        resolved_up = _resolve_window(pm_window_idx.reset_index())
        if resolved_up is None:
            # PM price didn't reach 0.95/0.05 — fall back to coin price direction
            if window_move > 0:
                resolved_up = True
            elif window_move < 0:
                resolved_up = False
            else:
                n_unresolved += 1
                continue  # truly flat window, skip

        pm_ts_arr = pm_window_idx.index.values  # sorted int64 timestamps

        for sig in sigma_multiples:
            up_thresh   = open_price + sig * sigma
            down_thresh = open_price - sig * sigma

            # Numpy threshold detection — avoids Python loop over ~300 seconds
            if min_elapsed_secs > 0:
                start_pos = int(np.searchsorted(win_ts_arr, wts + min_elapsed_secs, side="left"))
                search_pr = win_pr_arr[start_pos:]
                search_ts = win_ts_arr[start_pos:]
            else:
                search_pr = win_pr_arr
                search_ts = win_ts_arr

            up_hits = np.where(search_pr >= up_thresh)[0]
            dn_hits = np.where(search_pr <= down_thresh)[0]
            up_trig = int(search_ts[up_hits[0]]) if len(up_hits) else None
            dn_trig = int(search_ts[dn_hits[0]]) if len(dn_hits) else None

            if up_trig is None and dn_trig is None:
                continue
            if up_trig is not None and (dn_trig is None or up_trig <= dn_trig):
                trigger_ts  = up_trig
                trigger_dir = "up"
                price_col   = "up_ask"
                won         = bool(resolved_up)
            else:
                trigger_ts  = dn_trig
                trigger_dir = "down"
                price_col   = "dn_ask"
                won         = not bool(resolved_up)

            if trigger_ts is None:
                continue

            # PM lookups: use sorted index + searchsorted instead of boolean masks
            trig_pos = int(np.searchsorted(pm_ts_arr, trigger_ts, side="left"))
            pm_price_at_trigger: float | None = None
            if trig_pos < len(pm_ts_arr) and price_col in pm_window_idx.columns:
                v = pm_window_idx.iloc[trig_pos].get(price_col)
                if not pd.isna(v):
                    pm_price_at_trigger = float(v)

            fill_pos = int(np.searchsorted(pm_ts_arr, trigger_ts + 1, side="left"))
            if fill_pos >= len(pm_ts_arr):
                fill_pos = 0
            if price_col not in pm_window_idx.columns:
                continue
            pm_row = pm_window_idx.iloc[fill_pos]
            if pd.isna(pm_row.get(price_col)):
                continue
            pm_price = float(pm_row[price_col])

            fill2_pos = int(np.searchsorted(pm_ts_arr, trigger_ts + 2, side="left"))
            if fill2_pos >= len(pm_ts_arr):
                fill2_pos = 0
            pm_price_2s: float | None = None
            if price_col in pm_window_idx.columns:
                v2 = pm_window_idx.iloc[fill2_pos].get(price_col)
                if not pd.isna(v2):
                    pm_price_2s = float(v2)

            # Orderbook imbalance
            imbalance_col = "up_imbalance" if trigger_dir == "up" else "dn_imbalance"
            imbalance_at_trigger: float | None = None
            imbalance_at_fill: float | None = None
            if trig_pos < len(pm_ts_arr) and imbalance_col in pm_window_idx.columns:
                v = pm_window_idx.iloc[trig_pos].get(imbalance_col)
                if not pd.isna(v):
                    imbalance_at_trigger = float(v)
            if imbalance_col in pm_row.index and not pd.isna(pm_row.get(imbalance_col)):
                imbalance_at_fill = float(pm_row[imbalance_col])

            # Velocity lookups: searchsorted on global ts_idx instead of O(N) boolean scan
            _px: dict[int, float | None] = {}
            for offset in (0, 2, 4, 5, 10):
                idx = int(np.searchsorted(ts_idx, trigger_ts - offset, side="right")) - 1
                _px[offset] = float(vals[idx]) if idx >= 0 else None

            def _diff(a: int, b: int) -> float | None:
                pa, pb = _px[a], _px[b]
                return (pa - pb) / sigma if pa is not None and pb is not None else None

            p0, p2, p4, p5, p10 = _px[0], _px[2], _px[4], _px[5], _px[10]
            v2  = _diff(0, 2)
            v10 = _diff(0, 10)

            vel_ratio: float | None = None
            if v2 is not None and v10 is not None and v10 != 0.0:
                vel_ratio = abs(v2) / abs(v10)

            vel_decay: float | None = None
            if v2 is not None and v10 is not None:
                vel_decay = abs(v10) - abs(v2)

            # acc_positive: use window numpy array with searchsorted — no boolean mask
            acc_positive: float | None = None
            sign = 1.0 if trigger_dir == "up" else -1.0
            lo10 = int(np.searchsorted(win_ts_arr, trigger_ts - 10, side="left"))
            hi10 = int(np.searchsorted(win_ts_arr, trigger_ts + 1,  side="left"))
            trig_slice_pr = win_pr_arr[lo10:hi10]
            if len(trig_slice_pr) >= 3:
                per_sec_vel = np.diff(trig_slice_pr) * sign
                per_sec_acc = np.diff(per_sec_vel)
                if len(per_sec_acc) > 0:
                    acc_positive = float((per_sec_acc > 0).mean())

            vels: dict[str, float | None] = {
                "vel_2s":      v2,
                "vel_5s":      _diff(0, 5),
                "vel_10s":     v10,
                "acc_4s":      (p0 - 2*p2 + p4) / sigma  # type: ignore[operator]
                               if p0 is not None and p2 is not None and p4 is not None else None,
                "acc_10s":     (p0 - 2*p5 + p10) / sigma  # type: ignore[operator]
                               if p0 is not None and p5 is not None and p10 is not None else None,
                "vel_ratio":   vel_ratio,
                "vel_decay":   vel_decay,
                "acc_positive": acc_positive,
            }

            records.append({
                "asset":          asset,
                "window_ts":      wts,
                "hour_utc":       hour_utc,
                "window_move":    window_move,
                "sigma":          sig,
                "sigma_abs":      sig * sigma,
                "vol_ratio":      vol_ratios.get(wts),
                "vol_ratio_3h":   vol_ratios_3h.get(wts),
                "vol_ratio_1h":   vol_ratios_1h.get(wts),
                "trigger_dir":    trigger_dir,
                "trigger_ts":          trigger_ts,
                "trigger_second":      trigger_ts - wts,
                "pm_price_at_trigger": pm_price_at_trigger,
                "pm_price":            pm_price,
                "imbalance_at_trigger": imbalance_at_trigger,
                "imbalance_at_fill":    imbalance_at_fill,
                "resolved_up":         resolved_up,
                "won":                 won,
                **vels,
            })

    return pd.DataFrame(records), n_unresolved


# ── Rolling vol ratio ─────────────────────────────────────────────────────────

def compute_rolling_vol_ratios(
    windows: list[int],
    coin_series: pd.Series,
    sigma: float,
    lookback_secs: int = 6 * 3600,
    min_windows: int = 6,
) -> dict[int, float | None]:
    """
    For each window in `windows`, compute realized vol over the prior `lookback_secs`
    of coin windows (strictly before wts), normalized by `sigma`.

    Uses all aligned 5-min coin windows in the lookback range, not just PM windows,
    so the estimate is dense even at the start of the PM data period.

    Returns window_ts → vol_ratio, or None if fewer than `min_windows` prior samples.
    """
    ts_idx = coin_series.index.values
    vals   = coin_series.values

    # Pre-compute moves for every valid coin window across the full coin history
    if len(ts_idx) == 0:
        return {}
    first_ts = int(ts_idx[0])
    last_ts  = int(ts_idx[-1])
    wstart = (first_ts // WINDOW_SECS) * WINDOW_SECS
    if first_ts % WINDOW_SECS != 0:
        wstart += WINDOW_SECS
    wend = (last_ts // WINDOW_SECS) * WINDOW_SECS

    map_keys_list: list[int] = []
    map_vals_list: list[float] = []
    for wts in range(wstart, wend + 1, WINDOW_SECS):
        lo = int(np.searchsorted(ts_idx, wts,               side="left"))
        hi = int(np.searchsorted(ts_idx, wts + WINDOW_SECS, side="left"))
        if (hi - lo) < 280:
            continue
        map_keys_list.append(wts)
        map_vals_list.append(float(vals[hi - 1]) - float(vals[lo]))

    if not map_keys_list:
        return {wts: None for wts in windows}

    # Sorted arrays allow O(log W) lookup per window instead of O(W) dict scan
    map_keys_arr = np.array(map_keys_list)
    map_vals_arr = np.array(map_vals_list)

    result: dict[int, float | None] = {}
    for wts in windows:
        lo = int(np.searchsorted(map_keys_arr, wts - lookback_secs, side="left"))
        hi = int(np.searchsorted(map_keys_arr, wts,                 side="left"))
        if (hi - lo) < min_windows:
            result[wts] = None
        else:
            rolling_sigma = float(np.std(map_vals_arr[lo:hi]))
            result[wts] = rolling_sigma / sigma if sigma > 0 else None

    return result


# ── EWMA sigma ─────────────────────────────────────────────────────────────────

EWMA_WARMUP = 20   # windows to skip before the EWMA estimate is considered reliable
EWMA_LAMBDA = 0.95  # fixed decay factor for all analysis
EWMA_REFRESH_SECS = WINDOW_SECS


def compute_ewma_sigmas(
    windows_sorted: list[int],
    coin_series: pd.Series,
    lambda_: float,
    refresh_secs: int = EWMA_REFRESH_SECS,
) -> dict[int, float]:
    """
    Walk-forward EWMA variance estimator (RiskMetrics-style).

    For window t, returns sigma_t = sqrt(ewma_var) estimated ONLY from moves in
    windows 0 … t-1, so the returned value is safe to use for window t with no
    lookahead bias.

        ewma_var_t = λ · ewma_var_{t-1} + (1-λ) · move_{t-1}²
        sigma_t    = sqrt(ewma_var_t)

    Windows with fewer than 280 coin-price points are skipped and do not update
    the EWMA.  The very first eligible window initialises ewma_var = move_0² but
    is excluded from the returned map (no prior estimate available).
    """
    cadence_windows = max(1, int(refresh_secs // WINDOW_SECS))
    # Extract numpy arrays once — avoids O(N) boolean mask per window
    ts_idx = coin_series.index.values
    vals   = coin_series.values
    ewma_var: float | None = None
    result: dict[int, float] = {}
    eligible_idx = 0

    for wts in windows_sorted:
        lo = int(np.searchsorted(ts_idx, wts,               side="left"))
        hi = int(np.searchsorted(ts_idx, wts + WINDOW_SECS, side="left"))
        if (hi - lo) < 280:
            continue
        move = float(vals[hi - 1]) - float(vals[lo])

        # Sigma available to TRADE this window = what was known before it started
        if ewma_var is not None:
            result[wts] = float(np.sqrt(ewma_var))

        # Refresh the EWMA only on the chosen cadence.
        # refresh_secs=300 -> every 5m window; refresh_secs=1800 -> every 6th window (30m).
        if eligible_idx % cadence_windows == 0:
            if ewma_var is None:
                ewma_var = move ** 2          # seed with first observed move²
            else:
                ewma_var = lambda_ * ewma_var + (1.0 - lambda_) * move ** 2

        eligible_idx += 1

    return result


def analyze_asset_ewma(
    asset: str,
    pm_df: pd.DataFrame,
    coin_series: pd.Series,
    sigma_multiples: list[float],
    lambda_: float,
    refresh_secs: int = EWMA_REFRESH_SECS,
) -> tuple[pd.DataFrame, int]:
    """
    Walk-forward version of analyze_asset().

    Uses a per-window EWMA sigma (estimated from all prior windows only) as the
    volatility baseline instead of the global static std-dev.  The first
    EWMA_WARMUP windows are excluded so the estimate has time to stabilise.

    Returns (records_df, n_unresolved).  records_df has the same core schema as
    analyze_asset() output, plus two extra columns: ``ewma_sigma`` and ``lambda``.
    """
    asset_pm = pm_df[pm_df["asset"] == asset].copy()
    windows = sorted(asset_pm["window_ts"].unique())

    ewma_sigmas = compute_ewma_sigmas(windows, coin_series, lambda_, refresh_secs=refresh_secs)

    # Use mean EWMA sigma as the normalization baseline for vol_ratio
    ewma_vals = [v for v in ewma_sigmas.values() if v > 0]
    baseline_sigma = float(np.mean(ewma_vals)) if ewma_vals else 1.0
    vol_ratios    = compute_rolling_vol_ratios(windows, coin_series, baseline_sigma)
    vol_ratios_3h = compute_rolling_vol_ratios(windows, coin_series, baseline_sigma, lookback_secs=3 * 3600)
    vol_ratios_1h  = compute_rolling_vol_ratios(windows, coin_series, baseline_sigma, lookback_secs=1 * 3600)

    # Extract numpy arrays once — avoids repeated O(N) boolean masks
    ts_idx = coin_series.index.values
    vals   = coin_series.values

    # Pre-group PM data by window and sort index once — avoids O(N_pm) scan + re-sort per window
    pm_by_window: dict[int, pd.DataFrame] = {
        int(wts): grp.set_index("ts").sort_index()
        for wts, grp in asset_pm.groupby("window_ts")
    }

    records = []
    n_unresolved = 0

    for i, wts in enumerate(windows):
        ewma_sigma = ewma_sigmas.get(wts)
        if ewma_sigma is None or ewma_sigma <= 0:
            continue
        if i < EWMA_WARMUP:
            continue

        lo = int(np.searchsorted(ts_idx, wts,               side="left"))
        hi = int(np.searchsorted(ts_idx, wts + WINDOW_SECS, side="left"))
        if (hi - lo) < 280:
            continue

        win_ts_arr = ts_idx[lo:hi]
        win_pr_arr = vals[lo:hi]

        open_price  = float(win_pr_arr[0])
        window_move = float(win_pr_arr[-1]) - open_price
        hour_utc = datetime.fromtimestamp(wts, tz=timezone.utc).hour

        pm_window_idx = pm_by_window.get(wts)
        if pm_window_idx is None or len(pm_window_idx) < 280:
            continue
        resolved_up = _resolve_window(pm_window_idx.reset_index())
        if resolved_up is None:
            # PM price didn't reach 0.95/0.05 — fall back to coin price direction
            if window_move > 0:
                resolved_up = True
            elif window_move < 0:
                resolved_up = False
            else:
                n_unresolved += 1
                continue  # truly flat window, skip

        pm_ts_arr = pm_window_idx.index.values  # sorted int64 timestamps

        for sig in sigma_multiples:
            up_thresh   = open_price + sig * ewma_sigma
            down_thresh = open_price - sig * ewma_sigma

            # Numpy threshold detection — avoids Python loop over ~300 seconds
            up_hits = np.where(win_pr_arr >= up_thresh)[0]
            dn_hits = np.where(win_pr_arr <= down_thresh)[0]
            up_trig = int(win_ts_arr[up_hits[0]]) if len(up_hits) else None
            dn_trig = int(win_ts_arr[dn_hits[0]]) if len(dn_hits) else None

            if up_trig is None and dn_trig is None:
                continue
            if up_trig is not None and (dn_trig is None or up_trig <= dn_trig):
                trigger_ts  = up_trig
                trigger_dir = "up"
                price_col   = "up_ask"
                won         = bool(resolved_up)
            else:
                trigger_ts  = dn_trig
                trigger_dir = "down"
                price_col   = "dn_ask"
                won         = not bool(resolved_up)

            if trigger_ts is None:
                continue

            # PM lookups: use sorted index + searchsorted instead of boolean masks
            trig_pos = int(np.searchsorted(pm_ts_arr, trigger_ts, side="left"))
            pm_price_at_trigger: float | None = None
            if trig_pos < len(pm_ts_arr) and price_col in pm_window_idx.columns:
                v = pm_window_idx.iloc[trig_pos].get(price_col)
                if not pd.isna(v):
                    pm_price_at_trigger = float(v)

            fill_pos = int(np.searchsorted(pm_ts_arr, trigger_ts + 1, side="left"))
            if fill_pos >= len(pm_ts_arr):
                fill_pos = 0
            if price_col not in pm_window_idx.columns:
                continue
            pm_row = pm_window_idx.iloc[fill_pos]
            if pd.isna(pm_row.get(price_col)):
                continue
            pm_price = float(pm_row[price_col])

            fill2_pos = int(np.searchsorted(pm_ts_arr, trigger_ts + 2, side="left"))
            if fill2_pos >= len(pm_ts_arr):
                fill2_pos = 0
            pm_price_2s: float | None = None
            if price_col in pm_window_idx.columns:
                v2 = pm_window_idx.iloc[fill2_pos].get(price_col)
                if not pd.isna(v2):
                    pm_price_2s = float(v2)

            # Orderbook imbalance
            imbalance_col = "up_imbalance" if trigger_dir == "up" else "dn_imbalance"
            imbalance_at_trigger: float | None = None
            imbalance_at_fill: float | None = None
            if trig_pos < len(pm_ts_arr) and imbalance_col in pm_window_idx.columns:
                v = pm_window_idx.iloc[trig_pos].get(imbalance_col)
                if not pd.isna(v):
                    imbalance_at_trigger = float(v)
            if imbalance_col in pm_row.index and not pd.isna(pm_row.get(imbalance_col)):
                imbalance_at_fill = float(pm_row[imbalance_col])

            # Velocity lookups: searchsorted on global ts_idx instead of O(N) boolean scan
            _px: dict[int, float | None] = {}
            for offset in (0, 2, 4, 5, 10):
                idx = int(np.searchsorted(ts_idx, trigger_ts - offset, side="right")) - 1
                _px[offset] = float(vals[idx]) if idx >= 0 else None

            def _diff(a: int, b: int) -> float | None:
                pa, pb = _px[a], _px[b]
                return (pa - pb) / ewma_sigma if pa is not None and pb is not None else None

            p0, p2, p4, p5, p10 = _px[0], _px[2], _px[4], _px[5], _px[10]
            v2  = _diff(0, 2)
            v10 = _diff(0, 10)

            vel_ratio: float | None = None
            if v2 is not None and v10 is not None and v10 != 0.0:
                vel_ratio = abs(v2) / abs(v10)

            vel_decay: float | None = None
            if v2 is not None and v10 is not None:
                vel_decay = abs(v10) - abs(v2)

            # acc_positive: use window numpy array with searchsorted — no boolean mask
            acc_positive: float | None = None
            sign = 1.0 if trigger_dir == "up" else -1.0
            lo10 = int(np.searchsorted(win_ts_arr, trigger_ts - 10, side="left"))
            hi10 = int(np.searchsorted(win_ts_arr, trigger_ts + 1,  side="left"))
            trig_slice_pr = win_pr_arr[lo10:hi10]
            if len(trig_slice_pr) >= 3:
                per_sec_vel = np.diff(trig_slice_pr) * sign
                per_sec_acc = np.diff(per_sec_vel)
                if len(per_sec_acc) > 0:
                    acc_positive = float((per_sec_acc > 0).mean())

            vels: dict[str, float | None] = {
                "vel_2s":      v2,
                "vel_5s":      _diff(0, 5),
                "vel_10s":     v10,
                "acc_4s":      (p0 - 2*p2 + p4) / ewma_sigma  # type: ignore[operator]
                               if p0 is not None and p2 is not None and p4 is not None else None,
                "acc_10s":     (p0 - 2*p5 + p10) / ewma_sigma  # type: ignore[operator]
                               if p0 is not None and p5 is not None and p10 is not None else None,
                "vel_ratio":   vel_ratio,
                "vel_decay":   vel_decay,
                "acc_positive": acc_positive,
            }

            records.append({
                "asset":          asset,
                "window_ts":      wts,
                "hour_utc":       hour_utc,
                "window_move":    window_move,
                "sigma":          sig,
                "sigma_abs":      sig * ewma_sigma,
                "ewma_sigma":     ewma_sigma,
                "lambda":         lambda_,
                "ewma_refresh_secs": refresh_secs,
                "vol_ratio":      vol_ratios.get(wts),
                "vol_ratio_3h":   vol_ratios_3h.get(wts),
                "vol_ratio_1h":   vol_ratios_1h.get(wts),
                "trigger_dir":    trigger_dir,
                "trigger_ts":          trigger_ts,
                "trigger_second":      trigger_ts - wts,
                "pm_price_at_trigger": pm_price_at_trigger,
                "pm_price":            pm_price,
                "pm_price_2s":         pm_price_2s,
                "imbalance_at_trigger": imbalance_at_trigger,
                "imbalance_at_fill":    imbalance_at_fill,
                "resolved_up":         resolved_up,
                "won":                 won,
                **vels,
            })

    return pd.DataFrame(records), n_unresolved


# ── time-to-reprice ───────────────────────────────────────────────────────────

def compute_reprice_times(
    records: pd.DataFrame,
    prices_dir: str,
    target_win_rates: dict,   # (asset, sigma) → actual win rate
) -> pd.DataFrame:
    """
    For each record, find the first second after trigger where pm_price
    has crossed the actual win rate for that (asset, sigma) bucket.
    Returns records with a 'seconds_to_reprice' column added.
    Loads PM data one asset at a time to avoid holding everything in memory.
    """
    result_parts = []

    for asset, asset_records in records.groupby("asset"):
        pm_df = load_prices_for_asset(prices_dir, str(asset))
        pm_idx = pm_df.set_index(["window_ts", "ts"]).sort_index() if not pm_df.empty else None

        out = []
        for _, row in asset_records.iterrows():
            key = (row["asset"], row["sigma"])
            target = target_win_rates.get(key)
            if target is None or pm_idx is None:
                out.append(None)
                continue

            price_col = "up_ask" if row["trigger_dir"] == "up" else "dn_ask"
            try:
                wts = row["window_ts"]
                tts = row["trigger_ts"]
                if wts not in pm_idx.index.get_level_values(0):
                    out.append(None)
                    continue
                pm_window = pm_idx.loc[wts]
                pm_window = pm_window[pm_window.index > tts].sort_index()
            except Exception:
                out.append(None)
                continue

            if pm_window.empty or price_col not in pm_window.columns:
                out.append(None)
                continue

            crossed = pm_window[pm_window[price_col] >= target]
            if crossed.empty:
                out.append(None)
            else:
                out.append(int(crossed.index[0]) - int(tts))

        part = asset_records.copy()
        part["seconds_to_reprice"] = out
        result_parts.append(part)
        del pm_df, pm_idx

    if not result_parts:
        records = records.copy()
        records["seconds_to_reprice"] = None
        return records
    return pd.concat(result_parts).reindex(records.index)


# ── report sections ───────────────────────────────────────────────────────────

def section_summary(df: pd.DataFrame) -> str:
    total_windows = df.groupby("asset")["window_ts"].nunique()

    rows = []
    for key, grp in df.groupby(["asset", "sigma"]):
        asset, sig = key  # type: ignore[misc]
        win = grp["won"].mean()
        avg_pm = grp["pm_price"].mean()
        edge = _edge_from_win_and_pm(win, avg_pm)
        n_total = total_windows.get(asset, len(grp))
        n_fills = grp["window_ts"].nunique()
        fill_rate = n_fills / n_total
        rows.append({
            "asset":            asset,
            "sigma":            sig,
            "n_fills":          n_fills,
            "fill_rate%":       round(fill_rate * 100, 1),
            "win%":             round(win * 100, 1),
            "avg_pm":           round(avg_pm, 4),
            "edge":             round(edge, 4),
            "edge_per_session": round(edge * fill_rate, 4),
        })
    tbl = pd.DataFrame(rows)

    # ── Asset ranking (one line per asset, best sigma entry) ──────────────────
    ranking_lines = ["**Asset ranking by edge/session** (best sigma entry):\n"]
    ranking_rows = []
    for asset, adf in tbl.groupby("asset"):
        best = adf.loc[adf["edge_per_session"].idxmax()]
        avg_eps  = adf.groupby("sigma")["edge_per_session"].mean().max()
        avg_edge = adf.groupby("sigma")["edge"].mean().max()
        avg_fill = adf[adf["sigma"] == best["sigma"]]["fill_rate%"].mean()
        ranking_rows.append((avg_eps, asset, best["sigma"], avg_fill, avg_edge * 100, avg_eps * 100))
    ranking_rows.sort(reverse=True)
    for rank, (_, asset, sig, fill, edge_pct, eps_pct) in enumerate(ranking_rows, 1):
        ranking_lines.append(
            f"{rank}. **{asset}** — {sig}σ entry — "
            f"fill rate {fill:.0f}% — edge/fill {edge_pct:+.1f}% — edge/session {eps_pct:+.2f}%"
        )
    ranking_lines.append("")

    asset_tables = []
    for asset, adf in tbl.sort_values(["asset", "sigma"]).groupby("asset", sort=False):
        asset_tables.append(f"### {asset}\n")
        asset_tables.append(adf.drop(columns="asset").to_markdown(index=False))
        asset_tables.append("")

    return "\n".join(ranking_lines) + "\n" + "\n".join(asset_tables)


def _utc_to_cst(utc_hour: int) -> int:
    """Convert UTC hour to CST hour (UTC-6)."""
    return (utc_hour - 6) % 24



def section_hour_of_day(df: pd.DataFrame) -> str:
    """
    Edge by UTC hour — session buckets, ASCII bar chart, top/bottom hours.
    CST equivalents shown alongside UTC (CST = UTC-6).
    Only includes trades that pass the price filter (pm_price × 1.015 < win_rate).
    """
    df = _price_filter(df)
    if df.empty:
        return "_No trades pass the price filter._"
    out = []

    # ── Aggregate chart across all assets ─────────────────────────────────────
    out.append("### ALL ASSETS (aggregate)\n")
    out.append("**Edge by hour (UTC)** — avg across all assets/sigma — each █ ≈ 1% edge:\n")
    out.append("```")
    agg_hour_edges: dict[int, float] = {}
    for hour in range(24):
        grp = df[df["hour_utc"] == hour]
        if grp.empty:
            continue
        agg_hour_edges[hour] = float((grp["won"] - _price_with_fee(grp["pm_price"])).mean())

    for hour in range(24):
        if hour not in agg_hour_edges:
            continue
        e    = agg_hour_edges[hour]
        n    = len(df[df["hour_utc"] == hour])
        bar_len = max(0, int(abs(e) * 100))
        bar  = ("█" * bar_len) if e >= 0 else ("░" * bar_len)
        sign = "+" if e >= 0 else "-"
        cst  = _utc_to_cst(hour)
        out.append(f"  {hour:02d}h UTC / {cst:02d}h CST  {sign}{abs(e):.3f}  {bar}  (n={n})")
    out.append("```")

    if agg_hour_edges:
        sorted_agg = sorted(agg_hour_edges.items(), key=lambda x: x[1], reverse=True)
        top3    = sorted_agg[:3]
        bottom3 = sorted_agg[-3:]
        out.append(
            "\n**Best hours:**  " +
            "  ".join(f"{h:02d}h UTC/{_utc_to_cst(h):02d}h CST ({e:+.3f})" for h, e in top3)
        )
        out.append(
            "**Worst hours:** " +
            "  ".join(f"{h:02d}h UTC/{_utc_to_cst(h):02d}h CST ({e:+.3f})" for h, e in bottom3)
        )
    out.append("")
    out.append("---")
    out.append("")
    return "\n".join(out)


def section_cascade(df: pd.DataFrame, sigma_levels: list[float]) -> str:
    """
    ASCII flow chain showing fill rate at each sigma level and cascade rates.
    Format: 0.5σ [fill%] --cascade%--> 1.0σ [fill%] --cascade%--> ...
    """
    total_windows = df.groupby("asset")["window_ts"].nunique()
    out = ["```"]

    for asset in sorted(df["asset"].unique()):
        adf = df[df["asset"] == asset]
        n_total = total_windows.get(asset, 1)

        parts = []
        for i, sig in enumerate(sigma_levels):
            windows_at = set(adf[adf["sigma"] == sig]["window_ts"])
            fill_pct = len(windows_at) / n_total * 100

            if i == 0:
                parts.append(f"{sig}σ [{fill_pct:.0f}%]")
            else:
                prev_windows = set(adf[adf["sigma"] == sigma_levels[i-1]]["window_ts"])
                cascade = len(prev_windows & windows_at) / len(prev_windows) * 100 if prev_windows else 0
                parts.append(f"--{cascade:.0f}%--> {sig}σ [{fill_pct:.0f}%]")

        label = f"{asset:4s}:  "
        out.append(label + "  ".join(parts))

    out.append("```")
    out.append("\n_Each node shows fill rate (% of all sessions). Arrows show what % of triggering sessions continue to the next level._")
    return "\n".join(out)


_DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def section_day_of_week(df: pd.DataFrame) -> str:
    """
    Edge and win rate by day of week (UTC), aggregated across all assets and sigmas.
    Only includes trades that pass the price filter (pm_price × 1.015 < win_rate).
    """
    df = _price_filter(df)
    if df.empty:
        return "_No trades pass the price filter._"

    df = df.copy()
    df["dow"] = pd.to_datetime(df["window_ts"], unit="s", utc=True).dt.dayofweek  # 0=Mon

    out = []
    dow_edges: dict[int, tuple[float, float, int]] = {}
    for dow in range(7):
        grp = df[df["dow"] == dow]
        if grp.empty:
            continue
        edge   = float((grp["won"] - _price_with_fee(grp["pm_price"])).mean())
        wr     = float(grp["won"].mean())
        dow_edges[dow] = (edge, wr, len(grp))

    out.append("**Edge by day of week (UTC)** — each █ ≈ 1% edge:\n")
    out.append("```")
    for dow in range(7):
        if dow not in dow_edges:
            continue
        edge, wr, n = dow_edges[dow]
        bar_len = max(0, int(abs(edge) * 100))
        bar  = ("█" * bar_len) if edge >= 0 else ("░" * bar_len)
        sign = "+" if edge >= 0 else "-"
        out.append(f"  {_DOW_NAMES[dow]}  {sign}{abs(edge):.3f}  {bar}  win={wr:.1%}  (n={n})")
    out.append("```")

    if dow_edges:
        sorted_dow = sorted(dow_edges.items(), key=lambda x: x[1][0], reverse=True)
        out.append(
            "\n**Best days:**  " +
            "  ".join(f"{_DOW_NAMES[d]} ({e:+.3f})" for d, (e, _, _) in sorted_dow[:3])
        )
        out.append(
            "**Worst days:** " +
            "  ".join(f"{_DOW_NAMES[d]} ({e:+.3f})" for d, (e, _, _) in sorted_dow[-3:])
        )

    out.append("")
    return "\n".join(out)


def section_half_day(df: pd.DataFrame) -> str:
    """
    Edge and win rate split by half-day (AM = 00–11 UTC, PM = 12–23 UTC),
    further broken down by sigma level. One table per asset.
    CST (UTC-6) date and half shown alongside UTC.
    Only includes trades that pass the price filter (pm_price × 1.015 < win_rate),
    pooled across all assets and sigmas — same as how you actually trade.
    """
    df = _price_filter(df)
    if df.empty:
        return "_No trades pass the price filter._"
    df = df.copy()
    df["date"]     = df["window_ts"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).date())
    df["half"]     = df["hour_utc"].apply(lambda h: "AM" if h < 12 else "PM")
    df["half_day"] = df["date"].astype(str) + " " + df["half"]

    CST_OFFSET = -6
    df["cst_hour"] = (df["hour_utc"] + CST_OFFSET) % 24
    df["cst_date"] = df["window_ts"].apply(
        lambda ts: datetime.fromtimestamp(ts - 6 * 3600, tz=timezone.utc).date()
    )
    df["cst_half"] = df["cst_hour"].apply(lambda h: "AM" if h < 12 else "PM")

    half_day_stats = []
    for half_day in sorted(df["half_day"].unique()):
        hdf  = df[df["half_day"] == half_day]
        date, half = half_day.rsplit(" ", 1)
        n    = hdf["window_ts"].nunique()
        if n < HALF_DAY_MIN_WINDOWS:
            continue
        win  = hdf["won"].mean()
        pm   = hdf["pm_price"].mean()
        edge = _edge_from_win_and_pm(win, pm)
        half_day_stats.append((date, half, n, win, pm, edge))

    out = []

    if not half_day_stats:
        return f"_No half-day buckets with >= {HALF_DAY_MIN_WINDOWS} windows._"

    out.append(f"_Filtered to half-days with >= {HALF_DAY_MIN_WINDOWS} windows._\n")

    # ASCII bar chart
    out.append("```")
    for date, half, n, _, _, edge in half_day_stats:
        bar_len = max(0, int(abs(edge) * 100))
        bar  = ("█" * bar_len) if edge >= 0 else ("░" * bar_len)
        sign = "+" if edge >= 0 else "-"
        flag = " ⚠" if edge < 0 else ""
        out.append(f"  {date} {half}  {sign}{abs(edge):.3f}  {bar}{flag}")
    out.append("```\n")

    return "\n".join(out)


VOL_RATIO_BUCKETS = [
    ("low",    "< 0.75",       lambda r: r < 0.75),
    ("normal", "0.75 – 1.25",  lambda r: (r >= 0.75) & (r <= 1.25)),
    ("high",   "> 1.25",       lambda r: r > 1.25),
]

IMBALANCE_BUCKETS = [
    ("bid-heavy",  "> 0.60",       lambda r: r > 0.60),
    ("balanced",   "0.50 – 0.60",  lambda r: (r >= 0.50) & (r <= 0.60)),
    ("ask-heavy",  "< 0.50",       lambda r: r < 0.50),
]


def _quantile_imbalance_buckets(series: pd.Series) -> tuple[float, float, list[tuple]]:
    """
    Compute tercile breakpoints from the imbalance distribution and return
    (q33, q67, buckets) where buckets is a list of (label, range_str, mask_fn)
    tuples analogous to IMBALANCE_BUCKETS but with even ~N/3 samples per bin.
    """
    q33 = float(series.quantile(0.333))
    q67 = float(series.quantile(0.667))
    buckets = [
        ("bid-heavy", f"> {q67:.2f}", lambda r, q=q67: r > q),
        ("balanced",  f"{q33:.2f} – {q67:.2f}", lambda r, lo=q33, hi=q67: (r >= lo) & (r <= hi)),
        ("ask-heavy", f"< {q33:.2f}", lambda r, q=q33: r < q),
    ]
    return q33, q67, buckets


def section_vol_ratio(df: pd.DataFrame, col: str = "vol_ratio", lookback_label: str = "6h",
                      agg_only: bool = False) -> str:
    """
    Edge by trailing realized vol regime.

    vol_ratio = std(prior-Nh window moves) / global_sigma.
      low    (<0.75) — quiet market, sigma crossings are rarer but stronger signals
      normal (0.75–1.25) — typical conditions
      high   (>1.25) — noisy market, crossings happen on smaller-than-intended moves

    Each window is assigned the vol_ratio computed from the hours strictly before
    it starts, so there is no lookahead bias.
    Only includes trades that pass the price filter (pm_price × 1.015 < win_rate).
    """
    df = _price_filter(df)
    if df.empty:
        return "_No trades pass the price filter._"
    valid = df[df[col].notna()].copy()
    if valid.empty:
        return f"_No {col} data available (need ≥6 prior windows per asset)._"

    total_windows = df.groupby("asset")["window_ts"].nunique()

    out = []
    if not agg_only:
        for asset in sorted(valid["asset"].unique()):
            adf = valid[valid["asset"] == asset]
            n_total = int(total_windows.get(asset, len(adf)))
            out.append(f"### {asset}\n")
            out.append("| sigma | regime | vol_ratio range | n_fills | fill% | win% | avg_pm | edge | edge/session |")
            out.append("|---|---|---|---:|---:|---:|---:|---:|---:|")

            for sig, sgrp in adf.groupby("sigma"):
                for label, range_str, mask_fn in VOL_RATIO_BUCKETS:
                    grp = sgrp[mask_fn(sgrp[col])]
                    if grp.empty:
                        out.append(f"| {sig}σ | {label} | {range_str} | — | — | — | — | — | — |")
                        continue
                    n_fills  = grp["window_ts"].nunique()
                    fill_pct = n_fills / n_total * 100
                    win      = grp["won"].mean()
                    avg_pm   = grp["pm_price"].mean()
                    edge     = _edge_from_win_and_pm(win, avg_pm)
                    eps      = edge * (n_fills / n_total)
                    out.append(
                        f"| {sig}σ | {label} | {range_str} | {n_fills} | {fill_pct:.1f}% |"
                        f" {win*100:.1f}% | {avg_pm:.4f} | {edge:+.4f} | {eps:+.4f} |"
                    )

            out.append("")

    # Aggregate bar chart across all assets at each sigma
    out.append("### All Assets (aggregate)\n")
    out.append("```")
    for sig, sgrp in valid.groupby("sigma"):
        out.append(f"  {sig}σ:")
        for label, range_str, mask_fn in VOL_RATIO_BUCKETS:
            grp = sgrp[mask_fn(sgrp[col])]
            if grp.empty:
                continue
            win  = grp["won"].mean()
            edge = _edge_from_win_and_pm(win, grp["pm_price"].mean())
            bar_len = max(0, int(abs(edge) * 100))
            bar  = ("█" * bar_len) if edge >= 0 else ("░" * bar_len)
            sign = "+" if edge >= 0 else "-"
            out.append(f"    {label:6s} ({range_str:10s})  {sign}{abs(edge):.3f}  {bar}  (n={len(grp)})")
    out.append("```")

    out.append(
        f"\n_vol_ratio = std(prior {lookback_label} window moves) ÷ global σ. "
        "Computed walk-forward: each window uses only data from before it starts. "
        "low = quiet regime, high = noisy regime._"
    )
    return "\n".join(out)


# ── orderbook imbalance analysis ──────────────────────────────────────────────

def section_edge_by_imbalance(df: pd.DataFrame) -> str:
    """
    Edge by orderbook imbalance regime at trigger time.

    imbalance = bid_volume / (bid_volume + ask_volume):
      bid-heavy (>0.60)  — more demand on bid side
      balanced (0.50–0.60) — neutral liquidity distribution
      ask-heavy (<0.50)  — more supply on ask side

    Only includes trades that pass the price filter (pm_price × 1.015 < win_rate).
    """
    df = _price_filter(df)
    if df.empty:
        return "_No trades pass the price filter._"
    valid = df[df["imbalance_at_trigger"].notna()].copy()
    if valid.empty:
        return "_No imbalance data available._"

    q33, q67, buckets = _quantile_imbalance_buckets(valid["imbalance_at_trigger"])

    out = []
    out.append(
        f"_Tercile buckets computed from the full imbalance distribution: "
        f"ask-heavy < {q33:.2f} (bottom third), balanced {q33:.2f}–{q67:.2f} (middle third), "
        f"bid-heavy > {q67:.2f} (top third). Each bucket contains ~equal sample counts._\n"
    )

    # Aggregate bar chart across all assets at each sigma
    out.append("### All Assets (aggregate)\n")
    out.append("```")
    for sig, sgrp in valid.groupby("sigma"):
        out.append(f"  {sig}σ:")
        for label, range_str, mask_fn in buckets:
            grp = sgrp[mask_fn(sgrp["imbalance_at_trigger"])]
            if grp.empty:
                continue
            win  = grp["won"].mean()
            edge = _edge_from_win_and_pm(win, grp["pm_price"].mean())
            bar_len = max(0, int(abs(edge) * 100))
            bar  = ("█" * bar_len) if edge >= 0 else ("░" * bar_len)
            sign = "+" if edge >= 0 else "-"
            out.append(f"    {label:10s} ({range_str:12s})  {sign}{abs(edge):.3f}  {bar}  (n={len(grp)})")
    out.append("```")

    out.append(
        f"\n_imbalance = bid_volume ÷ (bid_volume + ask_volume) at trigger time, for the triggered side. "
        f"Tercile breakpoints: q33={q33:.2f}, q67={q67:.2f} — each bucket contains ~1/3 of all triggers. "
        "Only includes price-filtered trades (pm_price × 1.015 < win_rate)._"
    )
    return "\n".join(out)

# ── execution slippage ───────────────────────────────────────────────────────

def section_slippage(df: pd.DataFrame) -> str:
    """
    1-second execution slippage: change in the relevant ask price from the trigger second (t)
    to the fill second (t+1).  Positive = market repriced against us; negative = in our favour.
    This is purely the price movement in that 1 second — the 1.5% fee is NOT included here
    (it appears in the edge/PnL tables).
    Also compares aggregate 1-second vs 2-second slippage across all assets.
    Only includes trades that pass the price filter (pm_price × 1.015 < win_rate).
    """
    df = _price_filter(df)
    if df.empty:
        return "_No trades pass the price filter._"
    valid = df[df["pm_price_at_trigger"].notna() & df["pm_price"].notna()].copy()
    if valid.empty:
        return "_No slippage data available._"

    valid["slippage"]     = valid["pm_price"] - valid["pm_price_at_trigger"]
    valid["slippage_pct"] = (valid["slippage"] / valid["pm_price_at_trigger"] * 100).replace(
        [float("inf"), float("-inf")], float("nan")
    )

    has_2s = "pm_price_2s" in valid.columns
    valid_2s = valid[valid["pm_price_2s"].notna()].copy() if has_2s else pd.DataFrame()
    if not valid_2s.empty:
        valid_2s["slippage_2s"] = valid_2s["pm_price_2s"] - valid_2s["pm_price_at_trigger"]

    out = []
    for asset in sorted(valid["asset"].unique()):
        adf = valid[valid["asset"] == asset]
        out.append(f"### {asset}\n")
        out.append("| sigma | n | ask@trigger | ask@fill | avg_slip | median_slip | p75_slip | adverse% |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|")

        for sig, grp in adf.groupby("sigma"):
            slip = grp["slippage"]
            out.append(
                f"| {sig}σ | {len(grp)} |"
                f" {grp['pm_price_at_trigger'].mean():.4f} |"
                f" {grp['pm_price'].mean():.4f} |"
                f" {slip.mean():+.5f} |"
                f" {slip.median():+.5f} |"
                f" {slip.quantile(0.75):+.5f} |"
                f" {(slip > 0).mean()*100:.0f}% |"
            )
        out.append("")

    # aggregate comparison across all assets / sigmas
    out.append("### All Assets (aggregate: 1s vs 2s)\n")
    out.append("| horizon | n | ask@trigger | ask@fill | avg_slip | median_slip | p75_slip | adverse% | flat% | favourable% |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    slip_1s = valid["slippage"]
    out.append(
        f"| 1s (t+1) | {len(valid)} |"
        f" {valid['pm_price_at_trigger'].mean():.4f} |"
        f" {valid['pm_price'].mean():.4f} |"
        f" {slip_1s.mean():+.5f} |"
        f" {slip_1s.median():+.5f} |"
        f" {slip_1s.quantile(0.75):+.5f} |"
        f" {(slip_1s > 0).mean()*100:.0f}% |"
        f" {(slip_1s == 0).mean()*100:.0f}% |"
        f" {(slip_1s < 0).mean()*100:.0f}% |"
    )

    if not valid_2s.empty:
        slip_2s = valid_2s["slippage_2s"]
        out.append(
            f"| 2s (t+2) | {len(valid_2s)} |"
            f" {valid_2s['pm_price_at_trigger'].mean():.4f} |"
            f" {valid_2s['pm_price_2s'].mean():.4f} |"
            f" {slip_2s.mean():+.5f} |"
            f" {slip_2s.median():+.5f} |"
            f" {slip_2s.quantile(0.75):+.5f} |"
            f" {(slip_2s > 0).mean()*100:.0f}% |"
            f" {(slip_2s == 0).mean()*100:.0f}% |"
            f" {(slip_2s < 0).mean()*100:.0f}% |"
        )
    else:
        out.append("| 2s (t+2) | 0 | — | — | — | — | — | — | — | — |")

    slip_pct = valid["slippage_pct"].dropna()

    out.append(
        f"\n_avg_slip in PM-price units (0–1 scale). avg_slip_pct ≈ {slip_pct.mean():+.3f}% of ask. "
        "adverse = PM repriced against you; flat = no movement; favourable = moved in your direction. "
        "Per-asset tables are 1s fills (t+1); aggregate table compares t+1 vs t+2._"
    )
    return "\n".join(out)


# ── report builder ────────────────────────────────────────────────────────────

def build_report(
    df: pd.DataFrame,
    sigma_levels: list[float],
    unresolved: dict[str, int] | None = None,
) -> str:
    # compute actual win rates per bucket for reprice analysis
    target_win_rates = {}
    for key, grp in df.groupby(["asset", "sigma"]):
        asset, sig = key  # type: ignore[misc]
        target_win_rates[(asset, sig)] = grp["won"].mean()

    unresolved = unresolved or {}
    total_windows = df.groupby("asset")["window_ts"].nunique()
    unresolved_rows = []
    for asset in sorted(set(list(total_windows.index) + list(unresolved.keys()))):
        n_res   = int(total_windows.get(asset, 0))
        n_unres = unresolved.get(asset, 0)
        n_total = n_res + n_unres
        pct     = f"{n_unres / n_total * 100:.1f}%" if n_total > 0 else "—"
        unresolved_rows.append(f"| {asset} | {n_total} | {n_res} | {n_unres} | {pct} |")

    unresolved_table = "\n".join([
        "| asset | total windows | resolved | unresolved | unresolved% |",
        "|---|---:|---:|---:|---:|",
        *unresolved_rows,
    ])

    lines = [
        "# Threshold Edge Report",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "---",
        "",
        "## Overview",
        "",
        "Each row represents windows where the coin price crossed `window_open ± N×σ`",
        f"(σ = EWMA walk-forward volatility estimate, λ={EWMA_LAMBDA}, updated each window from prior history only).",
        "We take the **first** crossing in either direction and bet accordingly.",
        "**Entry filter**: only trade when `pm_price × 1.015 < win_rate` (price filter applied throughout).",
        "**Edge** = actual win rate − effective entry cost (ask at `trigger_ts + 1` + 1.5% buy fee), filled sessions only.",
        "Entry price uses `up_ask` (UP triggers) / `dn_ask` (DOWN triggers) at `trigger_ts + 1`.",
        "**Edge/session** = edge × fill_rate — expected value per session regardless of whether a trigger fires.",
        f"First {EWMA_WARMUP} windows per asset are excluded while the EWMA estimate warms up.",
        "",
        "**Resolution**: last PM price ≥ 0.95 → UP won; ≤ 0.05 → DOWN won. "
        "If PM price is ambiguous, the coin price direction (close vs open) determines the outcome. "
        "Only windows with a flat coin move (close == open) remain unresolved.",
        "",
        "**Unresolved windows** (coin move exactly flat — excluded from all analysis):",
        "",
        unresolved_table,
        "",
        "---",
        "",
        "## Summary — Edge by Asset / Sigma",
        "",
        section_summary(df),
        "",
        "---",
        "",
        "## Hour of Day (UTC) — When Is the Edge Largest?",
        "",
        "Thinner hours may have slower Polymarket repricing → more edge.",
        "",
        section_hour_of_day(df),
        "",
        "---",
        "",
        "## Day of Week — Does the Day Matter?",
        "",
        "Aggregated across all assets and sigmas. Trades that pass the price filter only.",
        "",
        section_day_of_week(df),
        "",
        "---",
        "",
        "## Cascade Rate — If Coin Crosses 0.5σ, How Often Does It Reach Higher Sigmas?",
        "",
        "Shows momentum continuation. High cascade% at lower base sigmas means",
        "a small initial move is a reliable precursor to a larger one.",
        "",
        section_cascade(df, sigma_levels),
        "",
        "---",
        "",
        "## Edge by Orderbook Imbalance — Does Market Structure Matter?",
        "",
        "Does performance vary with the orderbook imbalance (bid vs ask volume)?",
        "High imbalance (>0.65) = bid-heavy (more demand). ",
        "Low imbalance (<0.45) = ask-heavy (more supply).",
        "",
        section_edge_by_imbalance(df),
        "",
        "---",
        "",
        "## 1-Second Execution Slippage — Cost of Fill Delay",
        "",
        "How much does the Polymarket ask price move in the 1 second between signal and fill?",
        "`adverse` = PM repriced against us before our order landed.",
        "Does NOT include the 1.5% fee (see edge/PnL tables for that).",
        "",
        section_slippage(df),
        "",
        "---",
        "",
        "## Edge by Half-Day — Is It Profitable on All Days?",
        "",
        "Each half-day = 12-hour block (AM = 00:00–11:59 UTC, PM = 12:00–23:59 UTC).",
        "Rows sorted oldest → newest. Negative edge rows are highlighted with `[!]`.",
        "",
        section_half_day(df),
        "",
        "---",
        "",
        "## Recent Backtest — 4h / 12h / 24h",
        "",
        "Recommended strategy (best sigma per asset) tested on the most recent windows.",
        "Price filter applied throughout: only entries where `pm_price × 1.015 < win_rate`.",
        "Edge/session = edge × fill% — expected value per window whether or not a trigger fires.",
        "",
        section_recent_backtest(df),
        "",
        "---",
        "",
        "## Config YAML",
        "",
        "Best entry per asset based on highest `edge_per_session`.",
        "Copy-paste into `config/assets.yaml`.",
        "",
        _config_yaml(df),
    ]
    return "\n".join(lines)


def section_recent_backtest(df: pd.DataFrame) -> str:
    """
    For each asset's recommended strategy (best sigma by edge/session), test on the
    last 4h / 12h / 24h of data (price-filtered only).
    """
    HORIZONS = [("4h", 48), ("12h", 144), ("24h", 288)]

    total_windows = df.groupby("asset")["window_ts"].nunique()
    out = [
        "_Price filter applied: only entries where `pm_price × 1.015 < win_rate`. "
        "Win rate (hurdle) = observed win rate over the full dataset at recommended sigma. "
        "Edge/session = edge × fill%._",
        "",
    ]

    for asset in sorted(df["asset"].unique()):
        adf = df[df["asset"] == asset]
        n_total_all = int(total_windows.get(asset, 1))

        # pick recommended sigma (best edge/session over full dataset)
        best_sig = None
        best_eps = -999.0
        for sig, grp in adf.groupby("sigma"):
            win    = grp["won"].mean()
            avg_pm = grp["pm_price"].mean()
            edge   = _edge_from_win_and_pm(win, avg_pm)
            eps    = edge * (grp["window_ts"].nunique() / n_total_all)
            if eps > best_eps:
                best_eps = eps
                best_sig = sig

        if best_sig is None:
            continue

        full_grp = adf[adf["sigma"] == best_sig]
        win_rate = full_grp["won"].mean()
        all_windows = sorted(adf["window_ts"].unique())

        out.append(f"### {asset}  (recommended: {best_sig}σ | hurdle: {win_rate:.3f})\n")
        out.append("| horizon | fills (fill%) | win% | avg_price | edge | edge/session |")
        out.append("|---|---|---:|---:|---:|---:|")

        for label, n_windows in HORIZONS:
            recent_windows = set(all_windows[-n_windows:])
            recent = full_grp[full_grp["window_ts"].isin(recent_windows)]
            n_recent = len(recent_windows)
            taken = recent[_price_with_fee(recent["pm_price"]) < win_rate]

            if taken.empty:
                out.append(f"| {label} | — | — | — | — | — |")
            else:
                n_fills  = len(taken)
                fill_pct = n_fills / n_recent * 100
                w   = taken["won"].mean()
                p   = taken["pm_price"].mean()
                e   = _edge_from_win_and_pm(w, p)
                eps = e * (n_fills / n_recent)
                out.append(
                    f"| {label} | {n_fills} ({fill_pct:.0f}%) | {w*100:.1f}% | {p:.3f} | {e:+.4f} | {eps:+.4f} |"
                )

        out.append("")

    return "\n".join(out)


def analyze_asset_multi_entry_ewma(
    asset: str,
    pm_df: pd.DataFrame,
    coin_series: pd.Series,
    sigma_multiples: list[float],
    lambda_: float,
    win_rates_by_key: dict,  # (asset, sigma, direction) → float
) -> pd.DataFrame:
    """
    Find ALL threshold crossings per window per sigma (both directions).
    A crossing is detected each time the coin price *transitions* from one side
    of the threshold to the other (up: below → at/above; down: above → at/below).
    Multiple entries per window are allowed, including opposing-direction entries.

    For each crossing the price filter is applied:
        entry if pm_price × 1.015 < win_rate[asset, sigma, direction]

    Returns a flat DataFrame with one row per crossing (all crossings, not just
    price-filtered ones — use the `entry_taken` column to filter).

    Columns:
        asset, window_ts, sigma, trigger_dir, trigger_ts, trigger_second,
        pm_price, won, entry_taken, pnl
    """
    asset_pm = pm_df[pm_df["asset"] == asset].copy()
    windows = sorted(asset_pm["window_ts"].unique())

    ewma_sigmas = compute_ewma_sigmas(windows, coin_series, lambda_)

    ts_idx = coin_series.index.values
    vals   = coin_series.values

    pm_by_window: dict[int, pd.DataFrame] = {
        int(wts): grp.set_index("ts").sort_index()
        for wts, grp in asset_pm.groupby("window_ts")
    }

    records = []

    for i, wts in enumerate(windows):
        ewma_sigma = ewma_sigmas.get(wts)
        if ewma_sigma is None or ewma_sigma <= 0:
            continue
        if i < EWMA_WARMUP:
            continue

        lo = int(np.searchsorted(ts_idx, wts,               side="left"))
        hi = int(np.searchsorted(ts_idx, wts + WINDOW_SECS, side="left"))
        if (hi - lo) < 280:
            continue

        win_ts_arr = ts_idx[lo:hi]
        win_pr_arr = vals[lo:hi]

        open_price  = float(win_pr_arr[0])
        window_move = float(win_pr_arr[-1]) - open_price

        pm_window_idx = pm_by_window.get(wts)
        if pm_window_idx is None or len(pm_window_idx) < 280:
            continue
        resolved_up = _resolve_window(pm_window_idx.reset_index())
        if resolved_up is None:
            if window_move > 0:
                resolved_up = True
            elif window_move < 0:
                resolved_up = False
            else:
                continue  # truly flat, skip

        pm_ts_arr = pm_window_idx.index.values

        for sig in sigma_multiples:
            up_thresh   = open_price + sig * ewma_sigma
            dn_thresh   = open_price - sig * ewma_sigma

            win_rate_up = win_rates_by_key.get((asset, sig, "up"))
            win_rate_dn = win_rates_by_key.get((asset, sig, "down"))

            # Detect all crossing transitions in this window
            for j in range(1, len(win_pr_arr)):
                p_prev = float(win_pr_arr[j - 1])
                p_curr = float(win_pr_arr[j])
                t_curr = int(win_ts_arr[j])

                for direction in ("up", "down"):
                    if direction == "up":
                        crossed = (p_prev < up_thresh) and (p_curr >= up_thresh)
                        price_col = "up_ask"
                        won = bool(resolved_up)
                        win_rate = win_rate_up
                    else:
                        crossed = (p_prev > dn_thresh) and (p_curr <= dn_thresh)
                        price_col = "dn_ask"
                        won = not bool(resolved_up)
                        win_rate = win_rate_dn

                    if not crossed:
                        continue

                    # PM price lookup at trigger + 1
                    fill_pos = int(np.searchsorted(pm_ts_arr, t_curr + 1, side="left"))
                    if fill_pos >= len(pm_ts_arr):
                        fill_pos = 0
                    if price_col not in pm_window_idx.columns:
                        continue
                    pm_row = pm_window_idx.iloc[fill_pos]
                    if pd.isna(pm_row.get(price_col)):
                        continue
                    pm_price = float(pm_row[price_col])

                    entry_taken = (
                        win_rate is not None
                        and _price_with_fee(pm_price) < win_rate
                    )
                    pnl = (1.0 - _price_with_fee(pm_price)) if (won and entry_taken) \
                          else (-_price_with_fee(pm_price))   if (not won and entry_taken) \
                          else 0.0

                    records.append({
                        "asset":          asset,
                        "window_ts":      wts,
                        "sigma":          sig,
                        "trigger_dir":    direction,
                        "trigger_ts":     t_curr,
                        "trigger_second": t_curr - wts,
                        "pm_price":       pm_price,
                        "resolved_up":    resolved_up,
                        "won":            won,
                        "entry_taken":    entry_taken,
                        "pnl":            pnl,
                    })

    return pd.DataFrame(records)


def section_multi_entry_pnl(multi_df: pd.DataFrame, first_df: pd.DataFrame) -> str:
    """
    PnL metrics for the multi-entry strategy (every threshold crossing that passes
    the price filter), compared against the single-entry (first crossing only) baseline.

    For each (asset, sigma):
        n_windows, n_triggers, n_entries, entries/window,
        win%, avg_pm, avg_pnl/entry, total_pnl, pnl/window

    Also shows an aggregate summary across all assets by sigma.
    """
    if multi_df.empty:
        return "_No multi-entry data available._"

    out = []

    # ── Per-asset tables ───────────────────────────────────────────────────────
    total_windows = multi_df.groupby("asset")["window_ts"].nunique()

    for asset in sorted(multi_df["asset"].unique()):
        adf = multi_df[multi_df["asset"] == asset]
        n_windows = int(total_windows.get(asset, 0))

        out.append(f"### {asset}\n")
        out.append(
            "| sigma | n_windows | n_triggers | n_entries | entries/window"
            " | win% | avg_pm | avg_pnl/entry | total_pnl | pnl/window |"
        )
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

        for sig, sgrp in adf.groupby("sigma"):
            n_triggers = len(sgrp)
            taken      = sgrp[sgrp["entry_taken"]]
            n_entries  = len(taken)
            if n_entries == 0:
                out.append(
                    f"| {sig}σ | {n_windows} | {n_triggers} | 0 | 0.00"
                    f" | — | — | — | — | — |"
                )
                continue
            entries_pw = n_entries / n_windows
            win_pct    = taken["won"].mean() * 100
            avg_pm     = taken["pm_price"].mean()
            avg_pnl    = taken["pnl"].mean()
            total_pnl  = taken["pnl"].sum()
            pnl_pw     = total_pnl / n_windows
            out.append(
                f"| {sig}σ | {n_windows} | {n_triggers} | {n_entries}"
                f" | {entries_pw:.2f}"
                f" | {win_pct:.1f}%"
                f" | {avg_pm:.4f}"
                f" | {avg_pnl:+.4f}"
                f" | {total_pnl:+.3f}"
                f" | {pnl_pw:+.4f} |"
            )
        out.append("")

    # ── Aggregate by sigma (all assets) ───────────────────────────────────────
    out.append("### All Assets (aggregate)\n")
    out.append(
        "| sigma | n_windows | n_triggers | n_entries | entries/window"
        " | win% | avg_pm | avg_pnl/entry | total_pnl | pnl/window |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    agg_n_windows = int(multi_df.groupby("asset")["window_ts"].nunique().sum())
    for sig, sgrp in multi_df.groupby("sigma"):
        n_triggers = len(sgrp)
        taken      = sgrp[sgrp["entry_taken"]]
        n_entries  = len(taken)
        if n_entries == 0:
            out.append(f"| {sig}σ | {agg_n_windows} | {n_triggers} | 0 | — | — | — | — | — | — |")
            continue
        entries_pw = n_entries / agg_n_windows
        win_pct    = taken["won"].mean() * 100
        avg_pm     = taken["pm_price"].mean()
        avg_pnl    = taken["pnl"].mean()
        total_pnl  = taken["pnl"].sum()
        pnl_pw     = total_pnl / agg_n_windows
        out.append(
            f"| {sig}σ | {agg_n_windows} | {n_triggers} | {n_entries}"
            f" | {entries_pw:.2f}"
            f" | {win_pct:.1f}%"
            f" | {avg_pm:.4f}"
            f" | {avg_pnl:+.4f}"
            f" | {total_pnl:+.3f}"
            f" | {pnl_pw:+.4f} |"
        )
    out.append("")

    # ── Multi vs single comparison (aggregate, all assets all sigmas) ─────────
    if not first_df.empty:
        out.append("### Multi-entry vs First-only (all assets · all sigmas)\n")
        out.append("| strategy | n_entries | win% | avg_pnl/entry | total_pnl |")
        out.append("|---|---:|---:|---:|---:|")

        multi_taken = multi_df[multi_df["entry_taken"]]
        if not multi_taken.empty:
            out.append(
                f"| multi-entry | {len(multi_taken)}"
                f" | {multi_taken['won'].mean()*100:.1f}%"
                f" | {multi_taken['pnl'].mean():+.4f}"
                f" | {multi_taken['pnl'].sum():+.3f} |"
            )

        # reconstruct first-only pnl from first_df using same fee logic
        first_filt = _price_filter(first_df)
        if not first_filt.empty:
            first_pnl = (first_filt["won"].astype(float) - _price_with_fee(first_filt["pm_price"]))
            out.append(
                f"| first-only  | {len(first_filt)}"
                f" | {first_filt['won'].mean()*100:.1f}%"
                f" | {first_pnl.mean():+.4f}"
                f" | {first_pnl.sum():+.3f} |"
            )

        out.append("")

    out.append(
        "_Entry condition: every coin-price crossing of `window_open ± N×σ` "
        "(transition from one side to the other) where `pm_price × 1.015 < directional win_rate`. "
        "Opposing-direction entries in the same window are allowed. "
        "pnl/entry = (1 − pm_price × 1.015) if won else (−pm_price × 1.015). "
        "pnl/window = total_pnl ÷ n_windows (includes windows with no entries)._"
    )
    return "\n".join(out)


def section_ewma_cadence_comparison(fast_df: pd.DataFrame, slow_df: pd.DataFrame) -> str:
    """
    Compare the current 5-minute EWMA cadence against a 30-minute refresh cadence.

    Uses the same price filter as the rest of the report: only entries where
    pm_price × 1.015 < win_rate for the respective cadence / asset / sigma bucket.
    """
    if fast_df.empty or slow_df.empty:
        return "_No 30-minute EWMA comparison data available._"

    HORIZONS = [("4h", 48), ("12h", 144), ("24h", 288)]

    def _evaluate(df: pd.DataFrame, label: str) -> dict[str, dict[str, float | int | None]]:
        out: dict[str, dict[str, float | int | None]] = {}
        total_windows = df.groupby("asset")["window_ts"].nunique()
        for horizon_label, n_windows in HORIZONS:
            parts = []
            n_trades = 0
            for asset in sorted(df["asset"].unique()):
                adf = df[df["asset"] == asset]
                n_total_all = int(total_windows.get(asset, 1))

                best_sig = None
                best_eps = -999.0
                for sig, grp in adf.groupby("sigma"):
                    win = grp["won"].mean()
                    avg_pm = grp["pm_price"].mean()
                    edge = _edge_from_win_and_pm(win, avg_pm)
                    eps = edge * (grp["window_ts"].nunique() / n_total_all)
                    if eps > best_eps:
                        best_eps = eps
                        best_sig = sig

                if best_sig is None:
                    continue

                full_grp = adf[adf["sigma"] == best_sig]
                win_rate = full_grp["won"].mean()
                all_windows = sorted(adf["window_ts"].unique())
                recent_windows = set(all_windows[-n_windows:])
                recent = full_grp[full_grp["window_ts"].isin(recent_windows)]
                taken = recent[_price_with_fee(recent["pm_price"]) < win_rate]
                if taken.empty:
                    continue

                parts.append(taken)
                n_trades += len(taken)

            if not parts:
                out[horizon_label] = {"edge": None, "eps": None, "n": 0}
                continue

            agg = pd.concat(parts, ignore_index=True)
            win = agg["won"].mean()
            avg_pm = agg["pm_price"].mean()
            edge = _edge_from_win_and_pm(win, avg_pm)
            n_windows_total = int(df.groupby("asset")["window_ts"].nunique().sum())
            eps = edge * (len(agg) / n_windows_total) if n_windows_total else None
            out[horizon_label] = {"edge": edge, "eps": eps, "n": len(agg)}
        return out

    fast = _evaluate(fast_df, "5m")
    slow = _evaluate(slow_df, "30m")

    out = []
    out.append("### All Assets (aggregate, price-filtered)")
    out.append("| horizon | 5m cadence edge | 5m cadence edge/session | 30m cadence edge | 30m cadence edge/session |")
    out.append("|---|---:|---:|---:|---:|")
    for horizon_label, _ in HORIZONS:
        f = fast.get(horizon_label, {})
        s = slow.get(horizon_label, {})
        f_edge = f.get("edge")
        f_eps  = f.get("eps")
        s_edge = s.get("edge")
        s_eps  = s.get("eps")
        out.append(
            f"| {horizon_label} | "
            f"{('—' if f_edge is None else f'{float(f_edge):+.4f}')} | "
            f"{('—' if f_eps is None else f'{float(f_eps):+.4f}')} | "
            f"{('—' if s_edge is None else f'{float(s_edge):+.4f}')} | "
            f"{('—' if s_eps is None else f'{float(s_eps):+.4f}')} |"
        )
    out.append("")
    out.append(
        "_Price-filter only. 5m cadence = current EWMA refresh each 5-minute window. "
        "30m cadence = EWMA refresh every 30 minutes (6 windows). "
        "edge/session is the per-session expected value after fill-rate._"
    )
    return "\n".join(out)


def _config_yaml(df: pd.DataFrame) -> str:
    """
    For each asset, emit YAML config blocks for sigma_entry 0.3 and 0.5.
    """
    SHOW_SIGMAS   = [0.3, 0.5]
    total_windows = df.groupby("asset")["window_ts"].nunique()
    blocks = []

    for asset in sorted(df["asset"].unique()):
        adf = df[df["asset"] == asset]

        # Use the most recent window's EWMA sigma as sigma_value (current volatility estimate)
        if "ewma_sigma" in adf.columns:
            latest_wts  = adf["window_ts"].max()
            sigma_value = float(adf[adf["window_ts"] == latest_wts]["ewma_sigma"].iloc[0])
        else:
            ref         = adf[adf["sigma"] > 0].iloc[0]
            sigma_value = ref["sigma_abs"] / ref["sigma"]

        n_total    = total_windows.get(asset, 1)
        sig_stats  = {}
        for sig, sgrp in adf.groupby("sigma"):
            if sig not in SHOW_SIGMAS:
                continue
            win           = sgrp["won"].mean()
            avg_pm        = sgrp["pm_price"].mean()
            edge          = _edge_from_win_and_pm(win, avg_pm)
            fill_rate     = sgrp["window_ts"].nunique() / n_total
            eps           = edge * fill_rate
            sig_stats[sig] = dict(win=win, edge=edge, eps=eps)

        if not sig_stats:
            continue

        lines = []
        for sig in SHOW_SIGMAS:
            if sig not in sig_stats:
                continue
            s = sig_stats[sig]
            lines.append(
                f"{asset}:  # entry: {sig}σ  |  edge/fill: {s['edge']*100:+.1f}%  |  edge/session: {s['eps']*100:+.2f}%\n"
                f"  sigma_value: {sigma_value:.8g}\n"
                f"  sigma_entry: {sig}\n"
                f"  max_pm_price: {s['win']/(1.0 + BUY_FEE_RATE):.2f}\n"
                f"  name: mom_{asset.strip().lower()}"
            )
        blocks.append("\n\n".join(lines))

    return "```yaml\n" + "\n\n".join(blocks) + "\n```"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--assets",     nargs="+", default=list(ASSET_TO_SYMBOL.keys()))
    p.add_argument("--prices-dir", default="data/prices")
    p.add_argument("--coin-dir",   default="data/coin_prices")
    p.add_argument("--sigma",      nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.75])
    p.add_argument("--out-csv",    default="data/reports/threshold_edge.csv")
    p.add_argument("--out-report", default="data/reports/threshold_edge.md")
    p.add_argument(
        "--from-csv", action="store_true", default=False,
        help="Skip data loading/analysis and regenerate the report from the existing --out-csv file",
    )
    p.add_argument(
        "--ewma-only", action="store_true", default=False,
        help="Print only the most recent EWMA sigma values per asset and exit (no full analysis)",
    )
    p.add_argument(
        "--ewma-n", type=int, default=5,
        help="Number of most recent EWMA values to show per asset (default: 5)",
    )
    p.add_argument(
        "--update-config", action="store_true", default=False,
        help="With --ewma-only: write the latest EWMA sigma back to --config as sigma_value",
    )
    p.add_argument(
        "--config", default="config/assets.yaml",
        help="Path to assets.yaml to update when --update-config is set (default: config/assets.yaml)",
    )
    p.add_argument(
        "--ewma-lambda", type=float, default=EWMA_LAMBDA,
        help=f"EWMA decay factor λ (default: {EWMA_LAMBDA})",
    )
    return p.parse_args()


def _print_ewma_only(
    assets: list[str],
    coin_dir: str,
    lambda_: float,
    n: int,
    update_config: str | None = None,
) -> None:
    """Load coin prices per asset, compute EWMA sigmas, and print the most recent n values.

    If update_config is a path to assets.yaml, writes the latest EWMA sigma for each
    asset back as sigma_value in that file.
    """
    rows = []
    latest_sigma: dict[str, float] = {}
    for asset in assets:
        coin = load_coin_prices(coin_dir, asset)
        if coin is None:
            continue
        ts_idx = coin.index.values
        if len(ts_idx) == 0:
            continue
        first_ts = int(ts_idx[0])
        last_ts  = int(ts_idx[-1])
        wstart = (first_ts // WINDOW_SECS) * WINDOW_SECS
        if first_ts % WINDOW_SECS != 0:
            wstart += WINDOW_SECS
        wend = (last_ts // WINDOW_SECS) * WINDOW_SECS
        windows = list(range(wstart, wend + 1, WINDOW_SECS))
        ewma_sigmas = compute_ewma_sigmas(windows, coin, lambda_)
        if not ewma_sigmas:
            continue
        recent = sorted(ewma_sigmas.items())[-n:]
        latest_sigma[asset] = round(recent[-1][1], 8)
        for wts, sigma in recent:
            dt = datetime.fromtimestamp(wts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            rows.append({"asset": asset, "window_utc": dt, "ewma_sigma": round(sigma, 8)})

    if not rows:
        print("No EWMA data available.")
        return

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    if update_config and latest_sigma:
        _update_config_sigmas(update_config, latest_sigma)


def _update_config_sigmas(config_path: str, latest_sigma: dict[str, float]) -> None:
    """Write sigma_value back into assets.yaml for each asset in latest_sigma."""
    import re
    try:
        text = open(config_path).read()
    except FileNotFoundError:
        print(f"[ewma] Config not found: {config_path}")
        return

    updated: list[str] = []
    for asset, sigma in latest_sigma.items():
        # Replace the sigma_value line under the asset's block.
        # Pattern: asset header, then somewhere within a few lines, sigma_value: <old>
        # We do a targeted line-by-line replacement to avoid clobbering YAML structure.
        new_text = re.sub(
            rf"(?m)^({re.escape(asset.upper())}:.*\n(?:[ \t]+.*\n)*?[ \t]+sigma_value:[ \t]*)[^\n]+",
            lambda m: m.group(1) + str(sigma),
            text,
        )
        if new_text != text:
            text = new_text
            updated.append(f"  {asset}: sigma_value → {sigma}")
        else:
            print(f"[ewma] Could not find sigma_value for {asset} in {config_path} — skipping")

    if updated:
        with open(config_path, "w") as f:
            f.write(text)
        print(f"\n[ewma] Updated {config_path}:")
        for line in updated:
            print(line)


def main() -> None:
    args = parse_args()

    lambda_ = args.ewma_lambda

    if args.ewma_only:
        _print_ewma_only(
            args.assets, args.coin_dir, lambda_, args.ewma_n,
            update_config=args.config if args.update_config else None,
        )
        return

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    if args.from_csv:
        if not os.path.exists(args.out_csv):
            log.error("--from-csv specified but %s does not exist", args.out_csv)
            sys.exit(1)
        full = pd.read_csv(args.out_csv)
        log.info("Loaded %d records from %s", len(full), args.out_csv)
        unresolved: dict[str, int] = {}
    else:
        all_records = []
        unresolved = {}

        for asset in args.assets:
            log.info("Loading data for %s…", asset)
            pm_df = load_prices_for_asset(args.prices_dir, asset)
            if pm_df.empty:
                log.warning("%s: no PM price data — skipping", asset)
                continue

            coin = load_coin_prices(args.coin_dir, asset)
            if coin is None:
                continue

            recs, n_unresolved = analyze_asset_ewma(asset, pm_df, coin, args.sigma, lambda_)
            unresolved[asset] = n_unresolved
            if not recs.empty:
                all_records.append(recs)

            del coin, pm_df  # free memory before loading next asset

        if not all_records:
            log.error("No data.")
            sys.exit(1)

        full = pd.concat(all_records, ignore_index=True)

        # compute target win rates then add reprice times
        target_win_rates = {}
        for key, grp in full.groupby(["asset", "sigma"]):
            asset, sig = key  # type: ignore[misc]
            target_win_rates[(asset, sig)] = grp["won"].mean()
        full = compute_reprice_times(full, args.prices_dir, target_win_rates)

        full.to_csv(args.out_csv, index=False)

    report = build_report(full, args.sigma, unresolved)

    with open(args.out_report, "w") as f:
        f.write(report)
    log.info("Report → %s", args.out_report)

if __name__ == "__main__":
    main()
