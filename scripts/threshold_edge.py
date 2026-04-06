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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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

# Mean slippage observed from live trades (fill_price - trigger_pm_price).
# Added to pm_price so edge = win_rate - (pm_price + slippage) — reflects real cost.
# Update from scripts/slippage_report.py as more data accumulates.
SLIPPAGE: dict[str, float] = {
    "BTC":  0.0411,
    "DOGE": 0.0830,
    "ETH":  0.0536,
    "SOL":  0.0376,
}


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

    # sigma = std dev of window close-minus-open moves
    window_moves = []
    for wts in windows:
        mask = (coin_series.index >= wts) & (coin_series.index < wts + WINDOW_SECS)
        prices = coin_series[mask]
        if len(prices) < 280:
            continue
        window_moves.append(float(prices.iloc[-1]) - float(prices.iloc[0]))

    if len(window_moves) < 10:
        log.warning("%s: only %d windows — skipping", asset, len(window_moves))
        return pd.DataFrame(), 0

    sigma = float(np.std(window_moves))
    log.info("%s: sigma=%.8g  windows=%d", asset, sigma, len(window_moves))

    records = []
    n_unresolved = 0

    for wts in windows:
        mask = (coin_series.index >= wts) & (coin_series.index < wts + WINDOW_SECS)
        prices = coin_series[mask]
        if len(prices) < 280:
            continue

        open_price   = float(prices.iloc[0])
        window_move  = float(prices.iloc[-1]) - open_price
        hour_utc = datetime.fromtimestamp(wts, tz=timezone.utc).hour

        pm_window_df = asset_pm[asset_pm["window_ts"] == wts]
        if len(pm_window_df) < 280:
            continue
        resolved_up = _resolve_window(pm_window_df)
        if resolved_up is None:
            # PM price didn't reach 0.95/0.05 — fall back to coin price direction
            if window_move > 0:
                resolved_up = True
            elif window_move < 0:
                resolved_up = False
            else:
                n_unresolved += 1
                continue  # truly flat window, skip

        pm_window_idx = pm_window_df.set_index("ts").sort_index()

        for sig in sigma_multiples:
            up_thresh   = open_price + sig * sigma
            down_thresh = open_price - sig * sigma

            up_trig = down_trig = None
            for ts, price in prices.items():
                if up_trig is None and price >= up_thresh:
                    up_trig = int(ts)
                if down_trig is None and price <= down_thresh:
                    down_trig = int(ts)
                if up_trig and down_trig:
                    break

            # pick whichever crossed first
            if up_trig is None and down_trig is None:
                continue
            if up_trig is not None and (down_trig is None or up_trig <= down_trig):
                trigger_ts  = up_trig
                trigger_dir = "up"
                price_col   = "up_price"
                won         = bool(resolved_up)
            else:
                trigger_ts  = down_trig
                trigger_dir = "down"
                price_col   = "down_price"
                won         = not bool(resolved_up)

            pm_before = pm_window_idx[pm_window_idx.index <= trigger_ts]
            if pm_before.empty:
                pm_before = pm_window_idx
            if pm_before.empty or price_col not in pm_before.columns:
                continue

            pm_row = pm_before.iloc[-1]
            if pd.isna(pm_row.get(price_col)):
                continue
            pm_price = float(pm_row[price_col])

            # velocity & acceleration (sigma units)
            assert trigger_ts is not None
            # fetch prices at t, t-2, t-4, t-5, t-10
            _px: dict[int, float | None] = {}
            for offset in (0, 2, 4, 5, 10):
                ts_lookup = trigger_ts - offset
                candidates = coin_series.index[coin_series.index <= ts_lookup]
                _px[offset] = float(coin_series[candidates[-1]]) if len(candidates) > 0 else None

            def _diff(a: int, b: int) -> float | None:
                pa, pb = _px[a], _px[b]
                return (pa - pb) / sigma if pa is not None and pb is not None else None

            # velocity: price change from t-N to t, in sigma units
            # acceleration: change in velocity over equal halves (positive = speeding up in trigger dir)
            p0, p2, p4, p5, p10 = _px[0], _px[2], _px[4], _px[5], _px[10]
            v2  = _diff(0, 2)
            v10 = _diff(0, 10)

            # velocity_ratio: abs(v2s) / abs(v10s) — >1 means speeding up into trigger
            vel_ratio: float | None = None
            if v2 is not None and v10 is not None and v10 != 0.0:
                vel_ratio = abs(v2) / abs(v10)

            # vel_decay: abs(v10s) - abs(v2s) — positive means decelerating into trigger
            vel_decay: float | None = None
            if v2 is not None and v10 is not None:
                vel_decay = abs(v10) - abs(v2)

            # acc_positive: % of per-second accelerations in trigger direction over last 10s
            acc_positive: float | None = None
            sign = 1.0 if trigger_dir == "up" else -1.0
            trig_slice = prices[
                (prices.index >= trigger_ts - 10) & (prices.index <= trigger_ts)
            ].sort_index()
            if len(trig_slice) >= 3:
                per_sec_vel = trig_slice.diff().dropna() * sign   # positive = moving in trigger dir
                per_sec_acc = per_sec_vel.diff().dropna()         # positive = speeding up
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
                "trigger_dir":    trigger_dir,
                "trigger_ts":     trigger_ts,
                "trigger_second": trigger_ts - wts,
                "pm_price":       pm_price,
                "resolved_up":    resolved_up,
                "won":            won,
                **vels,
            })

    return pd.DataFrame(records), n_unresolved


# ── EWMA sigma ─────────────────────────────────────────────────────────────────

EWMA_WARMUP = 20  # windows to skip before the EWMA estimate is considered reliable


def compute_ewma_sigmas(
    windows_sorted: list[int],
    coin_series: pd.Series,
    lambda_: float,
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
    ewma_var: float | None = None
    result: dict[int, float] = {}

    for wts in windows_sorted:
        mask = (coin_series.index >= wts) & (coin_series.index < wts + WINDOW_SECS)
        prices = coin_series[mask]
        if len(prices) < 280:
            continue
        move = float(prices.iloc[-1]) - float(prices.iloc[0])

        # Sigma available to TRADE this window = what was known before it started
        if ewma_var is not None:
            result[wts] = float(np.sqrt(ewma_var))

        # Update EWMA with this window's realised move
        if ewma_var is None:
            ewma_var = move ** 2          # seed with first observed move²
        else:
            ewma_var = lambda_ * ewma_var + (1.0 - lambda_) * move ** 2

    return result


def analyze_asset_ewma(
    asset: str,
    pm_df: pd.DataFrame,
    coin_series: pd.Series,
    sigma_multiples: list[float],
    lambda_: float,
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

    ewma_sigmas = compute_ewma_sigmas(windows, coin_series, lambda_)

    records = []
    n_unresolved = 0

    for i, wts in enumerate(windows):
        ewma_sigma = ewma_sigmas.get(wts)
        if ewma_sigma is None or ewma_sigma <= 0:
            continue
        if i < EWMA_WARMUP:
            continue

        mask = (coin_series.index >= wts) & (coin_series.index < wts + WINDOW_SECS)
        prices = coin_series[mask]
        if len(prices) < 280:
            continue

        open_price  = float(prices.iloc[0])
        window_move = float(prices.iloc[-1]) - open_price
        hour_utc = datetime.fromtimestamp(wts, tz=timezone.utc).hour

        pm_window_df = asset_pm[asset_pm["window_ts"] == wts]
        if len(pm_window_df) < 280:
            continue
        resolved_up = _resolve_window(pm_window_df)
        if resolved_up is None:
            # PM price didn't reach 0.95/0.05 — fall back to coin price direction
            if window_move > 0:
                resolved_up = True
            elif window_move < 0:
                resolved_up = False
            else:
                n_unresolved += 1
                continue  # truly flat window, skip

        pm_window_idx = pm_window_df.set_index("ts").sort_index()

        for sig in sigma_multiples:
            up_thresh   = open_price + sig * ewma_sigma
            down_thresh = open_price - sig * ewma_sigma

            up_trig = down_trig = None
            for ts, price in prices.items():
                if up_trig is None and price >= up_thresh:
                    up_trig = int(ts)
                if down_trig is None and price <= down_thresh:
                    down_trig = int(ts)
                if up_trig and down_trig:
                    break

            if up_trig is None and down_trig is None:
                continue
            if up_trig is not None and (down_trig is None or up_trig <= down_trig):
                trigger_ts  = up_trig
                trigger_dir = "up"
                price_col   = "up_price"
                won         = bool(resolved_up)
            else:
                trigger_ts  = down_trig
                trigger_dir = "down"
                price_col   = "down_price"
                won         = not bool(resolved_up)

            pm_before = pm_window_idx[pm_window_idx.index <= trigger_ts]
            if pm_before.empty:
                pm_before = pm_window_idx
            if pm_before.empty or price_col not in pm_before.columns:
                continue

            pm_row = pm_before.iloc[-1]
            if pd.isna(pm_row.get(price_col)):
                continue
            pm_price = float(pm_row[price_col])

            records.append({
                "asset":          asset,
                "window_ts":      wts,
                "hour_utc":       hour_utc,
                "window_move":    window_move,
                "sigma":          sig,
                "sigma_abs":      sig * ewma_sigma,
                "ewma_sigma":     ewma_sigma,
                "lambda":         lambda_,
                "trigger_dir":    trigger_dir,
                "trigger_ts":     trigger_ts,
                "trigger_second": trigger_ts - wts,
                "pm_price":       pm_price,
                "resolved_up":    resolved_up,
                "won":            won,
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

            price_col = "up_price" if row["trigger_dir"] == "up" else "down_price"
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
        edge = win - avg_pm
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
    for rank, (eps, asset, sig, fill, edge_pct, eps_pct) in enumerate(ranking_rows, 1):
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


def section_trigger_timing(df: pd.DataFrame) -> str:
    """
    For each asset: per sigma level, show edge across three timing buckets.
    """
    BUCKETS = [
        ("early", "0–60s",    lambda s: s < 60),
        ("mid",   "60–180s",  lambda s: (s >= 60) & (s < 180)),
        ("late",  "180–300s", lambda s: s >= 180),
    ]

    out = []
    df = df.copy()

    for asset in sorted(df["asset"].unique()):
        adf = df[df["asset"] == asset]
        out.append(f"### {asset}\n")
        out.append("| sigma | early 0–60s | mid 60–180s | late 180–300s |")
        out.append("|---|---|---|---|")

        for sig, _ in adf.groupby("sigma"):
            cells = []
            edges = []
            for _, label, mask_fn in BUCKETS:
                grp = adf[(adf["sigma"] == sig) & mask_fn(adf["trigger_second"])]
                if grp.empty:
                    cells.append("—")
                    edges.append(None)
                else:
                    win  = grp["won"].mean()
                    edge = win - grp["pm_price"].mean()
                    edges.append(edge)
                    cells.append(f"win={win*100:.0f}% edge={edge:+.3f} (n={len(grp)})")

            best_idx = max(
                (i for i, e in enumerate(edges) if e is not None),
                key=lambda i: edges[i],
                default=None,
            )
            if best_idx is not None:
                cells[best_idx] = f"**{cells[best_idx]}**"

            out.append(f"| {sig}σ | {cells[0]} | {cells[1]} | {cells[2]} |")

        out.append("")

    return "\n".join(out)


def section_velocity_cutoff(df: pd.DataFrame) -> str:
    """
    Per asset/sigma/velocity column: split trades into velocity quartiles and show
    win rate per bucket. Reveals whether high-velocity trades underperform.
    Also shows the overall win rate so each quartile can be compared to baseline.
    """
    VEL_COLS  = ["vel_2s", "vel_5s", "vel_10s"]
    QUARTILES = [0.0, 0.25, 0.50, 0.75, 1.0]

    out = []
    for asset in sorted(df["asset"].unique()):
        adf = df[df["asset"] == asset].copy()
        out.append(f"### {asset}\n")

        for sig, sgrp in adf.groupby("sigma"):
            baseline_wr = sgrp["won"].mean()
            out.append(f"**{sig}σ** — baseline win rate: {baseline_wr*100:.0f}%\n")
            out.append("| velocity | Q1 slowest | Q2 | Q3 | Q4 fastest |")
            out.append("|---|---|---|---|---|")

            for col in VEL_COLS:
                valid = sgrp[sgrp[col].notna()].copy()
                if valid.empty:
                    out.append(f"| {col} | — | — | — | — |")
                    continue

                valid["_abs"] = valid[col].abs()
                boundaries = valid["_abs"].quantile(QUARTILES).tolist()

                cells = []
                for i in range(4):
                    lo, hi = boundaries[i], boundaries[i + 1]
                    if i == 3:
                        bucket = valid[valid["_abs"] >= lo]
                    else:
                        bucket = valid[(valid["_abs"] >= lo) & (valid["_abs"] < hi)]
                    if bucket.empty:
                        cells.append("—")
                        continue
                    wr   = bucket["won"].mean()
                    diff = wr - baseline_wr
                    sign = "+" if diff >= 0 else ""
                    # bold if meaningfully worse than baseline (>3% below)
                    cell = f"{wr*100:.0f}% ({sign}{diff*100:.0f}%) n={len(bucket)}"
                    if diff < -0.03:
                        cell = f"**{cell}**"
                    cells.append(cell)

                out.append(f"| {col} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")

            out.append("")

        out.append("")

    out.append(
        "_Each bucket is a velocity quartile. Numbers show win% (delta vs baseline) and n. "
        "**Bold** = >3% below baseline win rate — consider filtering these trades._"
    )
    return "\n".join(out)


def _utc_to_cst(utc_hour: int) -> int:
    """Convert UTC hour to CST hour (UTC-6)."""
    return (utc_hour - 6) % 24


def _cst_range_label(utc_start: int, utc_end: int) -> str:
    """Return 'HH–HH CST' label for a UTC hour range."""
    cst_start = _utc_to_cst(utc_start)
    cst_end   = _utc_to_cst(utc_end % 24)
    return f"{cst_start:02d}–{cst_end:02d} CST"


def section_hour_of_day(df: pd.DataFrame) -> str:
    """
    Edge by UTC hour — session buckets, ASCII bar chart, top/bottom hours.
    CST equivalents shown alongside UTC (CST = UTC-6).
    """
    SESSION_BUCKETS = [
        ("00–04 UTC", _cst_range_label(0,  4),  range(0, 4)),
        ("04–08 UTC", _cst_range_label(4,  8),  range(4, 8)),
        ("08–12 UTC", _cst_range_label(8,  12), range(8, 12)),
        ("12–16 UTC", _cst_range_label(12, 16), range(12, 16)),
        ("16–20 UTC", _cst_range_label(16, 20), range(16, 20)),
        ("20–24 UTC", _cst_range_label(20, 24), range(20, 24)),
    ]

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
        agg_hour_edges[hour] = float((grp["won"] - grp["pm_price"]).mean())

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


def section_price_filter(df: pd.DataFrame) -> str:
    """
    Simulates the user's actual strategy: only enter when pm_price < asset win rate.
    For each (asset, sigma): splits triggers into taken vs skipped and compares edge.
    """
    total_windows = df.groupby("asset")["window_ts"].nunique()
    out = []

    for asset, adf in df.groupby("asset"):
        out.append(f"### {asset}\n")
        n_windows = int(total_windows.get(asset, 1))

        out.append("| sigma | win_rate | n_triggers | n_taken | fill% | taken_win% | taken_edge | edge/session | n_skipped | skipped_edge |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

        for sig, grp in adf.groupby("sigma"):
            win_rate = grp["won"].mean()
            taken    = grp[grp["pm_price"] < win_rate]
            skipped  = grp[grp["pm_price"] >= win_rate]

            n_taken  = len(taken)
            fill_pct = n_taken / n_windows * 100

            if n_taken > 0:
                t_win  = taken["won"].mean()
                t_fill = taken["pm_price"].mean()
                t_edge = t_win - t_fill
                t_eps  = t_edge * (n_taken / n_windows)
                taken_str = f"{t_win*100:.1f}% | {t_edge:+.4f} | {t_eps:+.4f}"
            else:
                taken_str = "— | — | —"

            if len(skipped) > 0:
                s_win  = skipped["won"].mean()
                s_fill = skipped["pm_price"].mean()
                s_edge = s_win - s_fill
                skipped_str = f"{s_edge:+.4f}"
            else:
                skipped_str = "—"

            out.append(
                f"| {sig} | {win_rate:.3f} | {len(grp)} | {n_taken} | {fill_pct:.0f}% |"
                f" {taken_str} | {len(skipped)} | {skipped_str} |"
            )

        out.append("")

    out.append(
        "_win_rate = observed win rate for that asset/sigma (your hurdle rate). "
        "taken = trades where pm_price < win_rate (you entered). "
        "skipped = trades where pm_price ≥ win_rate (correctly avoided — negative edge). "
        "edge/session = taken_edge × fill% — expected value per session._"
    )
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


def section_reprice(df: pd.DataFrame) -> str:
    """How long (seconds) after trigger until PM price catches up."""
    valid = df[df["seconds_to_reprice"].notna()].copy()
    if valid.empty:
        return "_No reprice data available (need multiple PM snapshots per window)._"
    rows = []
    for key, grp in valid.groupby(["asset", "sigma"]):
        asset, sig = key  # type: ignore[misc]
        rows.append({
            "asset": asset, "sigma": sig,
            "n_repriced":      len(grp),
            "median_secs":     round(grp["seconds_to_reprice"].median(), 1),
            "p25_secs":        round(grp["seconds_to_reprice"].quantile(0.25), 1),
            "p75_secs":        round(grp["seconds_to_reprice"].quantile(0.75), 1),
            "never_repriced%": round(
                df[(df["asset"] == asset) & (df["sigma"] == sig)]
                ["seconds_to_reprice"].isna().mean() * 100, 1),
        })
    tbl = pd.DataFrame(rows).sort_values(["asset", "sigma"])
    return tbl.to_markdown(index=False)


def section_velocity(df: pd.DataFrame) -> str:
    """
    Velocity (speed into trigger) and acceleration (speeding up vs slowing down), split by outcome.
    Velocity is shown as absolute magnitude; acceleration is signed (+= speeding up toward threshold).
    """
    vel_cols = ["vel_2s", "vel_5s", "vel_10s"]
    acc_cols = ["acc_4s", "acc_10s"]
    valid = df.dropna(subset=vel_cols, how="all").copy()
    if valid.empty:
        return "_No velocity data available._"

    def fmt(v: float | None, pct: bool = False) -> str:
        if v is None:
            return "—"
        v = 0.0 if v == 0.0 else v   # strip negative zero
        if pct:
            return f"{v*100:.0f}%"
        return f"{v:+.3f}"

    out = []
    for asset in sorted(valid["asset"].unique()):
        adf = valid[valid["asset"] == asset]
        out.append(f"### {asset}\n")

        header = "| σ | outcome | n | v2s | v5s | v10s | a4s | a10s | v_ratio | v_decay | acc_pos% |"
        sep    = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        out.append(header)
        out.append(sep)

        for sig, sgrp in adf.groupby("sigma"):
            for outcome, label in [(True, "won"), (False, "lost")]:
                grp = sgrp[sgrp["won"] == outcome]
                if grp.empty:
                    continue
                cells = [f"{sig}σ", label, str(len(grp))]
                for col in vel_cols:
                    vals = grp[col].dropna().abs()
                    cells.append(fmt(float(vals.mean()) if not vals.empty else None))
                for col in acc_cols:
                    vals = grp[col].dropna()
                    cells.append(fmt(float(vals.mean()) if not vals.empty else None))
                # vel_ratio: unitless ratio, no sign
                vr = grp["vel_ratio"].dropna()
                cells.append(f"{vr.mean():.2f}" if not vr.empty else "—")
                # vel_decay: signed sigma units
                vd = grp["vel_decay"].dropna()
                cells.append(fmt(float(vd.mean()) if not vd.empty else None))
                # acc_positive: percentage
                ap = grp["acc_positive"].dropna()
                cells.append(fmt(float(ap.mean()) if not ap.empty else None, pct=True))
                out.append("| " + " | ".join(cells) + " |")

        out.append("")

    out.append(
        "_v2s/v5s/v10s = avg |price change over 2/5/10s| ÷ σ. "
        "a4s/a10s = acceleration (2nd derivative, + = speeding up). "
        "v_ratio = v2s/v10s (>1 = accelerating into trigger). "
        "v_decay = v10s−v2s (+ = decelerating into trigger). "
        "acc_pos% = % of per-second accelerations in the trigger direction over last 10s._"
    )
    return "\n".join(out)


def section_half_day(df: pd.DataFrame) -> str:
    """
    Edge and win rate split by half-day (AM = 00–11 UTC, PM = 12–23 UTC),
    further broken down by sigma level. One table per asset.
    CST (UTC-6) date and half shown alongside UTC.
    """
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
        win  = hdf["won"].mean()
        pm   = hdf["pm_price"].mean()
        edge = win - pm
        half_day_stats.append((date, half, n, win, pm, edge))

    out = []

    # ASCII bar chart
    out.append("```")
    for date, half, n, _, _, edge in half_day_stats:
        bar_len = max(0, int(abs(edge) * 100))
        bar  = ("█" * bar_len) if edge >= 0 else ("░" * bar_len)
        sign = "+" if edge >= 0 else "-"
        flag = " ⚠" if edge < 0 else ""
        out.append(f"  {date} {half}  {sign}{abs(edge):.3f}  {bar}{flag}")
    out.append("```\n")

    # Table
    out.append("| date | half | n_windows | win% | avg_pm | edge |")
    out.append("|---|---|---:|---:|---:|---:|")
    for date, half, n, win, pm, edge in half_day_stats:
        flag = " ⚠" if edge < 0 else ""
        out.append(
            f"| {date} | {half} | {n} | {win*100:.0f}% | {pm:.3f} | {edge:+.3f}{flag} |"
        )

    return "\n".join(out)


def section_ewma_comparison(
    static_df: pd.DataFrame,
    ewma_dfs: dict[float, pd.DataFrame],
) -> str:
    """
    Side-by-side comparison of static sigma vs EWMA sigma (one or more λ values).

    For each asset × sigma-multiplier:
      static   — global std-dev computed over the full history
      EWMA λ   — walk-forward per-window threshold (first EWMA_WARMUP windows excluded)

    Three sub-tables:
      1. Full detail (fills, win%, avg_pm, edge, edge/session) per method
      2. Sigma magnitude: how much the adaptive sigma deviates from static
      3. Edge/session advantage: EWMA minus static (positive = EWMA better)
    """
    if not ewma_dfs:
        return "_No EWMA data available._"

    lambdas = sorted(ewma_dfs.keys())
    out = []

    # ── 1. Per-asset detail ─────────────────────────────────────────────────
    for asset in sorted(static_df["asset"].unique()):
        out.append(f"### {asset}\n")
        out.append("| sigma | method | n_windows | n_fills | fill% | win% | avg_pm | edge | edge/session |")
        out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")

        s_df    = static_df[static_df["asset"] == asset]
        s_total = s_df["window_ts"].nunique()

        ewma_totals = {
            lam: ewma_dfs[lam][ewma_dfs[lam]["asset"] == asset]["window_ts"].nunique()
            for lam in lambdas
        }

        for sig in sorted(s_df["sigma"].unique()):
            s_grp = s_df[s_df["sigma"] == sig]
            if not s_grp.empty:
                win    = s_grp["won"].mean()
                avg_pm = s_grp["pm_price"].mean()
                edge   = win - avg_pm
                fills  = s_grp["window_ts"].nunique()
                fr     = fills / s_total if s_total else 0.0
                out.append(
                    f"| {sig}σ | static | {s_total} | {fills} | {fr*100:.1f}% | "
                    f"{win*100:.1f}% | {avg_pm:.4f} | {edge:+.4f} | {edge*fr:+.4f} |"
                )

            for lam in lambdas:
                e_df    = ewma_dfs[lam][ewma_dfs[lam]["asset"] == asset]
                e_total = ewma_totals[lam]
                if e_total == 0:
                    continue
                e_grp = e_df[e_df["sigma"] == sig]
                if e_grp.empty:
                    out.append(
                        f"| {sig}σ | EWMA λ={lam} | {e_total} | 0 | 0.0% | — | — | — | — |"
                    )
                    continue
                win    = e_grp["won"].mean()
                avg_pm = e_grp["pm_price"].mean()
                edge   = win - avg_pm
                fills  = e_grp["window_ts"].nunique()
                fr     = fills / e_total
                out.append(
                    f"| {sig}σ | EWMA λ={lam} | {e_total} | {fills} | {fr*100:.1f}% | "
                    f"{win*100:.1f}% | {avg_pm:.4f} | {edge:+.4f} | {edge*fr:+.4f} |"
                )

        out.append("")

    # ── 2. Sigma magnitude ──────────────────────────────────────────────────
    out.append("### Sigma magnitude: static vs EWMA\n")
    hdr = (
        ["asset", "static σ"]
        + [f"EWMA λ={lam} mean" for lam in lambdas]
        + [f"EWMA λ={lam} std"  for lam in lambdas]
    )
    out.append("| " + " | ".join(hdr) + " |")
    out.append("|" + "|".join(["---"] * len(hdr)) + "|")

    for asset in sorted(static_df["asset"].unique()):
        s_df = static_df[static_df["asset"] == asset]
        if s_df.empty:
            continue
        ref          = s_df[s_df["sigma"] > 0].iloc[0]
        static_sigma = ref["sigma_abs"] / ref["sigma"]

        ewma_means, ewma_stds = [], []
        for lam in lambdas:
            e_df = ewma_dfs[lam][ewma_dfs[lam]["asset"] == asset]
            if "ewma_sigma" in e_df.columns and not e_df.empty:
                vals = e_df.drop_duplicates("window_ts")["ewma_sigma"]
                ewma_means.append(f"{vals.mean():.4g}")
                ewma_stds.append(f"{vals.std():.4g}")
            else:
                ewma_means.append("—")
                ewma_stds.append("—")

        out.append(
            f"| {asset} | {static_sigma:.4g} | "
            + " | ".join(ewma_means) + " | "
            + " | ".join(ewma_stds) + " |"
        )

    out.append("")

    # ── 3. Edge/session advantage ───────────────────────────────────────────
    out.append("### Edge/session advantage: EWMA − static (positive = EWMA better)\n")
    hdr2 = ["asset", "sigma"] + [f"EWMA λ={lam}" for lam in lambdas]
    out.append("| " + " | ".join(hdr2) + " |")
    out.append("|" + "|".join(["---"] * len(hdr2)) + "|")

    for asset in sorted(static_df["asset"].unique()):
        s_df    = static_df[static_df["asset"] == asset]
        s_total = s_df["window_ts"].nunique()

        for sig in sorted(s_df["sigma"].unique()):
            s_grp = s_df[s_df["sigma"] == sig]
            if s_grp.empty:
                continue
            s_edge = s_grp["won"].mean() - s_grp["pm_price"].mean()
            s_eps  = s_edge * (s_grp["window_ts"].nunique() / s_total)

            adv_cells = []
            for lam in lambdas:
                e_df    = ewma_dfs[lam][ewma_dfs[lam]["asset"] == asset]
                e_total = e_df["window_ts"].nunique()
                e_grp   = e_df[e_df["sigma"] == sig]
                if e_grp.empty or e_total == 0:
                    adv_cells.append("—")
                    continue
                e_edge = e_grp["won"].mean() - e_grp["pm_price"].mean()
                e_eps  = e_edge * (e_grp["window_ts"].nunique() / e_total)
                diff   = e_eps - s_eps
                adv_cells.append(f"{diff:+.4f}")

            out.append(f"| {asset} | {sig}σ | " + " | ".join(adv_cells) + " |")

    out.append("")
    out.append(
        f"_Static sigma: single value over entire history. "
        f"EWMA: walk-forward, updated each window, no lookahead. "
        f"First {EWMA_WARMUP} windows excluded from EWMA results (warmup). "
        f"Edge advantage = EWMA edge/session − static edge/session on their respective sample sets._"
    )
    return "\n".join(out)


# ── report builder ────────────────────────────────────────────────────────────

def build_report(
    df: pd.DataFrame,
    sigma_levels: list[float],
    unresolved: dict[str, int] | None = None,
    ewma_dfs: dict[float, pd.DataFrame] | None = None,
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
        "(σ = std dev of all 5-min window moves for that asset).",
        "We take the **first** crossing in either direction and bet accordingly.",
        "**Edge** = actual win rate − (avg Polymarket price + slippage) at trigger moment (filled sessions only).",
        "**Edge/session** = edge × fill_rate — expected value per session regardless of whether a trigger fires.",
        f"**Slippage applied** (from live trade data): {', '.join(f'{a}={v:+.4f}' for a, v in SLIPPAGE.items())}",
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
        "## Price Filter — Edge When Trading Below Win Rate",
        "",
        "Your actual strategy: only enter when `pm_price < win_rate` for that asset/sigma.",
        "Trades above the win rate hurdle are skipped — shown here to confirm they have negative edge.",
        "",
        section_price_filter(df),
        "",
        "---",
        "",
        "## Trigger Timing — Does It Matter When in the Window the Coin Moves?",
        "",
        "Early triggers (0–60s) leave more time for the market to catch up.",
        "Late triggers (180–300s) give less time but may have higher certainty.",
        "",
        section_trigger_timing(df),
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
        "## Cascade Rate — If Coin Crosses 0.5σ, How Often Does It Reach Higher Sigmas?",
        "",
        "Shows momentum continuation. High cascade% at lower base sigmas means",
        "a small initial move is a reliable precursor to a larger one.",
        "",
        section_cascade(df, sigma_levels),
        "",
        "---",
        "",
        "## Time to Reprice — How Long Does the Edge Window Last?",
        "",
        "`never_repriced%` = windows where PM price never reached the actual win rate",
        "before the window closed — the edge persisted all the way to resolution.",
        "",
        section_reprice(df),
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
        "## EWMA vs Static Sigma — Adaptive Volatility Comparison",
        "",
        "Static sigma is a single value computed over the entire history. "
        "EWMA (Exponentially Weighted Moving Average) sigma adapts after each session, "
        "giving more weight to recent moves (λ closer to 1 = slower decay = longer memory). "
        "The EWMA sigma used for window *t* is estimated from windows 0…t−1 only "
        "(strict walk-forward — no lookahead). "
        f"First {EWMA_WARMUP} windows are excluded while the estimate warms up.",
        "",
        section_ewma_comparison(df, ewma_dfs or {}),
        "",
        "---",
        "",
        "## Recent Backtest — 4h / 12h / 24h",
        "",
        "Recommended strategy (best sigma per asset) tested on the most recent windows.",
        "**A) Always enter** = buy every trigger. **B) Price filter** = only buy when `pm_price < win_rate`.",
        "Edge/session = edge × fill% — expected value per window whether or not a trigger fires.",
        "",
        section_recent_backtest(df),
        "",
        "---",
        "",
        "## Key Takeaways",
        "",
        _takeaways(df, sigma_levels),
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
    last 4h / 12h / 24h of data.

    Two strategies compared per horizon:
      A) Always enter  — every trigger that fires at the recommended sigma
      B) Price filter  — only enter when pm_price < win_rate (full-dataset win rate)

    Reports: fills, win%, avg_price, edge, edge/session.
    """
    HORIZONS = [("4h", 48), ("12h", 144), ("24h", 288)]

    total_windows = df.groupby("asset")["window_ts"].nunique()
    out = [
        "_Win rate (hurdle) = observed win rate over the full dataset at recommended sigma. "
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
            edge   = win - avg_pm
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
        out.append("| horizon | strategy | fills (fill%) | win% | avg_price | edge | edge/session |")
        out.append("|---|---|---|---:|---:|---:|---:|")

        for label, n_windows in HORIZONS:
            recent_windows = set(all_windows[-n_windows:])
            recent = full_grp[full_grp["window_ts"].isin(recent_windows)]
            n_recent = len(recent_windows)

            def _stats(sub: pd.DataFrame, n: int = n_recent) -> str:
                if sub.empty:
                    return "— | — | — | — | —"
                n_fills  = len(sub)
                fill_pct = n_fills / n * 100
                w   = sub["won"].mean()
                p   = sub["pm_price"].mean()
                e   = w - p
                eps = e * (n_fills / n)
                return f"{n_fills} ({fill_pct:.0f}%) | {w*100:.1f}% | {p:.3f} | {e:+.4f} | {eps:+.4f}"

            strat_a = recent
            strat_b = recent[recent["pm_price"] < win_rate]
            out.append(f"| {label} | A) always enter | {_stats(strat_a)} |")
            out.append(f"| {label} | B) price < win_rate | {_stats(strat_b)} |")

        out.append("")

    return "\n".join(out)


def _config_yaml(df: pd.DataFrame) -> str:
    """
    For each asset, pick the sigma with the best edge_per_session and emit YAML config.
    """
    total_windows = df.groupby("asset")["window_ts"].nunique()
    blocks = []

    for asset in sorted(df["asset"].unique()):
        adf = df[df["asset"] == asset]

        ref = adf[adf["sigma"] > 0].iloc[0]
        sigma_value = ref["sigma_abs"] / ref["sigma"]

        n_total = total_windows.get(asset, 1)

        best_sig  = None
        best_eps  = -999.0
        best_win  = 0.0
        best_edge = 0.0

        for sig, sgrp in adf.groupby("sigma"):
            win       = sgrp["won"].mean()
            avg_pm    = sgrp["pm_price"].mean()
            edge      = win - avg_pm
            fill_rate = sgrp["window_ts"].nunique() / n_total
            eps       = edge * fill_rate
            if eps > best_eps:
                best_eps  = eps
                best_sig  = sig
                best_win  = win
                best_edge = edge

        if best_sig is None:
            continue

        blocks.append(
            f"{asset}:  # best entry: {best_sig}σ  |  edge/fill: {best_edge*100:+.1f}%  |  edge/session: {best_eps*100:+.2f}%\n"
            f"  sigma_value: {sigma_value:.8g}\n"
            f"  sigma_entry: {best_sig}\n"
            f"  max_pm_price: {best_win:.2f}\n"
            f"  wallet_pct: 0.05\n"
            f"  name: mom_{asset.strip().lower()}"
        )

    return "```yaml\n" + "\n\n".join(blocks) + "\n```"


def _takeaways(df: pd.DataFrame, sigma_levels: list[float]) -> str:
    lines = []

    # best edge overall
    best_rows = []
    for key, grp in df.groupby(["asset", "sigma"]):
        asset, sig = key  # type: ignore[misc]
        win  = grp["won"].mean()
        edge = win - grp["pm_price"].mean()
        best_rows.append((edge, asset, sig, len(grp)))
    best_rows.sort(reverse=True)
    e, a, s, n = best_rows[0]
    lines.append(f"- **Best edge**: {a} at {s}σ — {round(e*100,1)}% edge over {n} windows")

    # asset with most consistent edge
    asset_edges = {}
    for key, grp in df.groupby(["asset", "sigma"]):
        asset, sig = key  # type: ignore[misc]
        win  = grp["won"].mean()
        edge = win - grp["pm_price"].mean()
        asset_edges.setdefault(asset, []).append(edge)
    best_asset = max(asset_edges, key=lambda a: np.mean(asset_edges[a]))
    lines.append(
        f"- **Most consistently mispriced asset**: {best_asset} "
        f"(avg edge {round(np.mean(asset_edges[best_asset])*100,1)}% across all sigma levels)"
    )

    # early vs late trigger at base sigma
    base_sig = sigma_levels[0]
    sub   = df[df["sigma"] == base_sig]
    early = sub[sub["trigger_second"] < 60]
    late  = sub[sub["trigger_second"] >= 180]
    if not early.empty and not late.empty:
        early_edge = float((early["won"] - early["pm_price"]).mean())
        late_edge  = float((late["won"]  - late["pm_price"]).mean())
        better = "early" if early_edge > late_edge else "late"
        lines.append(
            f"- **Triggers at {base_sig}σ**: {better} triggers have more edge "
            f"(early={round(early_edge*100,1)}%, late={round(late_edge*100,1)}%)"
        )

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--assets",     nargs="+", default=list(ASSET_TO_SYMBOL.keys()))
    p.add_argument("--prices-dir", default="data/prices")
    p.add_argument("--coin-dir",   default="data/coin_prices")
    p.add_argument("--sigma",      nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--out-csv",    default="data/reports/threshold_edge.csv")
    p.add_argument("--out-report", default="data/reports/threshold_edge.md")
    p.add_argument(
        "--lambda", dest="lambda_vals", nargs="+", type=float, default=[0.95, 0.97],
        metavar="LAM",
        help="EWMA decay factors to evaluate (default: 0.95 0.97)",
    )
    p.add_argument(
        "--no-ewma", dest="ewma", action="store_false", default=True,
        help="Skip EWMA sigma analysis (faster, omits section 8)",
    )
    p.add_argument(
        "--from-csv", action="store_true", default=False,
        help="Skip data loading/analysis and regenerate the report from the existing --out-csv file",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    ewma_dfs: dict[float, pd.DataFrame] = {}

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
        ewma_records: dict[float, list[pd.DataFrame]] = {lam: [] for lam in args.lambda_vals}

        for asset in args.assets:
            log.info("Loading data for %s…", asset)
            pm_df = load_prices_for_asset(args.prices_dir, asset)
            if pm_df.empty:
                log.warning("%s: no PM price data — skipping", asset)
                continue

            coin = load_coin_prices(args.coin_dir, asset)
            if coin is None:
                continue

            recs, n_unresolved = analyze_asset(asset, pm_df, coin, args.sigma)
            unresolved[asset] = n_unresolved
            if not recs.empty:
                all_records.append(recs)

            if args.ewma:
                for lam in args.lambda_vals:
                    lam_recs, _ = analyze_asset_ewma(asset, pm_df, coin, args.sigma, lam)
                    if not lam_recs.empty:
                        ewma_records[lam].append(lam_recs)

            del coin, pm_df  # free memory before loading next asset

        if not all_records:
            log.error("No data.")
            sys.exit(1)

        full = pd.concat(all_records, ignore_index=True)

        # Adjust pm_price by observed slippage so edge = win_rate - (pm_price + slippage).
        # Assets not in SLIPPAGE are unchanged (slippage assumed 0).
        full["pm_price"] = full["pm_price"] + full["asset"].map(SLIPPAGE).fillna(0.0)
        log.info("Applied slippage adjustments: %s", SLIPPAGE)

        # compute target win rates then add reprice times
        target_win_rates = {}
        for key, grp in full.groupby(["asset", "sigma"]):
            asset, sig = key  # type: ignore[misc]
            target_win_rates[(asset, sig)] = grp["won"].mean()
        full = compute_reprice_times(full, args.prices_dir, target_win_rates)

        full.to_csv(args.out_csv, index=False)
        log.info("Raw records → %s", args.out_csv)

        if args.ewma:
            for lam in args.lambda_vals:
                ewma_dfs[lam] = (
                    pd.concat(ewma_records[lam], ignore_index=True) if ewma_records[lam] else pd.DataFrame()
                )
                log.info("EWMA λ=%.2f: %d records", lam, len(ewma_dfs[lam]))
        else:
            log.info("EWMA skipped (--no-ewma)")

    report = build_report(full, args.sigma, unresolved, ewma_dfs)

    with open(args.out_report, "w") as f:
        f.write(report)
    log.info("Report → %s", args.out_report)

if __name__ == "__main__":
    main()
