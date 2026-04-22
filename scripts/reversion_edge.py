#!/usr/bin/env python3
"""
Failed-Breakout Mean-Reversion Edge Report.

For each 5-minute Polymarket window, using EWMA walk-forward sigma:
  1. Find the FIRST second the coin crosses window_open ± N×sigma (initial trigger).
  2. Observe the next WAIT_SECS seconds of coin price. If price does NOT reach the
     next sigma level in the same direction, classify as a "failed breakout".
  3. Enter the OPPOSITE direction at PM ask price at (trigger_ts + WAIT_SECS + 1).
  4. Measure edge: actual win rate vs effective entry cost (ask + 1.5% fee).

Sigma trigger levels: 0.25σ, 0.5σ, 1.0σ
Continuation threshold: 0.25→0.5, 0.5→1.0, 1.0→1.5

Report sections:
  1. Summary — edge by asset / sigma trigger level
  2. Failure rate — % of crossings that become failed breakouts
  3. Reversion depth — where is the coin at T+30 vs the trigger threshold?
  4. Stagnation type — stall (above threshold) vs reversal (below threshold)
  5. Hour of day — when are failed breakouts most profitable?
  6. Entry timing sensitivity — 15s / 30s / 45s wait
  7. Recent backtest

Usage:
    python scripts/reversion_edge.py
    python scripts/reversion_edge.py --assets BTC ETH --sigma 0.25 0.5 1.0
    python scripts/reversion_edge.py --wait-secs 30
    python scripts/reversion_edge.py --from-csv
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

# Reuse shared data-loading / EWMA utilities from threshold_edge
from scripts.threshold_edge import (
    ASSET_TO_SYMBOL,
    BUY_FEE_RATE,
    EWMA_LAMBDA,
    EWMA_REFRESH_SECS,
    EWMA_WARMUP,
    WINDOW_SECS,
    _price_with_fee,
    _resolve_window,
    compute_ewma_sigmas,
    compute_rolling_vol_ratios,
    load_coin_prices,
    load_prices_for_asset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%M:%S")
log = logging.getLogger(__name__)

DEFAULT_WAIT_SECS = 30

# For each trigger sigma level, the minimum continuation level that would
# invalidate the "failed breakout" classification.
NEXT_SIGMA_MAP: dict[float, float] = {
    0.25: 0.5,
    0.5:  1.0,
    1.0:  1.5,
}

SENSITIVITY_WAITS = [15, 30, 45]


def _edge_from_win_and_pm(win_rate: float, avg_pm_price: float) -> float:
    return float(win_rate - _price_with_fee(avg_pm_price))


def _price_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where pm_price × (1 + fee) < per-(asset, sigma) win rate."""
    win_rates = (
        df.groupby(["asset", "sigma"])["won"]
        .mean()
        .rename("_win_rate")
        .reset_index()
    )
    merged = df.merge(win_rates, on=["asset", "sigma"], how="left")
    merged.index = df.index
    return df[_price_with_fee(merged["pm_price"]) < merged["_win_rate"]]


# ── core analysis ─────────────────────────────────────────────────────────────

def analyze_asset_reversion(
    asset: str,
    pm_df: pd.DataFrame,
    coin_series: pd.Series,
    sigma_multiples: list[float],
    wait_secs: int = DEFAULT_WAIT_SECS,
    lambda_: float = EWMA_LAMBDA,
    refresh_secs: int = EWMA_REFRESH_SECS,
) -> tuple[pd.DataFrame, int]:
    """
    Walk-forward failed-breakout analysis for one asset.

    For each window:
      - Find the first coin price crossing of ±N×ewma_sigma.
      - Over the following `wait_secs` seconds, check whether price ever reaches
        the next continuation level (NEXT_SIGMA_MAP[N]).
      - If NOT → failed breakout. Record an entry in the OPPOSITE direction at
        the PM ask price at (trigger_ts + wait_secs + 1).

    Returns (records_df, n_unresolved).
    """
    asset_pm = pm_df[pm_df["asset"] == asset].copy()
    windows = sorted(asset_pm["window_ts"].unique())

    ewma_sigmas = compute_ewma_sigmas(windows, coin_series, lambda_, refresh_secs=refresh_secs)

    ewma_vals = [v for v in ewma_sigmas.values() if v > 0]
    baseline_sigma = float(np.mean(ewma_vals)) if ewma_vals else 1.0
    vol_ratios = compute_rolling_vol_ratios(windows, coin_series, baseline_sigma)

    ts_idx = coin_series.index.values
    vals   = coin_series.values

    pm_by_window: dict[int, pd.DataFrame] = {
        int(wts): grp.set_index("ts").sort_index()
        for wts, grp in asset_pm.groupby("window_ts")
    }

    records: list[dict] = []
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
        hour_utc    = datetime.fromtimestamp(wts, tz=timezone.utc).hour

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
                n_unresolved += 1
                continue

        pm_ts_arr = pm_window_idx.index.values

        for sig in sigma_multiples:
            next_sig = NEXT_SIGMA_MAP.get(sig)
            if next_sig is None:
                continue  # no continuation level defined

            up_thresh   = open_price + sig * ewma_sigma
            down_thresh = open_price - sig * ewma_sigma

            up_hits = np.where(win_pr_arr >= up_thresh)[0]
            dn_hits = np.where(win_pr_arr <= down_thresh)[0]
            up_trig = int(win_ts_arr[up_hits[0]]) if len(up_hits) else None
            dn_trig = int(win_ts_arr[dn_hits[0]]) if len(dn_hits) else None

            if up_trig is None and dn_trig is None:
                continue

            # Pick the first trigger
            if up_trig is not None and (dn_trig is None or up_trig <= dn_trig):
                trigger_ts  = up_trig
                trigger_dir = "up"
            else:
                trigger_ts  = dn_trig
                trigger_dir = "down"

            if trigger_ts is None:
                continue

            trigger_second = trigger_ts - wts

            # We need at least wait_secs seconds remaining before the window ends
            if trigger_second + wait_secs >= WINDOW_SECS - 1:
                continue

            # ── continuation check ────────────────────────────────────────────
            # Inspect coin price in [trigger_ts, trigger_ts + wait_secs]
            obs_lo = int(np.searchsorted(win_ts_arr, trigger_ts,             side="left"))
            obs_hi = int(np.searchsorted(win_ts_arr, trigger_ts + wait_secs, side="right"))
            obs_pr = win_pr_arr[obs_lo:obs_hi]

            if len(obs_pr) == 0:
                continue

            next_up_thresh   = open_price + next_sig * ewma_sigma
            next_down_thresh = open_price - next_sig * ewma_sigma

            if trigger_dir == "up":
                continued = bool(np.any(obs_pr >= next_up_thresh))
            else:
                continued = bool(np.any(obs_pr <= next_down_thresh))

            if continued:
                continue  # breakout succeeded → not a reversion signal

            # ── failed breakout confirmed ─────────────────────────────────────
            # Entry: opposite direction, PM ask at trigger_ts + wait_secs + 1
            entry_ts = trigger_ts + wait_secs
            # Price of coin at entry moment (for depth/stagnation stats)
            ep_idx = int(np.searchsorted(ts_idx, entry_ts, side="right")) - 1
            price_at_entry = float(vals[ep_idx]) if ep_idx >= 0 else float("nan")

            # How far did price move during the wait window? (in sigma units)
            if trigger_dir == "up":
                max_excursion = (float(np.max(obs_pr)) - (open_price + sig * ewma_sigma)) / ewma_sigma
                entry_depth   = (price_at_entry - (open_price + sig * ewma_sigma)) / ewma_sigma
                # stall: still above threshold; reversal: back below threshold
                stagnation_type = "reversal" if price_at_entry < up_thresh else "stall"
            else:
                max_excursion = ((open_price - sig * ewma_sigma) - float(np.min(obs_pr))) / ewma_sigma
                entry_depth   = ((open_price - sig * ewma_sigma) - price_at_entry) / ewma_sigma
                stagnation_type = "reversal" if price_at_entry > down_thresh else "stall"

            # Reversion direction for PM entry
            if trigger_dir == "up":
                rev_price_col = "dn_ask"   # we bet DOWN
                won           = not bool(resolved_up)
            else:
                rev_price_col = "up_ask"   # we bet UP
                won           = bool(resolved_up)

            # PM price lookup at entry_ts + 1
            fill_ts  = entry_ts + 1
            fill_pos = int(np.searchsorted(pm_ts_arr, fill_ts, side="left"))
            if fill_pos >= len(pm_ts_arr):
                continue
            if rev_price_col not in pm_window_idx.columns:
                continue
            pm_row = pm_window_idx.iloc[fill_pos]
            if pd.isna(pm_row.get(rev_price_col)):
                continue
            pm_price = float(pm_row[rev_price_col])

            # PM price at original trigger (for reference / comparison)
            orig_price_col   = "up_ask" if trigger_dir == "up" else "dn_ask"
            trig_fill_pos    = int(np.searchsorted(pm_ts_arr, trigger_ts + 1, side="left"))
            pm_price_at_trig: float | None = None
            if trig_fill_pos < len(pm_ts_arr) and orig_price_col in pm_window_idx.columns:
                v = pm_window_idx.iloc[trig_fill_pos].get(orig_price_col)
                if not pd.isna(v):
                    pm_price_at_trig = float(v)

            records.append({
                "asset":            asset,
                "window_ts":        wts,
                "hour_utc":         hour_utc,
                "window_move":      window_move,
                "sigma":            sig,
                "ewma_sigma":       ewma_sigma,
                "vol_ratio":        vol_ratios.get(wts),
                "trigger_dir":      trigger_dir,
                "trigger_ts":       trigger_ts,
                "trigger_second":   trigger_second,
                "stagnation_type":  stagnation_type,
                "max_excursion":    max_excursion,   # max overshoot past threshold in sigma
                "entry_depth":      entry_depth,     # price@T+30 relative to threshold in sigma
                "pm_price_at_trig": pm_price_at_trig,
                "pm_price":         pm_price,        # reversion entry price
                "resolved_up":      resolved_up,
                "won":              won,
                "wait_secs":        wait_secs,
            })

    return pd.DataFrame(records), n_unresolved


def analyze_sensitivity(
    asset: str,
    pm_df: pd.DataFrame,
    coin_series: pd.Series,
    sigma_multiples: list[float],
    wait_secs_list: list[int],
    lambda_: float = EWMA_LAMBDA,
    refresh_secs: int = EWMA_REFRESH_SECS,
) -> pd.DataFrame:
    """Run analyze_asset_reversion for multiple wait_secs values and combine."""
    frames = []
    for w in wait_secs_list:
        df, _ = analyze_asset_reversion(asset, pm_df, coin_series, sigma_multiples, w, lambda_, refresh_secs)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── report sections ───────────────────────────────────────────────────────────

def section_summary(df: pd.DataFrame) -> str:
    """Edge by asset and sigma trigger level."""
    total_windows = df.groupby("asset")["window_ts"].nunique()

    rows = []
    for (asset, sig), grp in df.groupby(["asset", "sigma"]):
        win      = grp["won"].mean()
        avg_pm   = grp["pm_price"].mean()
        edge     = _edge_from_win_and_pm(win, avg_pm)
        n_total  = int(total_windows.get(asset, len(grp)))
        n_fills  = grp["window_ts"].nunique()
        fill_rate = n_fills / n_total
        rows.append({
            "asset":        asset,
            "sigma_trigger": sig,
            "n_fills":      n_fills,
            "fill_rate%":   round(fill_rate * 100, 1),
            "win%":         round(win * 100, 1),
            "avg_pm":       round(avg_pm, 4),
            "edge":         round(edge, 4),
            "edge/session": round(edge * fill_rate, 4),
        })
    tbl = pd.DataFrame(rows)
    if tbl.empty:
        return "_No reversion signals found._"

    out = []
    for asset, adf in tbl.sort_values(["asset", "sigma_trigger"]).groupby("asset", sort=False):
        out.append(f"### {asset}\n")
        out.append(adf.drop(columns="asset").to_markdown(index=False))
        out.append("")
    return "\n".join(out)


def section_failure_rate(df_main: pd.DataFrame, df_all_triggers: pd.DataFrame | None) -> str:
    """
    Failure rate: of all sigma crossings, what % became failed breakouts?

    df_main = reversion records (failed breakouts only)
    df_all_triggers = if provided, the full trigger set from threshold_edge analysis
                      for comparison; otherwise we approximate from df_main windows.
    """
    out = []
    out.append(
        "_Failure rate = % of initial sigma crossings that did **not** continue to "
        "the next sigma level within 30 seconds._\n"
    )
    if df_all_triggers is None or df_all_triggers.empty:
        out.append(
            "_Full trigger baseline not provided — showing reversion signal count only._\n"
        )
        for (asset, sig), grp in df_main.groupby(["asset", "sigma"]):
            n = grp["window_ts"].nunique()
            out.append(f"  {asset} {sig}σ → {n} failed breakout windows")
        return "\n".join(out)

    out.append("| asset | sigma | total_crossings | failed | failure_rate% |")
    out.append("|---|---|---:|---:|---:|")
    for (asset, sig), rev_grp in df_main.groupby(["asset", "sigma"]):
        trig_grp = df_all_triggers[
            (df_all_triggers["asset"] == asset) & (df_all_triggers["sigma"] == sig)
        ]
        n_total  = trig_grp["window_ts"].nunique() if not trig_grp.empty else "—"
        n_failed = rev_grp["window_ts"].nunique()
        if isinstance(n_total, int) and n_total > 0:
            rate = f"{n_failed / n_total * 100:.0f}%"
        else:
            rate = "—"
        out.append(f"| {asset} | {sig}σ | {n_total} | {n_failed} | {rate} |")
    return "\n".join(out)


def section_stagnation_type(df: pd.DataFrame) -> str:
    """
    Break down failed breakouts by stagnation type:
      stall     — price still above/below trigger threshold at T+30 (momentum slowed)
      reversal  — price pulled back through the trigger threshold at T+30 (mean reversion)
    """
    out = []
    out.append("| asset | sigma | stagnation_type | n | win% | avg_pm | edge | edge/session |")
    out.append("|---|---|---|---:|---:|---:|---:|---:|")

    total_windows = df.groupby("asset")["window_ts"].nunique()
    for (asset, sig, stype), grp in df.groupby(["asset", "sigma", "stagnation_type"]):
        win      = grp["won"].mean()
        avg_pm   = grp["pm_price"].mean()
        edge     = _edge_from_win_and_pm(win, avg_pm)
        n_total  = int(total_windows.get(asset, 1))
        n_fills  = grp["window_ts"].nunique()
        eps      = edge * (n_fills / n_total)
        out.append(
            f"| {asset} | {sig}σ | {stype} | {n_fills} |"
            f" {win*100:.1f}% | {avg_pm:.4f} | {edge:+.4f} | {eps:+.4f} |"
        )

    out.append("")
    out.append(
        "_stall = price still on original side of threshold at T+30 (momentum faded). "
        "reversal = price has crossed back through the trigger threshold (mean reversion active)._"
    )
    return "\n".join(out)


def section_reversion_depth(df: pd.DataFrame) -> str:
    """
    Distribution of entry_depth at T+30 relative to the trigger threshold.
    entry_depth in sigma units: positive = still past threshold, negative = reversed past it.
    """
    out = []
    for (asset, sig), grp in df.groupby(["asset", "sigma"]):
        depth = grp["entry_depth"].dropna()
        if depth.empty:
            continue
        out.append(f"**{asset} {sig}σ** (n={len(depth)}):")
        out.append(f"  mean={depth.mean():+.3f}σ  median={depth.median():+.3f}σ  "
                   f"p25={depth.quantile(0.25):+.3f}σ  p75={depth.quantile(0.75):+.3f}σ  "
                   f"reversal%={((depth < 0).mean()*100):.0f}%")
        out.append("")
    out.append(
        "_entry_depth = (coin_price@T+30 minus trigger_threshold) ÷ ewma_sigma. "
        "Negative = price reversed back through the threshold. Positive = still on the far side._"
    )
    return "\n".join(out)


def _utc_to_cst(h: int) -> int:
    return (h - 6) % 24


def section_hour_of_day(df: pd.DataFrame) -> str:
    """Edge by UTC hour, aggregated across all assets and sigmas."""
    df = _price_filter(df)
    if df.empty:
        return "_No trades pass the price filter._"

    out = []
    agg: dict[int, float] = {}
    for hour in range(24):
        grp = df[df["hour_utc"] == hour]
        if grp.empty:
            continue
        agg[hour] = float((grp["won"] - _price_with_fee(grp["pm_price"])).mean())

    out.append("**Edge by hour (UTC)** — each █ ≈ 1% edge:\n")
    out.append("```")
    for hour in range(24):
        if hour not in agg:
            continue
        e = agg[hour]
        n = len(df[df["hour_utc"] == hour])
        bar_len = max(0, int(abs(e) * 100))
        bar  = ("█" * bar_len) if e >= 0 else ("░" * bar_len)
        sign = "+" if e >= 0 else "-"
        cst  = _utc_to_cst(hour)
        out.append(f"  {hour:02d}h UTC / {cst:02d}h CST  {sign}{abs(e):.3f}  {bar}  (n={n})")
    out.append("```")

    if agg:
        top3    = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:3]
        bottom3 = sorted(agg.items(), key=lambda x: x[1])[:3]
        out.append(
            "\n**Best hours:**  " +
            "  ".join(f"{h:02d}h UTC/{_utc_to_cst(h):02d}h CST ({e:+.3f})" for h, e in top3)
        )
        out.append(
            "**Worst hours:** " +
            "  ".join(f"{h:02d}h UTC/{_utc_to_cst(h):02d}h CST ({e:+.3f})" for h, e in bottom3)
        )
    return "\n".join(out)


def section_timing_sensitivity(df_all: pd.DataFrame) -> str:
    """
    Compare edge at different wait times (15s / 30s / 45s) for the same windows.
    df_all must contain records from all three wait_secs values.
    """
    out = []
    if df_all.empty or "wait_secs" not in df_all.columns:
        return "_No timing sensitivity data available._"

    out.append("| asset | sigma | wait_secs | n_fills | win% | avg_pm | edge | edge/session |")
    out.append("|---|---|---|---:|---:|---:|---:|---:|")

    total_windows = df_all.groupby("asset")["window_ts"].nunique()
    for (asset, sig, wait), grp in df_all.groupby(["asset", "sigma", "wait_secs"]):
        win    = grp["won"].mean()
        avg_pm = grp["pm_price"].mean()
        edge   = _edge_from_win_and_pm(win, avg_pm)
        n_fills = grp["window_ts"].nunique()
        n_total = int(total_windows.get(asset, 1))
        eps    = edge * (n_fills / n_total)
        out.append(
            f"| {asset} | {sig}σ | {int(wait)}s | {n_fills} |"
            f" {win*100:.1f}% | {avg_pm:.4f} | {edge:+.4f} | {eps:+.4f} |"
        )
    out.append("")
    out.append(
        "_wait_secs = how long after the initial threshold crossing before we enter. "
        "Shorter waits enter earlier but may catch momentum still in play. "
        "Longer waits enter later but fewer windows have enough remaining time._"
    )
    return "\n".join(out)


def section_recent_backtest(df: pd.DataFrame) -> str:
    """Edge on the most recent 4h / 12h / 24h of data (price-filtered)."""
    HORIZONS = [("4h", 48), ("12h", 144), ("24h", 288)]
    total_windows = df.groupby("asset")["window_ts"].nunique()
    out = [
        "_Price filter applied: only entries where `pm_price × 1.015 < win_rate`._\n",
    ]

    for asset in sorted(df["asset"].unique()):
        adf = df[df["asset"] == asset]
        n_total_all = int(total_windows.get(asset, 1))

        # pick best sigma by edge/session
        best_sig = None
        best_eps = -999.0
        for sig, grp in adf.groupby("sigma"):
            win  = grp["won"].mean()
            pm   = grp["pm_price"].mean()
            edge = _edge_from_win_and_pm(win, pm)
            eps  = edge * (grp["window_ts"].nunique() / n_total_all)
            if eps > best_eps:
                best_eps = eps
                best_sig = sig

        if best_sig is None:
            continue

        full_grp  = adf[adf["sigma"] == best_sig]
        win_rate  = full_grp["won"].mean()
        all_windows = sorted(adf["window_ts"].unique())

        out.append(f"### {asset}  (best trigger: {best_sig}σ | hurdle: {win_rate:.3f})\n")
        out.append("| horizon | fills (fill%) | win% | avg_price | edge | edge/session |")
        out.append("|---|---|---:|---:|---:|---:|")

        for label, n_win in HORIZONS:
            recent_windows = set(all_windows[-n_win:])
            recent = full_grp[full_grp["window_ts"].isin(recent_windows)]
            taken  = recent[_price_with_fee(recent["pm_price"]) < win_rate]
            n_recent = len(recent_windows)

            if taken.empty:
                out.append(f"| {label} | — | — | — | — | — |")
            else:
                n_fills  = len(taken)
                fill_pct = n_fills / n_recent * 100
                w  = taken["won"].mean()
                p  = taken["pm_price"].mean()
                e  = _edge_from_win_and_pm(w, p)
                eps = e * (n_fills / n_recent)
                out.append(
                    f"| {label} | {n_fills} ({fill_pct:.0f}%) | {w*100:.1f}% | {p:.3f} |"
                    f" {e:+.4f} | {eps:+.4f} |"
                )
        out.append("")

    return "\n".join(out)


# ── report builder ─────────────────────────────────────────────────────────────

def build_report(
    df: pd.DataFrame,
    df_sensitivity: pd.DataFrame,
    sigma_levels: list[float],
    wait_secs: int,
    unresolved: dict[str, int] | None = None,
    df_all_triggers: pd.DataFrame | None = None,
) -> str:
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

    next_levels = ", ".join(f"{s}→{NEXT_SIGMA_MAP[s]}" for s in sigma_levels if s in NEXT_SIGMA_MAP)

    lines = [
        "# Failed-Breakout Mean-Reversion Edge Report",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "---",
        "",
        "## Overview",
        "",
        "A **failed breakout** occurs when the coin price crosses a sigma threshold but does",
        f"**not** continue to the next sigma level within **{wait_secs} seconds**.",
        "When this happens we enter the **opposite** direction, betting that price will",
        "mean-revert back through the trigger threshold.",
        "",
        f"- **Trigger levels**: {', '.join(f'{s}σ' for s in sigma_levels)}",
        f"- **Continuation check**: {next_levels}  (price must NOT reach next level in {wait_secs}s)",
        f"- **Entry**: opposite-direction PM ask at `trigger_ts + {wait_secs} + 1s`",
        f"- **Sigma**: EWMA walk-forward (λ={EWMA_LAMBDA}), updated each window from prior history only",
        f"- **Fee**: {BUY_FEE_RATE*100:.1f}% buy fee applied to all edge calculations",
        f"- **Price filter**: only trade when `pm_price × {1+BUY_FEE_RATE} < win_rate`",
        "",
        "**Resolution**: last PM price ≥ 0.95 → UP won; ≤ 0.05 → DOWN won. "
        "Ambiguous windows fall back to coin price direction.",
        "",
        "**Unresolved windows** (coin exactly flat — excluded):",
        "",
        unresolved_table,
        "",
        "---",
        "",
        "## 1. Summary — Edge by Asset / Sigma Trigger Level",
        "",
        section_summary(df),
        "",
        "---",
        "",
        "## 2. Failure Rate — How Often Do Breakouts Fail?",
        "",
        section_failure_rate(df, df_all_triggers),
        "",
        "---",
        "",
        "## 3. Stagnation Type — Stall vs Reversal",
        "",
        "Does it matter whether the price just slowed down vs actively reversed?",
        "",
        section_stagnation_type(df),
        "",
        "---",
        "",
        "## 4. Reversion Depth — Where Is Price at T+30?",
        "",
        "Distribution of coin price at T+30, relative to the trigger threshold (in σ units).",
        "Negative = price has reversed back through the threshold. "
        "Positive = still on the breakout side.",
        "",
        section_reversion_depth(df),
        "",
        "---",
        "",
        "## 5. Hour of Day — When Are Failed Breakouts Most Profitable?",
        "",
        section_hour_of_day(df),
        "",
        "---",
        "",
        "## 6. Entry Timing Sensitivity — 15s / 30s / 45s Wait",
        "",
        "Same windows, same signals — does waiting longer or entering sooner help?",
        "",
        section_timing_sensitivity(df_sensitivity),
        "",
        "---",
        "",
        "## 7. Recent Backtest — 4h / 12h / 24h",
        "",
        section_recent_backtest(df),
    ]
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Failed-breakout mean-reversion edge report")
    p.add_argument("--assets",     nargs="+", default=list(ASSET_TO_SYMBOL.keys()))
    p.add_argument("--prices-dir", default="data/prices")
    p.add_argument("--coin-dir",   default="data/coin_prices")
    p.add_argument("--sigma",      nargs="+", type=float, default=[0.25, 0.5, 1.0])
    p.add_argument("--wait-secs",  type=int,  default=DEFAULT_WAIT_SECS,
                   help=f"Seconds to wait after trigger before checking for continuation (default: {DEFAULT_WAIT_SECS})")
    p.add_argument("--out-csv",    default="data/reports/reversion_edge.csv")
    p.add_argument("--out-report", default="data/reports/reversion_edge.md")
    p.add_argument(
        "--from-csv", action="store_true", default=False,
        help="Regenerate report from existing --out-csv (skip data reprocessing)",
    )
    p.add_argument(
        "--threshold-csv", default="data/reports/threshold_edge.csv",
        help="Path to threshold_edge CSV for failure-rate comparison",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    if args.from_csv:
        if not os.path.exists(args.out_csv):
            log.error("--from-csv specified but %s does not exist", args.out_csv)
            sys.exit(1)
        full = pd.read_csv(args.out_csv)
        log.info("Loaded %d reversion records from %s", len(full), args.out_csv)
        unresolved: dict[str, int] = {}
        df_sensitivity = pd.DataFrame()
    else:
        all_records     = []
        all_sensitivity = []
        unresolved      = {}

        for asset in args.assets:
            log.info("Processing %s…", asset)
            pm_df = load_prices_for_asset(args.prices_dir, asset)
            if pm_df.empty:
                log.warning("%s: no PM data — skipping", asset)
                continue

            coin = load_coin_prices(args.coin_dir, asset)
            if coin is None:
                continue

            recs, n_unres = analyze_asset_reversion(
                asset, pm_df, coin, args.sigma, args.wait_secs, EWMA_LAMBDA
            )
            unresolved[asset] = n_unres
            if not recs.empty:
                all_records.append(recs)
                log.info("%s: %d failed breakout events", asset, len(recs))

            # Sensitivity analysis (15s / 30s / 45s)
            sens = analyze_sensitivity(
                asset, pm_df, coin, args.sigma, SENSITIVITY_WAITS, EWMA_LAMBDA
            )
            if not sens.empty:
                all_sensitivity.append(sens)

            del coin, pm_df

        if not all_records:
            log.error("No reversion signals found across any asset.")
            sys.exit(1)

        full = pd.concat(all_records, ignore_index=True)
        full.to_csv(args.out_csv, index=False)
        log.info("Reversion records → %s  (%d rows)", args.out_csv, len(full))

        df_sensitivity = (
            pd.concat(all_sensitivity, ignore_index=True) if all_sensitivity else pd.DataFrame()
        )

    # Load threshold_edge CSV for failure-rate baseline (optional)
    df_all_triggers: pd.DataFrame | None = None
    if os.path.exists(args.threshold_csv):
        try:
            df_all_triggers = pd.read_csv(args.threshold_csv)
            log.info("Loaded %d threshold records from %s", len(df_all_triggers), args.threshold_csv)
        except Exception as e:
            log.warning("Could not load threshold CSV: %s", e)

    report = build_report(
        full,
        df_sensitivity,
        args.sigma,
        args.wait_secs,
        unresolved,
        df_all_triggers,
    )

    with open(args.out_report, "w") as f:
        f.write(report)
    log.info("Report → %s", args.out_report)


if __name__ == "__main__":
    main()
