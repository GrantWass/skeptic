#!/usr/bin/env python3
"""
Threshold edge analysis + report.

For each 5-minute Polymarket window:
  - Finds the first second the coin crosses window_open ± N*sigma
  - Records the up_price / down_price from Polymarket at that trigger moment
  - Determines resolution from last up_price in window (>= 0.9 → UP won)

Report sections:
  1. Summary — edge by asset / sigma / direction
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


# ── data loading ──────────────────────────────────────────────────────────────

def load_prices(prices_dir: str) -> pd.DataFrame:
    files = sorted(Path(prices_dir).glob("prices_*.csv"))
    if not files:
        raise FileNotFoundError(f"No prices_*.csv in {prices_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset=["ts", "window_ts", "asset"])
    df = df.sort_values(["asset", "window_ts", "ts"]).reset_index(drop=True)
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
    """Last up_price >= 0.9 → UP won, <= 0.1 → DOWN won. Mirrors fetcher.py."""
    if pm_window.empty or "up_price" not in pm_window.columns:
        return None
    last_up = pm_window.sort_values("ts")["up_price"].iloc[-1]
    if pd.isna(last_up):
        return None
    if last_up >= 0.9:
        return True
    if last_up <= 0.1:
        return False
    return None


# ── per-asset analysis ────────────────────────────────────────────────────────

def analyze_asset(
    asset: str,
    pm_df: pd.DataFrame,
    coin_series: pd.Series,
    sigma_multiples: list[float],
) -> pd.DataFrame:
    """
    Returns one record per (window, direction, sigma). Columns:
      asset, window_ts, hour_utc, sigma, direction,
      trigger_ts, trigger_second, pm_price, resolved_up,
      seconds_to_reprice  (how long until PM price >= actual win rate for that bucket)
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
        return pd.DataFrame()

    sigma = float(np.std(window_moves))
    log.info("%s: sigma=%.8g  windows=%d", asset, sigma, len(window_moves))

    records = []

    for wts in windows:
        mask = (coin_series.index >= wts) & (coin_series.index < wts + WINDOW_SECS)
        prices = coin_series[mask]
        if len(prices) < 280:
            continue

        open_price = float(prices.iloc[0])
        hour_utc = datetime.fromtimestamp(wts, tz=timezone.utc).hour

        pm_window_df = asset_pm[asset_pm["window_ts"] == wts]
        if len(pm_window_df) < 280:
            continue
        resolved_up = _resolve_window(pm_window_df)
        if resolved_up is None:
            continue

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

            for direction, trigger_ts, price_col in [
                ("up",   up_trig,   "up_price"),
                ("down", down_trig, "down_price"),
            ]:
                if trigger_ts is None:
                    continue

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
                    "sigma":          sig,
                    "sigma_abs":      sig * sigma,
                    "direction":      direction,
                    "trigger_ts":     trigger_ts,
                    "trigger_second": trigger_ts - wts,
                    "pm_price":       pm_price,
                    "resolved_up":    resolved_up,
                })

    return pd.DataFrame(records)


# ── time-to-reprice ───────────────────────────────────────────────────────────

def compute_reprice_times(
    records: pd.DataFrame,
    pm_df: pd.DataFrame,
    target_win_rates: dict,   # (asset, sigma, direction) → actual win rate
) -> pd.DataFrame:
    """
    For each record, find the first second after trigger where pm_price
    has crossed the actual win rate for that (asset, sigma, direction) bucket.
    Returns records with a 'seconds_to_reprice' column added.
    """
    out = []
    pm_indexed = pm_df.set_index(["asset", "window_ts", "ts"]).sort_index()

    for _, row in records.iterrows():
        key = (row["asset"], row["sigma"], row["direction"])
        target = target_win_rates.get(key)
        if target is None:
            out.append(None)
            continue

        price_col = "up_price" if row["direction"] == "up" else "down_price"
        try:
            pm_window = pm_df[
                (pm_df["asset"] == row["asset"]) &
                (pm_df["window_ts"] == row["window_ts"]) &
                (pm_df["ts"] > row["trigger_ts"])
            ].sort_values("ts")
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
            out.append(int(crossed.iloc[0]["ts"]) - int(row["trigger_ts"]))

    records = records.copy()
    records["seconds_to_reprice"] = out
    return records


# ── report sections ───────────────────────────────────────────────────────────

def _win(grp: pd.DataFrame, direction: str) -> float:
    if direction == "up":
        return grp["resolved_up"].mean()
    return (~grp["resolved_up"]).mean()


def section_summary(df: pd.DataFrame) -> str:
    total_windows = df.groupby("asset")["window_ts"].nunique()

    rows = []
    for (asset, sig, direction), grp in df.groupby(["asset", "sigma", "direction"]):
        win = _win(grp, direction)
        avg_pm = grp["pm_price"].mean()
        edge = win - avg_pm
        n_total = total_windows.get(asset, len(grp))
        n_fills = grp["window_ts"].nunique()
        fill_rate = n_fills / n_total
        rows.append({
            "asset":            asset,
            "sigma":            sig,
            "direction":        direction,
            "n_fills":          n_fills,
            "fill_rate%":       round(fill_rate * 100, 1),
            "win%":             round(win * 100, 1),
            "avg_pm":           round(avg_pm, 4),
            "edge":             round(edge, 4),
            "edge_per_session": round(edge * fill_rate, 4),
        })
    tbl = pd.DataFrame(rows)

    # ── Asset ranking (one line per asset, best sigma entry) ──────────────────
    ranking_lines = ["**Asset ranking by edge/session** (best sigma entry, avg up+down):\n"]
    ranking_rows = []
    for asset, adf in tbl.groupby("asset"):
        best = adf.loc[adf["edge_per_session"].idxmax()]
        avg_eps = adf.groupby("sigma")["edge_per_session"].mean().max()
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

    return "\n".join(ranking_lines) + "\n" + tbl.sort_values(["asset", "direction", "sigma"]).to_markdown(index=False)


def section_trigger_timing(df: pd.DataFrame) -> str:
    """
    For each asset: per sigma level, show edge across three timing buckets
    as a compact side-by-side comparison. Highlights the best bucket.
    """
    BUCKETS = [
        ("early", "0–60s",   lambda s: s < 60),
        ("mid",   "60–180s", lambda s: (s >= 60) & (s < 180)),
        ("late",  "180–300s",lambda s: s >= 180),
    ]

    out = []
    df = df.copy()

    for asset in sorted(df["asset"].unique()):
        adf = df[df["asset"] == asset]
        out.append(f"### {asset}\n")
        out.append("| sigma | direction | early 0–60s | mid 60–180s | late 180–300s |")
        out.append("|---|---|---|---|---|")

        for (sig, direction), _ in adf.groupby(["sigma", "direction"]):
            cells = []
            edges = []
            for _, label, mask_fn in BUCKETS:
                grp = adf[(adf["sigma"] == sig) & (adf["direction"] == direction) &
                          mask_fn(adf["trigger_second"])]
                if grp.empty:
                    cells.append("—")
                    edges.append(None)
                else:
                    win = _win(grp, direction)
                    edge = win - grp["pm_price"].mean()
                    edges.append(edge)
                    cells.append(f"win={win*100:.0f}% edge={edge:+.3f} (n={len(grp)})")

            # bold the best bucket
            best_idx = max(
                (i for i, e in enumerate(edges) if e is not None),
                key=lambda i: edges[i],
                default=None,
            )
            if best_idx is not None:
                cells[best_idx] = f"**{cells[best_idx]}**"

            out.append(f"| {sig}σ | {direction} | {cells[0]} | {cells[1]} | {cells[2]} |")

        out.append("")

    return "\n".join(out)


def section_hour_of_day(df: pd.DataFrame) -> str:
    """
    Edge by UTC hour, displayed as:
      - Trading session buckets (Asia / Europe / US)
      - Per-asset ASCII bar chart of avg edge by hour
      - Top 3 / bottom 3 hours per asset
    """
    SESSION_BUCKETS = [
        ("Asia       (00–08 UTC)", range(0, 8)),
        ("Europe     (08–14 UTC)", range(8, 14)),
        ("US         (14–22 UTC)", range(14, 22)),
        ("US Late    (22–24 UTC)", range(22, 24)),
    ]

    out = []

    for asset in sorted(df["asset"].unique()):
        adf = df[df["asset"] == asset]
        out.append(f"### {asset}")

        # --- session bucket summary ---
        out.append("\n**By trading session** (avg edge across all sigma/direction):\n")
        out.append("| Session | n | avg edge |")
        out.append("|---|---|---|")
        for label, hours in SESSION_BUCKETS:
            grp = adf[adf["hour_utc"].isin(hours)]
            if grp.empty:
                continue
            edges = []
            for direction, dgrp in grp.groupby("direction"):
                win = _win(dgrp, direction)
                edges.append(win - dgrp["pm_price"].mean())
            avg_edge = float(np.mean(edges)) if edges else 0.0
            out.append(f"| {label} | {len(grp)} | {avg_edge:+.4f} |")

        # --- ASCII bar chart ---
        out.append("\n**Edge by hour (UTC)** — each █ ≈ 1% edge:\n")
        out.append("```")
        hour_edges = {}
        for hour in range(24):
            grp = adf[adf["hour_utc"] == hour]
            if grp.empty:
                continue
            edges = []
            for direction, dgrp in grp.groupby("direction"):
                win = _win(dgrp, direction)
                edges.append(win - dgrp["pm_price"].mean())
            hour_edges[hour] = float(np.mean(edges)) if edges else 0.0

        for hour in range(24):
            if hour not in hour_edges:
                continue
            e = hour_edges[hour]
            bar_len = max(0, int(abs(e) * 100))
            bar = ("█" * bar_len) if e >= 0 else ("░" * bar_len)
            sign = "+" if e >= 0 else "-"
            out.append(f"  {hour:02d}h  {sign}{abs(e):.3f}  {bar}")
        out.append("```")

        # --- top/bottom hours ---
        if hour_edges:
            sorted_hours = sorted(hour_edges.items(), key=lambda x: x[1], reverse=True)
            top3    = sorted_hours[:3]
            bottom3 = sorted_hours[-3:]
            top_str    = "  ".join(f"{h:02d}h ({e:+.3f})" for h, e in top3)
            bottom_str = "  ".join(f"{h:02d}h ({e:+.3f})" for h, e in bottom3)
            out.append(f"\n**Best hours:**  {top_str}")
            out.append(f"**Worst hours:** {bottom_str}\n")

    return "\n".join(out)


def section_cascade(df: pd.DataFrame, sigma_levels: list[float]) -> str:
    """
    ASCII flow chain showing momentum continuation per asset/direction.
    Format: 0.5σ [fill%] --cascade%--> 1.0σ [fill%] --cascade%--> ...
    """
    total_windows = df.groupby("asset")["window_ts"].nunique()
    out = ["```"]

    for asset in sorted(df["asset"].unique()):
        adf = df[df["asset"] == asset]
        n_total = total_windows.get(asset, 1)

        for direction in ["up", "down"]:
            ddf = adf[adf["direction"] == direction]
            parts = []
            for i, sig in enumerate(sigma_levels):
                windows_at = set(ddf[ddf["sigma"] == sig]["window_ts"])
                fill_pct = len(windows_at) / n_total * 100

                if i == 0:
                    parts.append(f"{sig}σ [{fill_pct:.0f}%]")
                else:
                    prev_windows = set(ddf[ddf["sigma"] == sigma_levels[i-1]]["window_ts"])
                    cascade = len(prev_windows & windows_at) / len(prev_windows) * 100 if prev_windows else 0
                    parts.append(f"--{cascade:.0f}%--> {sig}σ [{fill_pct:.0f}%]")

            label = f"{asset:4s} {direction:4s}:  "
            out.append(label + "  ".join(parts))

        out.append("")

    out.append("```")
    out.append("\n_Each node shows fill rate (% of all sessions). Arrows show what % of triggering sessions continue to the next level._")
    return "\n".join(out)



def section_reprice(df: pd.DataFrame) -> str:
    """How long (seconds) after trigger until PM price catches up."""
    valid = df[df["seconds_to_reprice"].notna()].copy()
    if valid.empty:
        return "_No reprice data available (need multiple PM snapshots per window)._"
    rows = []
    for (asset, sig, direction), grp in valid.groupby(["asset", "sigma", "direction"]):
        rows.append({
            "asset": asset, "sigma": sig, "direction": direction,
            "n_repriced": len(grp),
            "median_secs": round(grp["seconds_to_reprice"].median(), 1),
            "p25_secs":    round(grp["seconds_to_reprice"].quantile(0.25), 1),
            "p75_secs":    round(grp["seconds_to_reprice"].quantile(0.75), 1),
            "never_repriced%": round(
                df[(df["asset"]==asset)&(df["sigma"]==sig)&(df["direction"]==direction)]
                ["seconds_to_reprice"].isna().mean() * 100, 1),
        })
    tbl = pd.DataFrame(rows).sort_values(["asset", "direction", "sigma"])
    return tbl.to_markdown(index=False)


# ── report builder ────────────────────────────────────────────────────────────

def build_report(df: pd.DataFrame, sigma_levels: list[float]) -> str:
    # compute actual win rates per bucket for reprice analysis
    target_win_rates = {}
    for (asset, sig, direction), grp in df.groupby(["asset", "sigma", "direction"]):
        target_win_rates[(asset, sig, direction)] = _win(grp, direction)

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
        "**Edge** = actual win rate − avg Polymarket price at trigger moment (filled sessions only).",
        "**Edge/session** = edge × fill_rate — expected value per session regardless of whether a trigger fires.",
        "",
        "---",
        "",
        "## 1. Summary — Edge by Asset / Sigma / Direction",
        "",
        section_summary(df),
        "",
        "---",
        "",
        "## 2. Trigger Timing — Does It Matter When in the Window the Coin Moves?",
        "",
        "Early triggers (0–60s) leave more time for the market to catch up.",
        "Late triggers (180–300s) give less time but may have higher certainty.",
        "",
        section_trigger_timing(df),
        "",
        "---",
        "",
        "## 3. Hour of Day (UTC) — When Is the Edge Largest?",
        "",
        "Thinner hours may have slower Polymarket repricing → more edge.",
        "",
        section_hour_of_day(df),
        "",
        "---",
        "",
        "## 4. Cascade Rate — If Coin Crosses 0.5σ, How Often Does It Reach Higher Sigmas?",
        "",
        "Shows momentum continuation. High cascade% at lower base sigmas means",
        "a small initial move is a reliable precursor to a larger one.",
        "",
        section_cascade(df, sigma_levels),
        "",
        "---",
        "",
        "## 5. Time to Reprice — How Long Does the Edge Window Last?",
        "",
        "`never_repriced%` = windows where PM price never reached the actual win rate",
        "before the window closed — the edge persisted all the way to resolution.",
        "",
        section_reprice(df),
        "",
        "---",
        "",
        "## Key Takeaways",
        "",
        _takeaways(df, sigma_levels),
        "",
        "---",
        "",
        "## Strategy Commands",
        "",
        "Best entry per asset based on highest `edge_per_session`.",
        "Remove `--dry-run` to trade live.",
        "",
        _strategy_commands(df),
    ]
    return "\n".join(lines)


def _strategy_commands(df: pd.DataFrame) -> str:
    """
    For each asset, pick the sigma with the best edge_per_session (averaged across
    up/down directions). Use win% at that sigma as max_pm_price.
    Sigma value is recovered from sigma_abs / sigma column.
    """
    total_windows = df.groupby("asset")["window_ts"].nunique()
    lines = []

    for asset in sorted(df["asset"].unique()):
        adf = df[df["asset"] == asset]

        # recover sigma_value from sigma_abs / sigma (both are stored per record)
        ref = adf[adf["sigma"] > 0].iloc[0]
        sigma_value = ref["sigma_abs"] / ref["sigma"]

        n_total = total_windows.get(asset, 1)

        best_sig = None
        best_eps = -999.0
        best_win = 0.0
        best_edge = 0.0

        for sig, sgrp in adf.groupby("sigma"):
            eps_list, win_list, edge_list = [], [], []
            for direction, dgrp in sgrp.groupby("direction"):
                win = _win(dgrp, direction)
                avg_pm = dgrp["pm_price"].mean()
                edge = win - avg_pm
                fill_rate = dgrp["window_ts"].nunique() / n_total
                eps_list.append(edge * fill_rate)
                win_list.append(win)
                edge_list.append(edge)
            avg_eps = float(np.mean(eps_list))
            if avg_eps > best_eps:
                best_eps  = avg_eps
                best_sig  = sig
                best_win  = float(np.mean(win_list))
                best_edge = float(np.mean(edge_list))

        if best_sig is None:
            continue

        lines.append(
            f"```bash\n"
            f"# {asset}  |  best entry: {best_sig}σ  |  edge/fill: {best_edge*100:+.1f}%  |  edge/session: {best_eps*100:+.2f}%\n"
            f"python scripts/live_momentum_buy.py \\\n"
            f"  --asset {asset} \\\n"
            f"  --sigma-value {sigma_value:.8g} \\\n"
            f"  --sigma-entry {best_sig} \\\n"
            f"  --max-pm-price {best_win:.2f} \\\n"
            f"  --dry-run\n"
            f"```\n"
        )

    return "\n".join(lines)


def _takeaways(df: pd.DataFrame, sigma_levels: list[float]) -> str:
    lines = []

    # best edge overall
    best_rows = []
    for (asset, sig, direction), grp in df.groupby(["asset", "sigma", "direction"]):
        win = _win(grp, direction)
        edge = win - grp["pm_price"].mean()
        best_rows.append((edge, asset, sig, direction, len(grp)))
    best_rows.sort(reverse=True)
    e, a, s, d, n = best_rows[0]
    lines.append(f"- **Best edge**: {a} {d} at {s}σ — {round(e*100,1)}% edge over {n} windows")

    # asset with most consistent edge
    asset_edges = {}
    for (asset, sig, direction), grp in df.groupby(["asset", "sigma", "direction"]):
        win = _win(grp, direction)
        edge = win - grp["pm_price"].mean()
        asset_edges.setdefault(asset, []).append(edge)
    best_asset = max(asset_edges, key=lambda a: np.mean(asset_edges[a]))
    lines.append(
        f"- **Most consistently mispriced asset**: {best_asset} "
        f"(avg edge {round(np.mean(asset_edges[best_asset])*100,1)}% across all sigma/direction combos)"
    )

    # early vs late trigger comparison at 0.5 sigma
    base_sig = sigma_levels[0]
    for direction in ["up", "down"]:
        sub = df[(df["sigma"] == base_sig) & (df["direction"] == direction)].copy()
        if sub.empty:
            continue
        early = sub[sub["trigger_second"] < 60]
        late  = sub[sub["trigger_second"] >= 180]
        if early.empty or late.empty:
            continue
        early_edge = _win(early, direction) - early["pm_price"].mean()
        late_edge  = _win(late, direction)  - late["pm_price"].mean()
        better = "early" if early_edge > late_edge else "late"
        lines.append(
            f"- **{direction.upper()} triggers at {base_sig}σ**: {better} triggers have more edge "
            f"(early={round(early_edge*100,1)}%, late={round(late_edge*100,1)}%)"
        )

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--assets", nargs="+", default=list(ASSET_TO_SYMBOL.keys()))
    p.add_argument("--prices-dir", default="data/prices")
    p.add_argument("--coin-dir",   default="data/coin_prices")
    p.add_argument("--sigma", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    p.add_argument("--out-csv",    default="data/reports/threshold_edge.csv")
    p.add_argument("--out-report", default="data/reports/threshold_edge.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    pm_df = load_prices(args.prices_dir)

    all_records = []
    for asset in args.assets:
        coin = load_coin_prices(args.coin_dir, asset)
        if coin is None:
            continue
        recs = analyze_asset(asset, pm_df, coin, args.sigma)
        if not recs.empty:
            all_records.append(recs)

    if not all_records:
        log.error("No data.")
        sys.exit(1)

    full = pd.concat(all_records, ignore_index=True)

    # compute target win rates then add reprice times
    target_win_rates = {}
    for (asset, sig, direction), grp in full.groupby(["asset", "sigma", "direction"]):
        target_win_rates[(asset, sig, direction)] = _win(grp, direction)
    full = compute_reprice_times(full, pm_df, target_win_rates)

    full.to_csv(args.out_csv, index=False)
    log.info("Raw records → %s", args.out_csv)

    report = build_report(full, args.sigma)

    with open(args.out_report, "w") as f:
        f.write(report)
    log.info("Report → %s", args.out_report)

if __name__ == "__main__":
    main()
