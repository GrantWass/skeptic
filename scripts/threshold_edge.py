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
            n_unresolved += 1
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


# ── time-to-reprice ───────────────────────────────────────────────────────────

def compute_reprice_times(
    records: pd.DataFrame,
    pm_df: pd.DataFrame,
    target_win_rates: dict,   # (asset, sigma) → actual win rate
) -> pd.DataFrame:
    """
    For each record, find the first second after trigger where pm_price
    has crossed the actual win rate for that (asset, sigma) bucket.
    Returns records with a 'seconds_to_reprice' column added.
    """
    out = []

    for _, row in records.iterrows():
        key = (row["asset"], row["sigma"])
        target = target_win_rates.get(key)
        if target is None:
            out.append(None)
            continue

        price_col = "up_price" if row["trigger_dir"] == "up" else "down_price"
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


def section_hour_of_day(df: pd.DataFrame) -> str:
    """
    Edge by UTC hour — session buckets, ASCII bar chart, top/bottom hours.
    """
    SESSION_BUCKETS = [
        ("00–04 UTC", range(0, 4)),
        ("04–08 UTC", range(4, 8)),
        ("08–12 UTC", range(8, 12)),
        ("12–16 UTC", range(12, 16)),
        ("16–20 UTC", range(16, 20)),
        ("20–24 UTC", range(20, 24)),
    ]

    out = []

    for asset in sorted(df["asset"].unique()):
        adf = df[df["asset"] == asset]
        out.append(f"### {asset}")

        # --- session bucket summary ---
        out.append("\n**By trading session** (avg edge across all sigma):\n")
        out.append("| Session | n | avg edge |")
        out.append("|---|---|---|")
        for label, hours in SESSION_BUCKETS:
            grp = adf[adf["hour_utc"].isin(hours)]
            if grp.empty:
                continue
            avg_edge = float((grp["won"] - grp["pm_price"]).mean())
            out.append(f"| {label} | {len(grp)} | {avg_edge:+.4f} |")

        # --- ASCII bar chart ---
        out.append("\n**Edge by hour (UTC)** — each █ ≈ 1% edge:\n")
        out.append("```")
        hour_edges = {}
        for hour in range(24):
            grp = adf[adf["hour_utc"] == hour]
            if grp.empty:
                continue
            hour_edges[hour] = float((grp["won"] - grp["pm_price"]).mean())

        for hour in range(24):
            if hour not in hour_edges:
                continue
            e = hour_edges[hour]
            bar_len = max(0, int(abs(e) * 100))
            bar  = ("█" * bar_len) if e >= 0 else ("░" * bar_len)
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
    """
    df = df.copy()
    df["date"]     = df["window_ts"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).date())
    df["half"]     = df["hour_utc"].apply(lambda h: "AM" if h < 12 else "PM")
    df["half_day"] = df["date"].astype(str) + " " + df["half"]

    out = []
    for asset in sorted(df["asset"].unique()):
        adf = df[df["asset"] == asset]

        out.append(f"### {asset}\n")
        out.append("| date | half | σ (std dev) | n | win% | avg_pm | edge |")
        out.append("|---|---|---:|---:|---:|---:|---:|")

        for half_day in sorted(adf["half_day"].unique()):
            hdf  = adf[adf["half_day"] == half_day]
            date, half = half_day.rsplit(" ", 1)
            # one move per window — deduplicate across sigma levels
            moves = hdf.drop_duplicates("window_ts")["window_move"]
            sigma = float(moves.std()) if len(moves) > 1 else float("nan")
            win  = hdf["won"].mean()
            pm   = hdf["pm_price"].mean()
            edge = win - pm
            flag = " ⚠" if edge < 0 else ""
            sigma_str = f"{sigma:.6g}" if not np.isnan(sigma) else "—"
            out.append(f"| {date} | {half} | {sigma_str} | {len(moves)} | {win*100:.0f}% | {pm:.3f} | {edge:+.3f}{flag} |")

        out.append("")

    return "\n".join(out)


# ── report builder ────────────────────────────────────────────────────────────

def build_report(df: pd.DataFrame, sigma_levels: list[float], unresolved: dict[str, int] | None = None) -> str:
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
        "**Edge** = actual win rate − avg Polymarket price at trigger moment (filled sessions only).",
        "**Edge/session** = edge × fill_rate — expected value per session regardless of whether a trigger fires.",
        "",
        "**Unresolved windows** (last PM price not ≥ 0.9 or ≤ 0.1 — excluded from all analysis):",
        "",
        unresolved_table,
        "",
        "---",
        "",
        "## 1. Summary — Edge by Asset / Sigma",
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
        "## 6. Velocity Into Trade — Does Faster = Better?",
        "",
        "How fast the coin was moving (in σ units) in the seconds before trigger,",
        "split by whether the trade won or lost.",
        "",
        section_velocity(df),
        "",
        "---",
        "",
        "## 7. Edge by Half-Day — Is It Profitable on All Days?",
        "",
        "Each half-day = 12-hour block (AM = 00:00–11:59 UTC, PM = 12:00–23:59 UTC).",
        "Rows sorted oldest → newest. Negative edge rows are highlighted with `[!]`.",
        "",
        section_half_day(df),
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
            f"  wallet_pct: 0.05"
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
    p.add_argument("--sigma",      nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    p.add_argument("--out-csv",    default="data/reports/threshold_edge.csv")
    p.add_argument("--out-report", default="data/reports/threshold_edge.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    pm_df = load_prices(args.prices_dir)

    all_records = []
    unresolved: dict[str, int] = {}
    for asset in args.assets:
        coin = load_coin_prices(args.coin_dir, asset)
        if coin is None:
            continue
        recs, n_unresolved = analyze_asset(asset, pm_df, coin, args.sigma)
        unresolved[asset] = n_unresolved
        if not recs.empty:
            all_records.append(recs)

    if not all_records:
        log.error("No data.")
        sys.exit(1)

    full = pd.concat(all_records, ignore_index=True)

    # compute target win rates then add reprice times
    target_win_rates = {}
    for key, grp in full.groupby(["asset", "sigma"]):
        asset, sig = key  # type: ignore[misc]
        target_win_rates[(asset, sig)] = grp["won"].mean()
    full = compute_reprice_times(full, pm_df, target_win_rates)

    full.to_csv(args.out_csv, index=False)
    log.info("Raw records → %s", args.out_csv)

    report = build_report(full, args.sigma, unresolved)

    with open(args.out_report, "w") as f:
        f.write(report)
    log.info("Report → %s", args.out_report)

if __name__ == "__main__":
    main()
