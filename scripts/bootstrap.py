#!/usr/bin/env python3
"""
Bootstrap stability analysis for threshold optimization.

Resamples historical sessions with replacement N times, runs the grid search
on each resample, and reports how consistently the same (buy, sell, fill_window)
region wins across runs.

  Stable winner  → the same region keeps winning → low overfitting risk
  Jumping winner → the best params are highly data-dependent → likely noise

Outputs:
  data/reports/bootstrap_<asset>.csv   — raw per-run results for each asset
  data/reports/bootstrap_report_*.md   — human-readable stability report

Usage:
    python scripts/bootstrap.py --from-prices
    python scripts/bootstrap.py --from-prices --n 200 --step 0.02
    python scripts/bootstrap.py --from-prices --fill-window-min 10 --fill-window-max 120
    python scripts/bootstrap.py --from-db --n 100
"""
import argparse
import asyncio
import logging
import os
import random
import sqlite3
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skeptic import config
from skeptic.research import analyzer, fetcher
from skeptic.research.fetcher import HistoricalSession

console = Console()
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reports")


# ── Session loaders (mirrors research.py) ────────────────────────────────────

def load_sessions_from_db(assets: list[str]) -> dict[str, list[HistoricalSession]]:
    if not os.path.exists(config.DB_PATH):
        return {}
    result: dict[str, list[HistoricalSession]] = {}
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        for asset in assets:
            rows = conn.execute(
                """
                SELECT s.*,
                    ps1_up.price AS up_price_m1_snap,
                    ps1_dn.price AS dn_price_m1_snap
                FROM sessions s
                LEFT JOIN price_snapshots ps1_up
                    ON ps1_up.session_id = s.session_id AND ps1_up.outcome='UP' AND ps1_up.minute_mark=1
                LEFT JOIN price_snapshots ps1_dn
                    ON ps1_dn.session_id = s.session_id AND ps1_dn.outcome='DOWN' AND ps1_dn.minute_mark=1
                WHERE s.asset = ?
                ORDER BY s.window_start_ts
                """,
                (asset,),
            ).fetchall()
            sessions = []
            for row in rows:
                hs = HistoricalSession(
                    asset=row["asset"],
                    condition_id=row["condition_id"],
                    window_start_ts=row["window_start_ts"],
                    up_token_id="",
                    down_token_id="",
                )
                up_m1 = row["up_price_m1_snap"] or row["up_price_m1"]
                dn_m1 = row["dn_price_m1_snap"] or row["down_price_m1"]
                if up_m1:
                    hs.up_trades_m1 = [float(up_m1)]
                if dn_m1:
                    hs.down_trades_m1 = [float(dn_m1)]
                res = row["resolution_price"]
                if res is not None:
                    hs.up_resolution = float(res)
                    hs.down_resolution = 1.0 - float(res)
                if row["sell_filled"] and row["sell_price_used"]:
                    sell_p = float(row["sell_price_used"])
                    if row["filled_outcome"] == "UP":
                        hs.up_trades_all = [sell_p]
                    elif row["filled_outcome"] == "DOWN":
                        hs.down_trades_all = [sell_p]
                sessions.append(hs)
            result[asset] = sessions
    finally:
        conn.close()
    return result


# ── Core bootstrap logic ──────────────────────────────────────────────────────

def run_bootstrap(
    sessions: list[HistoricalSession],
    n: int,
    buy_range: tuple[float, float],
    sell_range: tuple[float, float],
    step: float,
    fill_window: int,
    fill_window_range: tuple[int, int] | None,
    fill_window_step: int,
    seed: int | None,
) -> pd.DataFrame:
    """
    Run N bootstrap iterations. Each iteration resamples sessions with replacement,
    runs the grid search, and records the best params.

    Returns a DataFrame with one row per bootstrap run.
    """
    rng = random.Random(seed)
    do_3d = fill_window_range is not None

    rows = []
    for i in range(n):
        sample = rng.choices(sessions, k=len(sessions))
        if do_3d:
            df = analyzer.optimize_thresholds_3d(
                sample,
                buy_range=buy_range,
                sell_range=sell_range,
                step=step,
                fill_window_range=fill_window_range,
                fill_window_step=fill_window_step,
            )
        else:
            df = analyzer.optimize_thresholds(
                sample,
                buy_range=buy_range,
                sell_range=sell_range,
                step=step,
                fill_window=fill_window,
            )
        best = analyzer.best_params(df)
        if best:
            rows.append({
                "run":              i + 1,
                "buy":              best.get("buy"),
                "sell":             best.get("sell"),
                "fill_window":      best.get("fill_window", fill_window),
                "edge_per_session": best.get("edge_per_session"),
                "fill_rate":        best.get("fill_rate"),
                "sell_hit_rate":    best.get("sell_hit_rate"),
                "n_fills":          best.get("n_fills"),
            })

    return pd.DataFrame(rows)


# ── Statistics & report helpers ───────────────────────────────────────────────

def compute_stability(df: pd.DataFrame, step: float) -> dict:
    """
    Summarise the distribution of bootstrap winners.

    Stability score: fraction of runs whose best (buy, sell) falls within
    ±2 steps of the median.  Fill window is treated separately.
    """
    if df.empty:
        return {}

    med_buy  = df["buy"].median()
    med_sell = df["sell"].median()
    tol = step * 2

    near_consensus = (
        ((df["buy"]  - med_buy ).abs() <= tol + 1e-9) &
        ((df["sell"] - med_sell).abs() <= tol + 1e-9)
    )
    stability_score = float(near_consensus.mean())

    # Most common (binned) winner
    bin_buy  = (df["buy"]  / step).round() * step
    bin_sell = (df["sell"] / step).round() * step
    combo_counts = (
        pd.DataFrame({"buy": bin_buy.round(4), "sell": bin_sell.round(4)})
        .value_counts()
        .reset_index()
        .rename(columns={0: "count"})
    )
    top = combo_counts.iloc[0]

    label = (
        "stable"   if stability_score >= 0.70 else
        "moderate" if stability_score >= 0.40 else
        "unstable"
    )

    return {
        "n_runs":           len(df),
        "stability_score":  round(stability_score, 3),
        "stability_label":  label,
        "median_buy":       round(med_buy, 4),
        "median_sell":      round(med_sell, 4),
        "median_fw":        int(df["fill_window"].median()),
        "std_buy":          round(df["buy"].std(), 4),
        "std_sell":         round(df["sell"].std(), 4),
        "std_fw":           round(df["fill_window"].std(), 1),
        "mean_edge":        round(df["edge_per_session"].mean(), 6),
        "std_edge":         round(df["edge_per_session"].std(), 6),
        "top_buy":          float(top["buy"]),
        "top_sell":         float(top["sell"]),
        "top_count":        int(top["count"]),
        "top_pct":          round(int(top["count"]) / len(df), 3),
    }


def top_regions(df: pd.DataFrame, step: float, k: int = 10) -> pd.DataFrame:
    """Return the k most common winning (buy, sell, fill_window) bins."""
    if df.empty:
        return pd.DataFrame()
    fw_step = step * 100  # rough fill_window bin — overridden below
    # Use actual fill_window values as bins (they're already discrete)
    binned = pd.DataFrame({
        "buy":         ((df["buy"]  / step).round() * step).round(4),
        "sell":        ((df["sell"] / step).round() * step).round(4),
        "fill_window": df["fill_window"].astype(int),
    })
    counts = binned.value_counts().reset_index().rename(columns={0: "count"})
    counts["pct"] = (counts["count"] / len(df) * 100).round(1)
    return counts.head(k)


def write_report(
    all_sessions: dict[str, list[HistoricalSession]],
    per_asset_df: dict[str, pd.DataFrame],
    per_asset_stats: dict[str, dict],
    args: argparse.Namespace,
) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    import glob as _glob
    for _old in _glob.glob(os.path.join(REPORTS_DIR, "bootstrap_report_*.md")):
        os.remove(_old)
    import random as _r
    suffix = _r.randint(10000, 99999)
    path = os.path.join(REPORTS_DIR, f"bootstrap_report_{suffix}.md")
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Bootstrap Stability Report",
        f"_Generated: {now}_",
        "",
        "## What this measures",
        "",
        "Each asset's sessions are resampled with replacement **N times**. The full grid "
        "search is re-run on each resample and the winning (buy, sell, fill_window) is recorded.",
        "",
        "- **Stable** winner → same region wins most runs → parameter choice is robust, not a data fluke",
        "- **Unstable** winner → winner jumps around → edge may be noise, interpret results cautiously",
        "",
        f"_Settings: N={args.n} runs, step={args.step}, "
        f"buy {args.buy_min}–{args.buy_max}, sell {args.sell_min}–{args.sell_max}_",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Asset | N Sessions | Stability | Median Buy | Median Sell | Median FW | σ Buy | σ Sell | Mean Edge | σ Edge | Top Region | Top Region % |",
        "|-------|------------|-----------|------------|-------------|-----------|-------|--------|-----------|--------|------------|--------------|",
    ]

    for asset, stats in per_asset_stats.items():
        n_sess = len(all_sessions.get(asset, []))
        top = f"buy={stats['top_buy']} sell={stats['top_sell']}"
        lines.append(
            f"| {asset} | {n_sess} | **{stats['stability_label']}** ({stats['stability_score']:.0%}) "
            f"| {stats['median_buy']} | {stats['median_sell']} | {stats['median_fw']}s "
            f"| {stats['std_buy']} | {stats['std_sell']} "
            f"| {stats['mean_edge']} | {stats['std_edge']} "
            f"| {top} | {stats['top_pct']:.0%} |"
        )

    for asset, df in per_asset_df.items():
        stats = per_asset_stats.get(asset, {})
        if not stats:
            continue
        lines += [
            "",
            "---",
            "",
            f"## {asset}",
            "",
            f"**Stability: {stats['stability_label']}** ({stats['stability_score']:.0%} of runs within ±{args.step*2:.2f} of median)",
            "",
            f"| Metric | Buy | Sell | Fill Window | Edge/Session |",
            f"|--------|-----|------|-------------|--------------|",
            f"| Median | {stats['median_buy']} | {stats['median_sell']} | {stats['median_fw']}s | {stats['mean_edge']} |",
            f"| Std dev | {stats['std_buy']} | {stats['std_sell']} | {stats['std_fw']}s | {stats['std_edge']} |",
            "",
            "### Top winning regions",
            "",
        ]
        top_df = top_regions(df, args.step)
        if not top_df.empty:
            lines.append(top_df.to_markdown(index=False))
        else:
            lines.append("_No data._")

        # Stability interpretation
        lines += [
            "",
            "### Interpretation",
            "",
        ]
        label = stats["stability_label"]
        score = stats["stability_score"]
        std_buy = stats["std_buy"]
        std_sell = stats["std_sell"]
        if label == "stable":
            lines.append(
                f"The best parameters are **consistent across bootstrap runs** (score={score:.0%}). "
                f"The winning region does not depend heavily on which specific sessions were included. "
                f"σ buy={std_buy}, σ sell={std_sell} — low spread. This is a positive signal."
            )
        elif label == "moderate":
            lines.append(
                f"The best parameters show **moderate consistency** (score={score:.0%}). "
                f"σ buy={std_buy}, σ sell={std_sell}. "
                f"The edge may be real but treat the exact thresholds as approximate — "
                f"consider using the median rather than any single run's winner."
            )
        else:
            lines.append(
                f"⚠️ The best parameters are **unstable across bootstrap runs** (score={score:.0%}). "
                f"σ buy={std_buy}, σ sell={std_sell} — high spread. "
                f"Different subsets of the data produce very different winners. "
                f"This suggests the observed edge may be noise. "
                f"Collect more data before trading with confidence."
            )

    content = "\n".join(lines)
    with open(path, "w") as f:
        f.write(content)
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap stability analysis for threshold optimization")
    p.add_argument("--from-prices", action="store_true", help="Load from data/prices/ CSVs")
    p.add_argument("--from-db",     action="store_true", help="Load from sessions.db")
    p.add_argument("--assets", nargs="+", default=config.ASSETS)
    p.add_argument("--n",    type=int,   default=100,  help="Number of bootstrap iterations (default: 100)")
    p.add_argument("--step", type=float, default=0.03, help="Grid step size (default: 0.03)")
    p.add_argument("--buy-min",  type=float, default=0.10)
    p.add_argument("--buy-max",  type=float, default=0.49)
    p.add_argument("--sell-min", type=float, default=0.45)
    p.add_argument("--sell-max", type=float, default=0.96)
    p.add_argument("--fill-window",      type=int, default=60,   help="Single fill window in seconds (default: 60)")
    p.add_argument("--fill-window-min",  type=int, default=None, help="Min fill window for 3D bootstrap")
    p.add_argument("--fill-window-max",  type=int, default=None, help="Max fill window for 3D bootstrap")
    p.add_argument("--fill-window-step", type=int, default=10)
    p.add_argument("--min-points", type=int, default=280, help="Min price data points per window (prices mode)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load sessions ──────────────────────────────────────────────────────────
    if args.from_prices:
        console.print("Mode: [green]Price CSV analysis[/green]")
        all_sessions = fetcher.load_from_price_files(args.assets, min_points=args.min_points)
    elif args.from_db:
        console.print("Mode: [green]DB analysis[/green]")
        all_sessions = load_sessions_from_db(args.assets)
    else:
        console.print("[red]Specify --from-prices or --from-db.[/red]")
        console.print("Bootstrap requires local session data (API mode has too few sessions).")
        raise SystemExit(1)

    total = sum(len(s) for s in all_sessions.values())
    if total == 0:
        console.print("[red]No sessions found.[/red]")
        raise SystemExit(1)

    console.print(f"Loaded [bold]{total}[/bold] sessions across {len(all_sessions)} assets\n")

    fw_range = (
        (args.fill_window_min, args.fill_window_max)
        if args.fill_window_min is not None and args.fill_window_max is not None
        else None
    )

    buy_range  = (args.buy_min,  args.buy_max)
    sell_range = (args.sell_min, args.sell_max)

    n_grid_pts = (
        int((args.buy_max  - args.buy_min)  / args.step + 1) *
        int((args.sell_max - args.sell_min) / args.step + 1)
    )
    if fw_range:
        n_fw = len(range(fw_range[0], fw_range[1] + args.fill_window_step, args.fill_window_step))
        n_grid_pts *= n_fw

    console.print(
        f"[dim]Grid: ~{n_grid_pts:,} combinations per run × {args.n} runs "
        f"({'3D with fill window sweep' if fw_range else f'fill_window={args.fill_window}s'})[/dim]\n"
    )

    # ── Bootstrap per asset ────────────────────────────────────────────────────
    os.makedirs(REPORTS_DIR, exist_ok=True)

    per_asset_df:    dict[str, pd.DataFrame] = {}
    per_asset_stats: dict[str, dict]         = {}

    for asset, sessions in all_sessions.items():
        if not sessions:
            continue

        console.print(f"[bold]{asset}[/bold] — {len(sessions)} sessions, {args.n} bootstrap runs")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"  bootstrapping {asset}", total=args.n)

            # Run one iteration at a time so we can update the progress bar
            rng = random.Random(args.seed)
            rows = []
            do_3d = fw_range is not None

            for i in range(args.n):
                sample = rng.choices(sessions, k=len(sessions))
                if do_3d:
                    df = analyzer.optimize_thresholds_3d(
                        sample,
                        buy_range=buy_range,
                        sell_range=sell_range,
                        step=args.step,
                        fill_window_range=fw_range,
                        fill_window_step=args.fill_window_step,
                    )
                else:
                    df = analyzer.optimize_thresholds(
                        sample,
                        buy_range=buy_range,
                        sell_range=sell_range,
                        step=args.step,
                        fill_window=args.fill_window,
                    )
                best = analyzer.best_params(df)
                if best:
                    rows.append({
                        "run":              i + 1,
                        "buy":              best.get("buy"),
                        "sell":             best.get("sell"),
                        "fill_window":      best.get("fill_window", args.fill_window),
                        "edge_per_session": best.get("edge_per_session"),
                        "fill_rate":        best.get("fill_rate"),
                        "sell_hit_rate":    best.get("sell_hit_rate"),
                        "n_fills":          best.get("n_fills"),
                    })
                progress.advance(task)

        result_df = pd.DataFrame(rows)
        stats = compute_stability(result_df, args.step)
        per_asset_df[asset]    = result_df
        per_asset_stats[asset] = stats

        # Save raw bootstrap runs to CSV
        csv_path = os.path.join(REPORTS_DIR, f"bootstrap_{asset.lower()}.csv")
        result_df.to_csv(csv_path, index=False)

        # Print per-asset summary
        label  = stats.get("stability_label", "?")
        score  = stats.get("stability_score", 0)
        color  = "green" if label == "stable" else ("yellow" if label == "moderate" else "red")
        top_pct = stats.get("top_pct", 0)
        console.print(
            f"  Stability: [{color}]{label}[/{color}] ({score:.0%})  "
            f"median buy={stats.get('median_buy')} sell={stats.get('median_sell')} "
            f"fw={stats.get('median_fw')}s  "
            f"σ buy={stats.get('std_buy')} sell={stats.get('std_sell')}  "
            f"top region wins {top_pct:.0%} of runs"
        )

        # Show top regions table
        top_df = top_regions(result_df, args.step, k=5)
        if not top_df.empty:
            tbl = Table(show_header=True, header_style="bold", box=None)
            for col in top_df.columns:
                tbl.add_column(col)
            for _, row in top_df.iterrows():
                tbl.add_row(*[str(v) for v in row])
            console.print(tbl)

        console.print()

    # ── Write report ───────────────────────────────────────────────────────────
    report_path = write_report(all_sessions, per_asset_df, per_asset_stats, args)
    console.print(f"[bold green]Bootstrap complete! Report: {report_path}[/bold green]")
    console.print(f"Raw CSVs: data/reports/bootstrap_<asset>.csv")


if __name__ == "__main__":
    main()
