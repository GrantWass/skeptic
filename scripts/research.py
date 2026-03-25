#!/usr/bin/env python3
"""
Research phase CLI.

Three modes:
  --from-prices  Analyze per-second price CSVs from collect_prices.py (recommended)
  (default)      Fetch historical market data from the Polymarket API

Usage:
    # Collect prices first, then analyze:
    python scripts/collect_prices.py          # run for hours/days
    python scripts/research.py --from-prices

    # API-based historical fetch (limited — markets are new):
    python scripts/research.py --assets BTC ETH SOL --limit 200

    # Grid search with custom bounds:
    python scripts/research.py --from-prices --buy-min 0.25 --buy-max 0.45
"""
import argparse
import asyncio
import logging
import sqlite3
import sys
import os
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from rich.console import Console
from rich.table import Table

from skeptic import config
from skeptic.clients import clob as clob_client
from skeptic.research import fetcher, analyzer, reporter
from skeptic.research.fetcher import HistoricalSession

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Skeptic research phase")
    p.add_argument("--from-prices", action="store_true",
                   help="Analyze per-second price CSVs from collect_prices.py (best option)")
    p.add_argument("--assets", nargs="+", default=config.ASSETS, help="Assets to research")
    p.add_argument("--limit", type=int, default=200, help="Historical markets to fetch per asset (API mode)")
    p.add_argument("--buy-min", type=float, default=0.10)
    p.add_argument("--buy-max", type=float, default=0.49)
    p.add_argument("--sell-min", type=float, default=0.45)
    p.add_argument("--sell-max", type=float, default=0.96)
    p.add_argument("--step", type=float, default=0.03)
    p.add_argument("--capital", type=float, default=500.0, help="Starting capital in USDC for profit estimates")
    p.add_argument("--position-size", type=float, default=config.POSITION_SIZE_PCT,
                   help="Fraction of capital per trade (default: %(default)s)")
    p.add_argument("--min-points", type=int, default=280,
                   help="Minimum data points for a window to be included (default: 280)")
    p.add_argument("--spread-cost", type=float, default=0.002,
                   help="Estimated spread cost per share per crossing (default: 0.002). "
                        "Polymarket charges 0%% maker/taker fees; spread is the only cost.")
    p.add_argument("--fill-window", type=int, default=60,
                   help="Single fill window in seconds (default: 60). Ignored if --fill-window-min/max set.")
    p.add_argument("--fill-window-min", type=int, default=None,
                   help="Min fill window for 3D grid search (e.g. 10).")
    p.add_argument("--fill-window-max", type=int, default=None,
                   help="Max fill window for 3D grid search (e.g. 60).")
    p.add_argument("--fill-window-step", type=int, default=10,
                   help="Step size for fill window sweep (default: 10).")
    return p.parse_args()


async def run_api_mode(args: argparse.Namespace) -> dict[str, list[HistoricalSession]]:
    client = clob_client.build_client()
    return await fetcher.fetch_all_assets(args.assets, client, limit=args.limit)


async def main(args: argparse.Namespace) -> None:
    console.print("[bold cyan]Skeptic Research Phase[/bold cyan]")

    if args.from_prices:
        console.print(f"Mode: [green]Price CSV analysis[/green] (data/prices/)")
        all_sessions = fetcher.load_from_price_files(args.assets, min_points=args.min_points)
        total = sum(len(s) for s in all_sessions.values())
        console.print(f"Loaded {total} sessions from price CSVs\n")

        if total == 0:
            console.print(
                "[yellow]No price data found.[/yellow]\n"
                "Run the price collector first:\n"
                "  [bold]python scripts/collect_prices.py[/bold]\n"
                "Let it run for at least a few hours, then re-run research."
            )
            return

    else:
        console.print(f"Mode: [yellow]API fetch[/yellow] (up to {args.limit} markets/asset)")
        all_sessions = await run_api_mode(args)
        total = sum(len(s) for s in all_sessions.values())
        console.print(f"\nFetched {total} historical sessions\n")

        if total == 0:
            console.print("[red]No data found.[/red]")
            return

        if total < 30:
            console.print(
                f"[yellow]Warning: only {total} sessions. Results may not be statistically meaningful.\n"
                f"Consider running paper mode first to accumulate more data.[/yellow]\n"
            )

    # Determine whether to do a 3D sweep or a single fill-window search
    _fw_min = args.fill_window_min
    _fw_max = args.fill_window_max
    _do_3d  = _fw_min is not None and _fw_max is not None and _fw_max > _fw_min

    # Grid search per asset
    per_asset_best: dict[str, dict] = {}
    per_asset_robustness: dict[str, dict] = {}
    per_asset_best_nb: dict[str, dict] = {}
    per_asset_best_30pct: dict[str, dict] = {}
    for asset, sessions in all_sessions.items():
        if not sessions:
            continue
        if _do_3d:
            console.print(
                f"[dim]3D grid search for {asset} ({len(sessions)} sessions, "
                f"fill window {_fw_min}–{_fw_max}s step {args.fill_window_step}s)…[/dim]"
            )
            df = analyzer.optimize_thresholds_3d(
                sessions,
                buy_range=(args.buy_min, args.buy_max),
                sell_range=(args.sell_min, args.sell_max),
                step=args.step,
                fill_window_range=(_fw_min, _fw_max),
                fill_window_step=args.fill_window_step,
            )
        else:
            console.print(f"[dim]Optimizing thresholds for {asset} ({len(sessions)} sessions)…[/dim]")
            df = analyzer.optimize_thresholds(
                sessions,
                buy_range=(args.buy_min, args.buy_max),
                sell_range=(args.sell_min, args.sell_max),
                step=args.step,
                fill_window=args.fill_window,
            )
        reporter.save_full_grid(asset, df)
        best = analyzer.best_params(df)
        robustness = analyzer.neighborhood_robustness(df, best)
        best_nb = analyzer.best_neighborhood_params(df)
        best_30pct = analyzer.best_params_min_fill_rate(df, 0.30)
        per_asset_best[asset] = best
        per_asset_robustness[asset] = robustness
        per_asset_best_nb[asset] = best_nb
        per_asset_best_30pct[asset] = best_30pct
        if best:
            fw_str = f"  fill_window={int(best['fill_window'])}s" if "fill_window" in best else ""
            shape_str = f"  [{robustness.get('shape', '?')}  ratio={robustness.get('robustness_ratio')}]" if robustness else ""
            nb_str = ""
            if best_nb and best_nb.get("peak_vs_neighborhood") != "agree":
                nb_str = (
                    f"  [best neighborhood: buy={best_nb['buy']} sell={best_nb['sell']}"
                    f"  mean={best_nb['neighborhood_mean_edge']}  {best_nb['peak_vs_neighborhood']}]"
                )
            console.print(
                f"  {asset}: best buy={best['buy']:.2f}  sell={best['sell']:.2f}{fw_str}  "
                f"edge={best['edge_per_session']:.4f}  fill_rate={best['fill_rate']:.2%}{shape_str}{nb_str}"
            )

    # Asset ranking at current thresholds (only shown if thresholds are configured)
    if config.BUY_PRICE is not None and config.SELL_PRICE is not None:
        asset_ranking = analyzer.rank_assets(all_sessions, buy=config.BUY_PRICE, sell=config.SELL_PRICE, fill_window=args.fill_window)
        reporter.save_asset_ranking(asset_ranking)
        table = Table(title=f"Asset Ranking (buy={config.BUY_PRICE:.2f}, sell={config.SELL_PRICE:.2f})")
        for col in asset_ranking.columns:
            table.add_column(col)
        for _, row in asset_ranking.iterrows():
            table.add_row(*[str(v) for v in row])
        console.print(table)
    else:
        asset_ranking = pd.DataFrame()
        console.print(
            "[dim]Skipping asset ranking — BUY_PRICE/SELL_PRICE not set in config. "
            "Set them from the optimal thresholds above and re-run to compare assets.[/dim]"
        )

    data_source = "prices" if args.from_prices else "api"
    reporter.save_optimal_params(per_asset_best, {})
    report_path = reporter.write_report(
        per_asset_best, asset_ranking,
        per_asset_robustness=per_asset_robustness,
        per_asset_best_nb=per_asset_best_nb,
        per_asset_best_30pct=per_asset_best_30pct,
        data_source=data_source,
        capital=args.capital,
        position_size_pct=args.position_size,
        spread_cost=args.spread_cost,
    )
    console.print(f"\n[bold green]Research complete! Report: {report_path}[/bold green]")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
