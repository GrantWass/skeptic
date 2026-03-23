#!/usr/bin/env python3
"""
Research phase CLI.

Three modes:
  --from-prices  Analyze per-second price CSVs from collect_prices.py (recommended)
  --from-db      Analyze session snapshots from paper/live trading bot
  (default)      Fetch historical market data from the Polymarket API

Usage:
    # Collect prices first, then analyze:
    python scripts/collect_prices.py          # run for hours/days
    python scripts/research.py --from-prices

    # After paper-trading to build up data:
    python scripts/research.py --from-db

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
    p.add_argument("--from-db", action="store_true",
                   help="Analyze session snapshots from sessions.db")
    p.add_argument("--assets", nargs="+", default=config.ASSETS, help="Assets to research")
    p.add_argument("--limit", type=int, default=200, help="Historical markets to fetch per asset (API mode)")
    p.add_argument("--buy-min", type=float, default=0.20)
    p.add_argument("--buy-max", type=float, default=0.49)
    p.add_argument("--sell-min", type=float, default=0.51)
    p.add_argument("--sell-max", type=float, default=0.90)
    p.add_argument("--step", type=float, default=0.01)
    p.add_argument("--capital", type=float, default=500.0, help="Starting capital in USDC for profit estimates")
    p.add_argument("--position-size", type=float, default=config.POSITION_SIZE_PCT,
                   help="Fraction of capital per trade (default: %(default)s)")
    p.add_argument("--spread-cost", type=float, default=0.002,
                   help="Estimated spread cost per share per crossing (default: 0.002). "
                        "Polymarket charges 0%% maker/taker fees; spread is the only cost.")
    return p.parse_args()


def load_sessions_from_db(assets: list[str]) -> dict[str, list[HistoricalSession]]:
    """
    Load paper/live trading session data from sessions.db.
    Converts TradingSession rows + price_snapshots into HistoricalSession objects
    that the analyzer can process.
    """
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
                    ps0_up.price  AS up_price_open_snap,
                    ps0_dn.price  AS dn_price_open_snap,
                    ps1_up.price  AS up_price_m1_snap,
                    ps1_dn.price  AS dn_price_m1_snap
                FROM sessions s
                LEFT JOIN price_snapshots ps0_up
                    ON ps0_up.session_id = s.session_id AND ps0_up.outcome='UP' AND ps0_up.minute_mark=0
                LEFT JOIN price_snapshots ps0_dn
                    ON ps0_dn.session_id = s.session_id AND ps0_dn.outcome='DOWN' AND ps0_dn.minute_mark=0
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

                # Minute-1 prices as single-element trade lists (the price snapshot)
                up_m1 = row["up_price_m1_snap"] or row["up_price_m1"]
                dn_m1 = row["dn_price_m1_snap"] or row["down_price_m1"]
                if up_m1:
                    hs.up_trades_m1 = [float(up_m1)]
                if dn_m1:
                    hs.down_trades_m1 = [float(dn_m1)]

                # Resolution price (1.0 = win, 0.0 = loss)
                res = row["resolution_price"]
                if res is not None:
                    hs.up_resolution = float(res)
                    hs.down_resolution = 1.0 - float(res)

                # If we have fill + sell data, add the sell price as an "all" trade
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


async def run_api_mode(args: argparse.Namespace) -> dict[str, list[HistoricalSession]]:
    client = clob_client.build_client()
    return await fetcher.fetch_all_assets(args.assets, client, limit=args.limit)


async def main(args: argparse.Namespace) -> None:
    console.print("[bold cyan]Skeptic Research Phase[/bold cyan]")

    if args.from_prices:
        console.print(f"Mode: [green]Price CSV analysis[/green] (data/prices/)")
        all_sessions = fetcher.load_from_price_files(args.assets)
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

    elif args.from_db:
        console.print(f"Mode: [green]DB analysis[/green] (sessions.db)")
        all_sessions = load_sessions_from_db(args.assets)
        total = sum(len(s) for s in all_sessions.values())
        console.print(f"Loaded {total} sessions from {config.DB_PATH}\n")

        if total == 0:
            console.print(
                "[yellow]No sessions in DB yet.[/yellow]\n"
                "Run the paper trading bot first to collect data:\n"
                "  [bold]python scripts/trade.py --dry-run[/bold]\n"
                "Then re-run research after accumulating sessions."
            )
            return
    else:
        console.print(f"Mode: [yellow]API fetch[/yellow] (up to {args.limit} markets/asset)")
        console.print(
            "[dim]Note: 5-minute markets are new (~days old). API data is sparse.\n"
            "For better analysis, run paper mode first, then use --from-db.[/dim]\n"
        )
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

    # Grid search per asset
    per_asset_best: dict[str, dict] = {}
    for asset, sessions in all_sessions.items():
        if not sessions:
            continue
        console.print(f"[dim]Optimizing thresholds for {asset} ({len(sessions)} sessions)…[/dim]")
        df = analyzer.optimize_thresholds(
            sessions,
            buy_range=(args.buy_min, args.buy_max),
            sell_range=(args.sell_min, args.sell_max),
            step=args.step,
        )
        reporter.save_full_grid(asset, df)
        best = analyzer.best_params(df)
        per_asset_best[asset] = best
        if best:
            console.print(
                f"  {asset}: best buy={best['buy']:.2f}  sell={best['sell']:.2f}  "
                f"edge={best['edge_per_session']:.4f}  fill_rate={best['fill_rate']:.2%}"
            )

    # Asset ranking at current thresholds
    asset_ranking = analyzer.rank_assets(all_sessions, buy=config.BUY_PRICE, sell=config.SELL_PRICE)
    reporter.save_asset_ranking(asset_ranking)

    table = Table(title=f"Asset Ranking (buy={config.BUY_PRICE:.2f}, sell={config.SELL_PRICE:.2f})")
    for col in asset_ranking.columns:
        table.add_column(col)
    for _, row in asset_ranking.iterrows():
        table.add_row(*[str(v) for v in row])
    console.print(table)

    data_source = "prices" if args.from_prices else "db" if args.from_db else "api"
    reporter.save_optimal_params(per_asset_best, {})
    report_path = reporter.write_report(
        per_asset_best, asset_ranking,
        data_source=data_source,
        capital=args.capital,
        position_size_pct=args.position_size,
        spread_cost=args.spread_cost,
    )
    console.print(f"\n[bold green]Research complete! Report: {report_path}[/bold green]")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
