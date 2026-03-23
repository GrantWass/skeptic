#!/usr/bin/env python3
"""
Live trading bot CLI.

Usage:
    python scripts/trade.py                          # live trading, all configured assets
    python scripts/trade.py --dry-run               # paper mode — logs orders, no real trades
    python scripts/trade.py --assets BTC ETH        # trade only BTC and ETH
    python scripts/trade.py --buy 0.32 --sell 0.60  # override thresholds from config
"""
import argparse
import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console

from skeptic import config
from skeptic.strategy import runner

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Skeptic trading bot")
    p.add_argument(
        "--assets", nargs="+", default=None,
        help=f"Assets to trade (default: {config.ASSETS})"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Log orders without submitting to the CLOB"
    )
    p.add_argument(
        "--buy", type=float, default=None,
        help=f"Override buy threshold (default: {config.BUY_PRICE})"
    )
    p.add_argument(
        "--sell", type=float, default=None,
        help=f"Override sell threshold (default: {config.SELL_PRICE})"
    )
    return p.parse_args()


async def main(args: argparse.Namespace) -> None:
    # Apply CLI overrides to config at runtime
    if args.buy is not None:
        config.BUY_PRICE = args.buy
        console.print(f"[yellow]Override: BUY_PRICE = {config.BUY_PRICE}[/yellow]")
    if args.sell is not None:
        config.SELL_PRICE = args.sell
        console.print(f"[yellow]Override: SELL_PRICE = {config.SELL_PRICE}[/yellow]")

    mode = "[bold red]DRY RUN[/bold red]" if args.dry_run else "[bold green]LIVE[/bold green]"
    console.print(f"\n[bold]Skeptic Trading Bot — {mode}[/bold]")
    console.print(f"Buy @ {config.BUY_PRICE:.2f}  |  Sell @ {config.SELL_PRICE:.2f}  |  "
                  f"Size = {config.POSITION_SIZE_PCT:.0%} capital per window")
    console.print(f"Monitor window: {config.MONITOR_SECS}s\n")

    try:
        await runner.run(assets=args.assets, dry_run=args.dry_run)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped by user.[/yellow]")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
