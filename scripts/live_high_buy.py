#!/usr/bin/env python3
"""
Live high-probability buy executor.

Watches Polymarket 5-min UP/DOWN markets in real time and buys the first side
that hits the threshold price each window.

Usage:
    python scripts/live_high_buy.py --threshold 0.80
    python scripts/live_high_buy.py --threshold 0.80 --wallet-pct 10
    python scripts/live_high_buy.py --threshold 0.80 --dry-run
"""
import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skeptic import config
from skeptic.clients import clob as clob_client
from skeptic.executor.high_buy import HighBuyExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live high-buy executor")
    p.add_argument("--threshold", type=float, required=True,
                   help="Buy trigger price (e.g. 0.80)")
    p.add_argument("--assets", nargs="+", default=config.ASSETS,
                   help="Assets to monitor (default: all configured assets)")
    p.add_argument("--wallet-pct", type=float, default=float(config.POSITION_SIZE_PCT) * 100,
                   help="Percent of wallet balance to deploy per trade (default: %(default)s%%)")
    p.add_argument("--dry-run", action="store_true",
                   help="Log intended trades without placing real orders")
    p.add_argument("--name", default="default",
                   help="Instance name — used to separate trades/status files when running multiple instances (e.g. 'btc', 'doge')")
    p.add_argument("--cutoff", type=int, default=0,
                   help="Seconds into window before triggering buys (0 = any time, 180 = last 2 min)")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    wallet_pct = args.wallet_pct / 100.0

    # Show live balance at startup
    client = clob_client.build_client()
    balance = clob_client.get_usdc_balance(client)

    print(f"\nSkeptic Live High-Buy Executor")
    print(f"  Threshold   : {args.threshold:.2f}")
    print(f"  Assets      : {args.assets}")
    print(f"  Wallet      : {config.WALLET_ADDRESS}")
    print(f"  Balance     : ${balance:,.2f} USDC")
    print(f"  Per trade   : {wallet_pct:.1%} of wallet (~${balance * wallet_pct:,.2f} at current balance)")
    print(f"  Poll rate   : 10ms")
    print(f"  Cutoff      : {'any time' if args.cutoff == 0 else f'after {args.cutoff}s (last {(300 - args.cutoff) // 60}m {(300 - args.cutoff) % 60}s)'.replace(' 0s', '')}")
    print(f"  Dry run     : {args.dry_run}\n")

    if not args.dry_run:
        confirm = input("Start LIVE trading? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    executor = HighBuyExecutor(
        assets=args.assets,
        threshold=args.threshold,
        wallet_pct=wallet_pct,
        dry_run=args.dry_run,
        name=args.name,
        cutoff_secs=args.cutoff,
    )
    await executor.run()


class _TornadoNoiseFilter(logging.Filter):
    """Drop 'Task exception was never retrieved' noise from py-clob-client's tornado internals."""
    _SUPPRESS = frozenset(("WebSocketClosedError", "StreamClosedError"))

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno == logging.ERROR and record.exc_info:
            exc = record.exc_info[1]
            if exc is not None and type(exc).__name__ in self._SUPPRESS:
                return False
        return True


if __name__ == "__main__":
    logging.getLogger("asyncio").addFilter(_TornadoNoiseFilter())
    asyncio.run(main())
