#!/usr/bin/env python3
"""
Live coin-momentum buy executor.

Watches the real coin price (Binance) within each 5-minute Polymarket window.
Buys when:
  1. Coin moves >= sigma_entry * sigma_value from window open
  2. Polymarket ask for that side is <= max_pm_price

Usage:
    python scripts/live_momentum_buy.py --asset BTC --sigma-value 150.0 --sigma-entry 1.0 --max-pm-price 0.72
    python scripts/live_momentum_buy.py --asset DOGE --sigma-value 0.005 --sigma-entry 0.5 --max-pm-price 0.65 --direction up
    python scripts/live_momentum_buy.py --asset BTC --sigma-value 150.0 --sigma-entry 1.0 --max-pm-price 0.72 --dry-run
"""
import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skeptic import config
from skeptic.clients import clob as clob_client
from skeptic.executor.momentum_buy import MomentumBuyExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live coin-momentum buy executor")
    p.add_argument("--asset",        required=True,
                   help="Asset to trade (e.g. BTC, DOGE, ETH)")
    p.add_argument("--sigma-value",  type=float, required=True,
                   help="Sigma (std dev of 5-min window moves in USD) for the asset")
    p.add_argument("--sigma-entry",  type=float, required=True,
                   help="Number of sigmas the coin must move to trigger (e.g. 1.0)")
    p.add_argument("--max-pm-price", type=float, required=True,
                   help="Max Polymarket ask price to accept (our edge threshold, e.g. 0.72)")
    p.add_argument("--direction",    default="both", choices=["up", "down", "both"],
                   help="Which direction to trade (default: both)")
    p.add_argument("--wallet-pct",   type=float, default=float(config.POSITION_SIZE_PCT) * 100,
                   help="Percent of wallet to deploy per trade (default: %(default)s%%)")
    p.add_argument("--dry-run",      action="store_true",
                   help="Log intended trades without placing real orders")
    p.add_argument("--name",         default="momentum",
                   help="Instance name for trades/status files")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    wallet_pct = args.wallet_pct / 100.0

    client  = clob_client.build_client()
    balance = clob_client.get_usdc_balance(client)
    threshold_move = args.sigma_entry * args.sigma_value

    print(f"\nSkeptic Live Momentum-Buy Executor")
    print(f"  Asset         : {args.asset}")
    print(f"  Sigma value   : {args.sigma_value}")
    print(f"  Sigma entry   : {args.sigma_entry}σ  (triggers at ${threshold_move:.4f} move)")
    print(f"  Max PM price  : {args.max_pm_price:.2f}  (only buy if ask ≤ this)")
    print(f"  Direction     : {args.direction}")
    print(f"  Wallet        : {config.WALLET_ADDRESS}")
    print(f"  Balance       : ${balance:,.2f} USDC")
    print(f"  Per trade     : {wallet_pct:.1%} of wallet (~${balance * wallet_pct:,.2f})")
    print(f"  Dry run       : {args.dry_run}\n")

    if not args.dry_run:
        confirm = input("Start LIVE trading? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    executor = MomentumBuyExecutor(
        asset=args.asset,
        sigma_value=args.sigma_value,
        sigma_entry=args.sigma_entry,
        max_pm_price=args.max_pm_price,
        direction=args.direction,
        wallet_pct=wallet_pct,
        dry_run=args.dry_run,
        name=args.name,
    )
    await executor.run()


class _TornadoNoiseFilter(logging.Filter):
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
