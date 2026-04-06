#!/usr/bin/env python3
"""
Live coin-momentum buy executor.

Watches the real coin price (Binance) within each 5-minute Polymarket window.
Buys when:
  1. Coin moves >= sigma_entry * sigma_value from window open
  2. Polymarket ask for that side is <= max_pm_price

Usage:
    python scripts/live_momentum_buy.py --asset SOL                           # loads from config/assets.yaml
    python scripts/live_momentum_buy.py --asset SOL --dry-run                 # dry-run using config
    python scripts/live_momentum_buy.py --asset BTC --sigma-value 150.0 --sigma-entry 1.0 --max-pm-price 0.72  # CLI override
"""
import argparse
import asyncio
import logging
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skeptic import config
from skeptic.clients import clob as clob_client
from skeptic.executor.momentum_buy import MomentumBuyExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def _load_asset_config(config_path: str, asset: str) -> tuple[dict, dict]:
    """Load per-asset params and MODEL section from YAML config file."""
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get(asset.upper(), {}), cfg.get("MODEL", {})
    except FileNotFoundError:
        return {}, {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live coin-momentum buy executor")
    p.add_argument("--asset",        required=True,
                   help="Asset to trade (e.g. BTC, DOGE, ETH)")
    p.add_argument("--config",       default="config/assets.yaml",
                   help="Path to per-asset YAML config (default: config/assets.yaml)")
    p.add_argument("--sigma-value",  type=float, default=None,
                   help="Sigma (std dev of 5-min window moves) — overrides config file")
    p.add_argument("--sigma-entry",  type=float, default=None,
                   help="Number of sigmas needed to trigger — overrides config file")
    p.add_argument("--max-pm-price", type=float, default=None,
                   help="Max Polymarket ask price to accept — overrides config file")
    p.add_argument("--direction",    default=None, choices=["up", "down", "both"],
                   help="Which direction to trade (default: both)")
    p.add_argument("--wallet-pct",   type=float, default=None,
                   help="Percent of wallet to deploy per trade (default: from config or POSITION_SIZE_PCT)")
    p.add_argument("--fixed-usdc",   type=float, default=None,
                   help="Fixed dollar amount per trade in USDC (overrides --wallet-pct)")
    p.add_argument("--dry-run",      action="store_true",
                   help="Log intended trades without placing real orders")
    p.add_argument("--name",         default="momentum",
                   help="Instance name for trades/status files")

    args = p.parse_args()

    # Fill unset args from config file
    asset_cfg, model_cfg = _load_asset_config(args.config, args.asset)
    args.model_cfg = model_cfg
    if args.sigma_value  is None: args.sigma_value  = asset_cfg.get("sigma_value")
    if args.sigma_entry  is None: args.sigma_entry  = asset_cfg.get("sigma_entry")
    if args.max_pm_price is None: args.max_pm_price = asset_cfg.get("max_pm_price")
    if args.direction    is None: args.direction    = asset_cfg.get("direction", "both")
    if args.name == "momentum":   args.name         = asset_cfg.get("name", args.name)
    if args.fixed_usdc   is None: args.fixed_usdc = asset_cfg.get("fixed_usdc")
    if args.wallet_pct   is None:
        args.wallet_pct = asset_cfg.get("wallet_pct", float(config.POSITION_SIZE_PCT)) * 100

    missing = [name for name, val in [
        ("--sigma-value",  args.sigma_value),
        ("--sigma-entry",  args.sigma_entry),
        ("--max-pm-price", args.max_pm_price),
    ] if val is None]
    if missing:
        p.error(f"{', '.join(missing)} required (pass as CLI args or set in {args.config})")

    return args


async def main() -> None:
    args = parse_args()
    wallet_pct = args.wallet_pct / 100.0

    client  = clob_client.build_client()
    balance = clob_client.get_usdc_balance(client)
    threshold_move = args.sigma_entry * args.sigma_value

    if args.fixed_usdc is not None:
        per_trade_str = f"${args.fixed_usdc:.2f} (fixed)"
    else:
        per_trade_str = f"{wallet_pct:.1%} of wallet (~${balance * wallet_pct:,.2f})"

    print(f"\nSkeptic Live Momentum-Buy Executor")
    print(f"  Asset         : {args.asset}")
    print(f"  Sigma value   : {args.sigma_value}")
    print(f"  Sigma entry   : {args.sigma_entry}σ  (triggers at ${threshold_move:.4f} move)")
    print(f"  Max PM price  : {args.max_pm_price:.2f}  (only buy if ask ≤ this)")
    print(f"  Direction     : {args.direction}")
    print(f"  Wallet        : {config.WALLET_ADDRESS}")
    print(f"  Balance       : ${balance:,.2f} USDC")
    print(f"  Per trade     : {per_trade_str}")
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
        fixed_usdc=args.fixed_usdc,
        dry_run=args.dry_run,
        name=args.name,
        model_cfg=args.model_cfg,
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
