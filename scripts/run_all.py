#!/usr/bin/env python3
"""
Run all asset executors concurrently in a single process.

Every executor runs continuously so EWMA sigma keeps updating from live windows.
Whether real orders are placed is controlled entirely by momentum.enabled and
model.enabled in config/assets.yaml — no separate dry-run flag needed.

A supervisor loop re-reads config/assets.yaml at the start of every 5-minute
window and updates each executor's enabled flags in-place, so sigma state is
never lost on enable/disable transitions.

Usage:
    python scripts/run_all.py                        # all assets in config
    python scripts/run_all.py --assets BTC DOGE      # subset of assets
    python scripts/run_all.py --yes                  # skip confirmation
"""
import argparse
import asyncio
import logging
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skeptic import config
from skeptic.clients import clob as clob_client
from skeptic.executor.momentum_buy import MomentumBuyExecutor
from skeptic.utils.time import next_window_start

log = logging.getLogger("supervisor")

ASSETS_YAML     = "config/assets.yaml"
_NON_ASSET_KEYS = {"MOMENTUM", "MODEL"}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    with open(ASSETS_YAML) as f:
        return yaml.safe_load(f) or {}


def _state_label(asset_cfg: dict) -> str:
    parts = []
    if asset_cfg.get("momentum", {}).get("enabled", False):
        parts.append("momentum")
    if asset_cfg.get("model", {}).get("enabled", False):
        parts.append("model")
    return " + ".join(parts) if parts else "watching (disabled)"


def _build_executor(asset: str, cfg: dict) -> MomentumBuyExecutor:
    global_momentum = cfg.get("MOMENTUM", {})
    global_model    = cfg.get("MODEL", {})
    asset_cfg       = cfg.get(asset.upper(), {})
    momentum_cfg    = {**global_momentum, **asset_cfg.get("momentum", {})}
    model_cfg       = {**global_model,    **asset_cfg.get("model",    {})}

    fixed_usdc = asset_cfg.get("fixed_usdc") or model_cfg.get("fixed_usdc")
    wallet_pct = asset_cfg.get("wallet_pct", float(config.POSITION_SIZE_PCT))

    return MomentumBuyExecutor(
        asset=asset,
        sigma_value=asset_cfg["sigma_value"],
        sigma_entry=asset_cfg["sigma_entry"],
        max_pm_price=asset_cfg["max_pm_price"],
        direction=asset_cfg.get("direction", "both"),
        wallet_pct=wallet_pct,
        fixed_usdc=fixed_usdc,
        name=asset_cfg.get("name", f"mom_{asset.lower()}"),
        model_cfg=model_cfg,
        momentum_cfg=momentum_cfg,
    )


# ---------------------------------------------------------------------------
# Per-executor wrapper (restarts on crash)
# ---------------------------------------------------------------------------

async def _run_executor(asset: str, executor: MomentumBuyExecutor) -> None:
    asset_log = logging.getLogger(asset)
    try:
        asset_log.info("starting")
        await executor.run()
    except asyncio.CancelledError:
        asset_log.info("stopped")
        raise
    except Exception as exc:
        asset_log.error("crashed: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

async def _supervisor(candidates: list[str], cfg: dict) -> None:
    """
    Start all candidate executors immediately. At each window boundary,
    reload config and update each executor's enabled flags in-place —
    no restarts, so EWMA sigma is never lost.
    """

    # Build and start all executors now
    executors: dict[str, MomentumBuyExecutor] = {}
    tasks: dict[str, asyncio.Task] = {}

    for asset in candidates:
        if asset not in cfg:
            log.warning("%-6s  not found in config, skipping", asset)
            continue
        ex = _build_executor(asset, cfg)
        executors[asset] = ex
        tasks[asset] = asyncio.create_task(_run_executor(asset, ex), name=asset)

    async def _reconcile(new_cfg: dict) -> None:
        for asset, ex in executors.items():
            asset_cfg = new_cfg.get(asset, {})
            new_mom = bool(asset_cfg.get("momentum", {}).get("enabled", False))
            new_mod = bool(asset_cfg.get("model",    {}).get("enabled", False))

            changed = []
            if ex._momentum_enabled != new_mom:
                ex._momentum_enabled = new_mom
                changed.append(f"momentum={new_mom}")
            if ex._model_enabled != new_mod:
                ex._model_enabled = new_mod
                changed.append(f"model={new_mod}")

            label = _state_label(asset_cfg)
            if changed:
                log.info("  ~~ %-6s  %s  (%s)", asset, label, ", ".join(changed))
            else:
                log.info("  == %-6s  %s", asset, label)

        # Restart any executors that crashed
        for asset, task in list(tasks.items()):
            if task.done() and not task.cancelled():
                log.warning("%-6s  restarting after crash", asset)
                ex = executors[asset]
                tasks[asset] = asyncio.create_task(_run_executor(asset, ex), name=asset)

    log.info("--- initial state ---")
    await _reconcile(cfg)

    while True:
        secs = next_window_start() - time.time()
        log.debug("next reconcile in %.0fs", secs)
        await asyncio.sleep(max(secs, 1))

        new_cfg = _load_cfg()
        log.info("--- window reconcile ---")
        await _reconcile(new_cfg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    p = argparse.ArgumentParser(description="Run all asset executors with per-window config reload")
    p.add_argument("--assets", nargs="+", metavar="ASSET",
                   help="Assets to manage (default: all in config)")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Skip live-trading confirmation prompt")
    args = p.parse_args()

    cfg = _load_cfg()
    candidates = (
        [a.upper() for a in args.assets]
        if args.assets
        else [k for k in cfg if k not in _NON_ASSET_KEYS]
    )

    client  = clob_client.build_client()
    balance = clob_client.get_usdc_balance(client)

    print(f"\nSkeptic — run_all")
    print(f"  Wallet  : {config.WALLET_ADDRESS}")
    print(f"  Balance : ${balance:,.2f} USDC")
    print(f"\n  Assets  :")
    for asset in candidates:
        if asset not in cfg:
            print(f"    {asset:<6}  (not in config)")
            continue
        asset_cfg = cfg[asset]
        print(f"    {asset:<6}  sigma={asset_cfg['sigma_entry']}σ  "
              f"max_pm={asset_cfg['max_pm_price']}  [{_state_label(asset_cfg)}]")
    secs_to_next = int(next_window_start() - time.time())
    print(f"\n  Config reloaded at every window boundary (~{secs_to_next}s to next)")
    print()

    live_assets = [
        a for a in candidates
        if a in cfg and (
            cfg[a].get("momentum", {}).get("enabled", False)
            or cfg[a].get("model", {}).get("enabled", False)
        )
    ]
    if not args.yes:
        if live_assets:
            confirm = input(f"Start LIVE trading for {', '.join(live_assets)}? [y/N] ").strip().lower()
            if confirm != "y":
                print("Aborted.")
                return
        else:
            print("No live assets currently enabled — starting in watch-only mode.")

    try:
        await _supervisor(candidates, cfg)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        print("\nShutting down.")


class _TornadoNoiseFilter(logging.Filter):
    _SUPPRESS = frozenset(("WebSocketClosedError", "StreamClosedError"))

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno == logging.ERROR and record.exc_info:
            exc = record.exc_info[1]
            if exc is not None and type(exc).__name__ in self._SUPPRESS:
                return False
        return True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%M:%S",
    )
    logging.getLogger("asyncio").addFilter(_TornadoNoiseFilter())
    asyncio.run(main())
