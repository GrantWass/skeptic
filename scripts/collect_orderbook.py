#!/usr/bin/env python3
"""
Orderbook depth collector: logs Binance top-20 bid/ask depth snapshots for all assets.

Writes to data/orderbook/orderbook_YYYYMMDD.csv (gitignored) with columns:
  ts, asset, last_update_id,
  bid_px_1..20, bid_sz_1..20,
  ask_px_1..20, ask_sz_1..20,
  bid_vol, ask_vol, imbalance, spread, mid

Uses a single Binance combined WebSocket with @depth20@100ms per symbol.
Default sampling interval is 0.25 seconds (latest snapshot at each tick).
At 7 assets × 0.25s: ~28 rows/sec, ~2.4M rows/day.

Usage:
    python scripts/collect_orderbook.py
    python scripts/collect_orderbook.py --assets BTC ETH SOL
    python scripts/collect_orderbook.py --interval 0.25
"""
import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import websockets
from rich.console import Console

from skeptic import config

console = Console()
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", datefmt="%M:%S")

ORDERBOOK_DIR = Path("data/orderbook")
DEPTH_LEVELS  = 20
FLUSH_INTERVAL = 15.0  # seconds between disk flushes

BINANCE_WS_COMBINED = "wss://stream.binance.us:9443/stream?streams={streams}"

# HYPE is not listed on Binance spot; all others are.
ASSET_TO_SYMBOL: dict[str, str] = {
    "BTC":  "btcusdt",
    "ETH":  "ethusdt",
    "SOL":  "solusdt",
    "DOGE": "dogeusdt",
    "XRP":  "xrpusdt",
    "BNB":  "bnbusdt",
}

SYMBOL_TO_ASSET = {v: k for k, v in ASSET_TO_SYMBOL.items()}

# CSV header: fixed columns + 20 bid levels + 20 ask levels + aggregates
_DEPTH_COLS = (
    [f"bid_px_{i}" for i in range(1, DEPTH_LEVELS + 1)]
    + [f"bid_sz_{i}" for i in range(1, DEPTH_LEVELS + 1)]
    + [f"ask_px_{i}" for i in range(1, DEPTH_LEVELS + 1)]
    + [f"ask_sz_{i}" for i in range(1, DEPTH_LEVELS + 1)]
)
CSV_HEADER = ["ts", "asset", "last_update_id"] + _DEPTH_COLS + [
    "bid_vol", "ask_vol", "imbalance", "spread", "mid",
]


def orderbook_csv_path() -> Path:
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    return ORDERBOOK_DIR / f"orderbook_{date_str}.csv"


def ensure_csv(path: Path) -> None:
    ORDERBOOK_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)


# ─────────────────────────────────────────────────────────────────────────────
# Depth cache — updated by the WebSocket task, read by the log task
# ─────────────────────────────────────────────────────────────────────────────

class DepthCache:
    """Stores the most recent depth snapshot (raw Binance message) per symbol."""

    __slots__ = ("_snaps",)

    def __init__(self) -> None:
        self._snaps: dict[str, dict] = {}

    def update(self, symbol: str, msg: dict) -> None:
        self._snaps[symbol] = msg

    def get(self, symbol: str) -> dict | None:
        return self._snaps.get(symbol)


# ─────────────────────────────────────────────────────────────────────────────
# Row builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_row(asset: str, msg: dict, ts: float) -> list:
    """Convert a raw Binance depth20 message to a flat CSV row.

    Prices and sizes are kept as the original strings from Binance (full precision,
    no reformatting). Derived aggregates are computed here.
    """
    bids = msg.get("bids") or []   # [[price_str, size_str], ...]
    asks = msg.get("asks") or []
    last_update_id = msg.get("lastUpdateId", "")

    def _pad(levels: list, n: int) -> tuple[list[str], list[str]]:
        prices = [levels[i][0] if i < len(levels) else "" for i in range(n)]
        sizes  = [levels[i][1] if i < len(levels) else "" for i in range(n)]
        return prices, sizes

    bid_px, bid_sz = _pad(bids, DEPTH_LEVELS)
    ask_px, ask_sz = _pad(asks, DEPTH_LEVELS)

    bid_vol = sum(float(b[1]) for b in bids)
    ask_vol = sum(float(a[1]) for a in asks)
    total   = bid_vol + ask_vol

    imbalance = f"{bid_vol / total:.6f}" if total > 0 else ""
    best_bid  = float(bids[0][0]) if bids else None
    best_ask  = float(asks[0][0]) if asks else None
    spread    = f"{best_ask - best_bid:.8g}" if best_bid is not None and best_ask is not None else ""
    mid       = f"{(best_bid + best_ask) / 2:.8g}" if best_bid is not None and best_ask is not None else ""

    frac = ts - int(ts)
    ts_str = str(int(ts)) if frac == 0 else f"{ts:.3f}".rstrip('0')
    row: list = [ts_str, asset, last_update_id]
    row += bid_px
    row += bid_sz
    row += ask_px
    row += ask_sz
    row += [f"{bid_vol:.8g}", f"{ask_vol:.8g}", imbalance, spread, mid]
    return row


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket producer — fills DepthCache from Binance
# ─────────────────────────────────────────────────────────────────────────────

async def binance_depth_stream(assets: list[str], cache: DepthCache) -> None:
    symbols = [ASSET_TO_SYMBOL[a] for a in assets if a in ASSET_TO_SYMBOL]
    if not symbols:
        console.print("[red]No valid Binance symbols for requested assets[/red]")
        return

    streams = "/".join(f"{sym}@depth20@100ms" for sym in symbols)
    url     = BINANCE_WS_COMBINED.format(streams=streams)
    backoff = 1.0

    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                backoff = 1.0
                console.print(
                    f"[green]Binance depth20@100ms connected[/green] "
                    f"({len(symbols)} symbols: {', '.join(s.upper() for s in symbols)})"
                )
                async for raw in ws:
                    envelope    = json.loads(raw)
                    stream_name = envelope.get("stream", "")
                    msg         = envelope.get("data", envelope)
                    sym         = stream_name.split("@")[0]
                    if sym in SYMBOL_TO_ASSET:
                        cache.update(sym, msg)
        except Exception as exc:
            logging.warning(
                "Binance depth stream disconnected: %s — reconnect in %.0fs", exc, backoff
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


# ─────────────────────────────────────────────────────────────────────────────
# Log loop — samples DepthCache at `interval` seconds and writes to CSV
# ─────────────────────────────────────────────────────────────────────────────

async def log_loop(assets: list[str], cache: DepthCache, interval: float) -> None:
    """Write the latest depth snapshot for each asset every `interval` seconds."""
    valid_assets = [a for a in assets if a in ASSET_TO_SYMBOL]
    if not valid_assets:
        console.print("[yellow]No assets with Binance depth data (HYPE not on Binance spot)[/yellow]")
        return

    pending:      list[list] = []
    current_path: Path | None = None
    f             = None
    writer        = None
    last_flush    = time.time()
    last_log      = time.time()
    rows_written  = 0

    try:
        while True:
            now  = time.time()
            ts   = round(now * 4) / 4   # snap to nearest 0.25s boundary
            path = orderbook_csv_path()
            ensure_csv(path)

            # Handle UTC midnight rollover — flush + reopen
            if path != current_path:
                if f is not None and pending:
                    writer.writerows(pending)  # type: ignore[union-attr]
                    f.flush()
                    pending.clear()
                if f is not None:
                    f.close()
                current_path = path
                f      = open(path, "a", newline="")
                writer = csv.writer(f)
                rows_written = 0
                console.print(f"[dim]Orderbook → {path.name}[/dim]")

            for asset in valid_assets:
                sym  = ASSET_TO_SYMBOL[asset]
                snap = cache.get(sym)
                if snap is None:
                    continue
                pending.append(_build_row(asset, snap, ts))
                rows_written += 1

            # Flush buffered rows to disk every FLUSH_INTERVAL seconds
            if now - last_flush >= FLUSH_INTERVAL and pending:
                writer.writerows(pending)  # type: ignore[union-attr]
                f.flush()                  # type: ignore[union-attr]
                pending.clear()
                last_flush = now

            # Heartbeat log every 60 seconds
            if now - last_log >= 60:
                n_live = sum(1 for a in valid_assets if cache.get(ASSET_TO_SYMBOL[a]) is not None)
                console.print(
                    f"  [dim]Orderbook: {rows_written} rows today | "
                    f"{n_live}/{len(valid_assets)} assets live[/dim]"
                )
                last_log = now

            # Sleep to the next exact interval boundary
            await asyncio.sleep(interval - (now % interval))

    finally:
        if f is not None:
            if pending and writer is not None:
                writer.writerows(pending)
            f.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def run(assets: list[str], interval: float) -> None:
    cache = DepthCache()

    console.print(
        f"[bold cyan]Orderbook Collector[/bold cyan] — "
        f"assets: {', '.join(assets)}, depth: {DEPTH_LEVELS} levels, interval: {interval}s"
    )
    console.print(f"Logging to [green]data/orderbook/[/green] (gitignored)\n")

    stream_task = asyncio.create_task(binance_depth_stream(assets, cache), name="depth-ws")
    log_task    = asyncio.create_task(log_loop(assets, cache, interval),   name="depth-log")

    try:
        await asyncio.gather(stream_task, log_task)
    finally:
        stream_task.cancel()
        log_task.cancel()
        await asyncio.gather(stream_task, log_task, return_exceptions=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Log Binance depth-20 orderbook snapshots every second")
    p.add_argument("--assets",   nargs="+", default=config.ASSETS,
                   help="Assets to collect (default: all from config)")
    p.add_argument("--interval", type=float, default=0.25,
                   help="Seconds between snapshot writes (default: 0.25)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(run(args.assets, args.interval))
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow]")
