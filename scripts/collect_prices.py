#!/usr/bin/env python3
"""
Price collector: logs UP/DOWN prices every second for each 5-minute window.

Writes to data/prices/prices_YYYYMMDD.csv with columns:
  ts, window_ts, asset, up_price, down_price

Run this continuously to build historical price data for research.
Usage:
    python scripts/collect_prices.py
    python scripts/collect_prices.py --assets BTC ETH SOL
    python scripts/collect_prices.py --interval 2   # log every 2 seconds
"""
import argparse
import asyncio
import csv
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from rich.console import Console

from skeptic import config
from skeptic.clients import gamma as gamma_client
from skeptic.clients.ws import MarketChannel
from skeptic.models.market import Market
from skeptic.utils.time import (
    current_window_start,
    next_window_start,
    seconds_until_next_window,
    sleep_until,
)

console = Console()
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

PRICES_DIR = Path("data/prices")


def prices_csv_path() -> Path:
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    return PRICES_DIR / f"prices_{date_str}.csv"


def ensure_csv(path: Path) -> bool:
    """Create CSV with header if it doesn't exist. Returns True if header was written."""
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ts", "window_ts", "asset", "up_price", "down_price"])
        return True
    return False


async def collect_window(
    markets: list[Market],
    market_ws: MarketChannel,
    window_ts: int,
    interval: float,
    csv_path: Path,
) -> None:
    """Log prices every `interval` seconds for the duration of a 5-minute window."""
    window_end = window_ts + config.WINDOW_SECS
    rows_written = 0

    last_log = time.time()
    log_interval = 15  # seconds

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        while True:
            now = time.time()
            if now >= window_end:
                break

            ts = int(now)
            for market in markets:
                up_price = market_ws.price_cache.get(market.up_token.token_id)
                down_price = market_ws.price_cache.get(market.down_token.token_id)

                if up_price is not None or down_price is not None:
                    writer.writerow([
                        ts,
                        window_ts,
                        market.asset,
                        f"{up_price:.4f}" if up_price is not None else "",
                        f"{down_price:.4f}" if down_price is not None else "",
                    ])
                    rows_written += 1

            # Status log every 15 seconds
            if now - last_log >= log_interval:
                elapsed = int(now - window_ts)
                remaining = int(window_end - now)
                price_lines = []
                for market in markets:
                    up = market_ws.price_cache.get(market.up_token.token_id)
                    dn = market_ws.price_cache.get(market.down_token.token_id)
                    up_str = f"{up:.3f}" if up is not None else "?"
                    dn_str = f"{dn:.3f}" if dn is not None else "?"
                    price_lines.append(f"{market.asset} ↑{up_str}/↓{dn_str}")
                console.print(
                    f"  [dim]t+{elapsed:03d}s ({remaining}s left) | "
                    f"{' | '.join(price_lines)} | {rows_written} rows[/dim]"
                )
                last_log = now

            # Sleep to the next exact second boundary
            await asyncio.sleep(interval - (time.time() % interval))

        f.flush()

    return rows_written


async def run(assets: list[str], interval: float) -> None:
    market_ws = MarketChannel()
    ws_task = asyncio.create_task(market_ws.run(), name="market-ws")

    console.print(f"[bold cyan]Price Collector[/bold cyan] — assets: {', '.join(assets)}, interval: {interval}s")
    console.print(f"Logging to [green]data/prices/[/green]\n")

    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            window_ts = next_window_start()

            # Pre-fetch the first window's markets while waiting
            secs = window_ts - time.time()
            console.print(f"[dim]Window {window_ts} in {secs:.1f}s — fetching markets…[/dim]")
            prefetch_task = asyncio.create_task(
                _fetch_and_subscribe(assets, window_ts, http, market_ws)
            )

            while True:
                await sleep_until(float(window_ts))
                markets = await prefetch_task

                # Immediately kick off pre-fetch for the NEXT window so
                # subscriptions are warm before it starts
                next_ts = window_ts + config.WINDOW_SECS
                prefetch_task = asyncio.create_task(
                    _fetch_and_subscribe(assets, next_ts, http, market_ws)
                )

                if not markets:
                    console.print(f"[yellow]No markets found for window {window_ts}, skipping[/yellow]")
                    window_ts = next_ts
                    continue

                # Reconnect so Polymarket sends fresh book snapshots for this
                # window's tokens (snapshots only arrive on a fresh connection,
                # not for mid-connection subscriptions).
                console.print(f"  [dim]Reconnecting WS for fresh snapshots…[/dim]")
                await market_ws.reconnect()
                await asyncio.sleep(2.0)  # wait for reconnect + snapshot delivery

                csv_path = prices_csv_path()
                ensure_csv(csv_path)

                console.print(
                    f"[green]▶ Window {window_ts}[/green] — "
                    f"logging {', '.join(m.asset for m in markets)} "
                    f"→ {csv_path.name}"
                )

                n = await collect_window(markets, market_ws, window_ts, interval, csv_path)
                console.print(f"  [dim]✓ {n} rows written[/dim]")
                window_ts = next_ts

    finally:
        ws_task.cancel()
        await asyncio.gather(ws_task, return_exceptions=True)


async def _fetch_and_subscribe(
    assets: list[str],
    window_ts: int,
    http: httpx.AsyncClient,
    market_ws: MarketChannel,
) -> list[Market]:
    console.print(f"  [dim]Fetching markets for window {window_ts}…[/dim]")
    tasks = [gamma_client.get_current_window_market(a, window_ts, http) for a in assets]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    markets = []
    token_ids = []
    for asset, result in zip(assets, results):
        if isinstance(result, Exception) or result is None:
            console.print(f"  [yellow]⚠ No market found for {asset} window {window_ts}[/yellow]")
            continue
        markets.append(result)
        token_ids += [result.up_token.token_id, result.down_token.token_id]

    if not token_ids:
        console.print(f"  [red]No markets found for window {window_ts}[/red]")
        return markets

    await market_ws.subscribe(*token_ids)

    # Seed price cache from REST /book — Polymarket doesn't push a snapshot
    # for tokens subscribed mid-connection, only on fresh connections.
    seed_tasks = [_seed_price(tid, http, market_ws) for tid in token_ids]
    await asyncio.gather(*seed_tasks, return_exceptions=True)

    console.print(f"  [dim]Subscribed + seeded {len(markets)} markets[/dim]")
    return markets


async def _seed_price(token_id: str, http: httpx.AsyncClient, market_ws: MarketChannel) -> None:
    """Fetch the current order book via REST and seed the price cache."""
    try:
        resp = await http.get(
            f"{config.CLOB_HOST}/book",
            params={"token_id": token_id},
            timeout=5.0,
        )
        data = resp.json()
        bids = data.get("bids") or []
        asks = data.get("asks") or []
        bid = float(bids[-1]["price"]) if bids else None
        ask = float(asks[-1]["price"]) if asks else None
        if bid and ask:
            market_ws.price_cache.update(token_id, (bid + ask) / 2)
        elif bid:
            market_ws.price_cache.update(token_id, bid)
        elif ask:
            market_ws.price_cache.update(token_id, ask)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Log Polymarket 5-min prices every second")
    p.add_argument("--assets", nargs="+", default=config.ASSETS)
    p.add_argument("--interval", type=float, default=1.0, help="Seconds between price snapshots")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(run(args.assets, args.interval))
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow]")
