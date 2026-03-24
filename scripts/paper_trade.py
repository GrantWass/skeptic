#!/usr/bin/env python3
"""
Live paper trading simulation — tracks one asset with real-time prices.

Watches the live order book for a chosen asset and simulates the threshold
strategy window by window, tracking compounding P&L on a $500 base capital.

Run alongside collect_prices.py — this script only monitors one coin for
paper trading and does not interfere with data collection.

Usage:
    python scripts/paper_trade.py
    python scripts/paper_trade.py --asset BTC --buy 0.36 --sell 0.56
    python scripts/paper_trade.py --asset ETH --buy 0.34 --sell 0.58 --capital 1000
"""
import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from rich import box
from rich.console import Console
from rich.table import Table

from skeptic import config
from skeptic.clients import gamma as gamma_client
from skeptic.clients.ws import MarketChannel
from skeptic.utils.time import next_window_start, sleep_until

console = Console()

CAPITAL_DEFAULT = 500.0
POSITION_SIZE_PCT = 0.05        # 5% per window (matches config)
RESOLUTION_POLL_SECS = 25       # seconds after window end to wait for resolution price


def prompt_setup() -> tuple[str, float, float]:
    console.print(f"\n[bold cyan]Paper Trade Setup[/bold cyan]")
    console.print(f"Available assets: {', '.join(config.ASSETS)}\n")

    asset = input("Asset (e.g. BTC): ").strip().upper()
    if asset not in config.ASSETS:
        console.print(f"[yellow]Unknown asset '{asset}', defaulting to BTC.[/yellow]")
        asset = "BTC"

    try:
        buy = float(input("Buy threshold (e.g. 0.36): ").strip())
        sell = float(input("Sell threshold (e.g. 0.56): ").strip())
    except ValueError:
        console.print("[yellow]Invalid thresholds, using 0.36 / 0.56.[/yellow]")
        buy, sell = 0.36, 0.56

    return asset, buy, sell


async def _seed_price(token_id: str, http: httpx.AsyncClient, market_ws: MarketChannel) -> None:
    """Seed the price cache from the REST order book."""
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


async def run_window(
    asset: str,
    window_ts: int,
    market_ws: MarketChannel,
    up_token_id: str,
    down_token_id: str,
    buy: float,
    sell: float,
    capital: float,
    position_size_pct: float,
) -> dict:
    """
    Simulate one 5-minute window. Returns a result dict.

    Fill logic:
      - Watch UP and DOWN prices every 0.2s during the first 60 seconds
      - The FIRST side to touch buy_threshold is the fill
      - After fill: watch for sell hit until window end
      - If sell not hit: wait up to RESOLUTION_POLL_SECS for price to resolve to ~0 or ~1
    """
    window_end = window_ts + config.WINDOW_SECS
    m1_cutoff = window_ts + config.MONITOR_SECS

    position_usdc = capital * position_size_pct
    shares = position_usdc / buy

    filled = False
    filled_side = None      # "UP" or "DOWN"
    sell_hit = False
    resolution = None       # True = win, False = loss, None = unclear
    pnl = 0.0
    outcome_str = "no fill"

    ts_str = datetime.fromtimestamp(window_ts, tz=timezone.utc).strftime("%H:%M UTC")
    console.print(
        f"\n[bold green]▶ Window {ts_str}[/bold green]  "
        f"{asset} | buy≤{buy} | sell≥{sell} | "
        f"capital=[bold]${capital:,.2f}[/bold] | "
        f"deploying=${position_usdc:.2f}"
    )

    # ── Phase 1: watch for first fill in minute 1 ──────────────────────────
    while True:
        now = time.time()
        if now > m1_cutoff or now >= window_end:
            break

        up_price = market_ws.price_cache.get(up_token_id)
        down_price = market_ws.price_cache.get(down_token_id)

        up_hit = up_price is not None and up_price <= buy
        down_hit = down_price is not None and down_price <= buy

        if up_hit or down_hit:
            filled = True
            # First/cheapest hit wins; UP is tiebreaker
            if up_hit and down_hit:
                filled_side = "UP" if (up_price or 1.0) <= (down_price or 1.0) else "DOWN"
            elif up_hit:
                filled_side = "UP"
            else:
                filled_side = "DOWN"

            elapsed = int(now - window_ts)
            console.print(
                f"  [yellow bold]★ FILLED {filled_side}[/yellow bold] "
                f"@ {buy} at t+{elapsed}s | {shares:.2f} shares | ${position_usdc:.2f} deployed"
            )
            break

        await asyncio.sleep(0.2)

    if not filled:
        console.print(f"  [dim]— no fill in minute 1[/dim]")
        return {"window_ts": window_ts, "filled": False, "filled_side": None,
                "sell_hit": False, "resolution": None, "outcome": "no fill", "pnl": 0.0}

    # ── Phase 2: watch for sell hit until window end ───────────────────────
    filled_token_id = up_token_id if filled_side == "UP" else down_token_id

    while time.time() < window_end:
        price = market_ws.price_cache.get(filled_token_id)
        if price is not None and price >= sell:
            sell_hit = True
            pnl = shares * (sell - buy)
            outcome_str = f"sell hit @ {sell}"
            console.print(
                f"  [bold green]✓ SELL HIT[/bold green] "
                f"@ {sell} | PnL=[bold green]${pnl:+.2f}[/bold green]"
            )
            break
        await asyncio.sleep(0.2)

    if sell_hit:
        return {"window_ts": window_ts, "filled": True, "filled_side": filled_side,
                "sell_hit": True, "resolution": None, "outcome": outcome_str, "pnl": pnl}

    # ── Phase 3: wait for resolution (up to RESOLUTION_POLL_SECS after window end) ──
    # Cap deadline so we don't bleed into the next window's setup time
    poll_deadline = window_end + RESOLUTION_POLL_SECS
    console.print(f"  [dim]Sell not hit — waiting for resolution…[/dim]")

    while time.time() < poll_deadline:
        price = market_ws.price_cache.get(filled_token_id)
        if price is not None:
            if price >= 0.9:
                resolution = True
                pnl = shares * (1.0 - buy)
                outcome_str = "resolution win"
                console.print(
                    f"  [green]✓ RESOLUTION WIN[/green] "
                    f"(price={price:.3f}) | PnL=[bold green]${pnl:+.2f}[/bold green]"
                )
                break
            elif price <= 0.1:
                resolution = False
                pnl = -position_usdc
                outcome_str = "resolution loss"
                console.print(
                    f"  [red]✗ RESOLUTION LOSS[/red] "
                    f"(price={price:.3f}) | PnL=[bold red]${pnl:+.2f}[/bold red]"
                )
                break
        await asyncio.sleep(1.0)

    if resolution is None:
        outcome_str = "resolution unclear"
        console.print(f"  [yellow]? Resolution unclear — counting as scratch[/yellow]")

    return {"window_ts": window_ts, "filled": True, "filled_side": filled_side,
            "sell_hit": False, "resolution": resolution, "outcome": outcome_str, "pnl": pnl}


def print_summary(results: list[dict], capital: float, initial_capital: float) -> None:
    table = Table(box=box.SIMPLE_HEAVY, show_header=True)
    table.add_column("Window", style="dim", width=8)
    table.add_column("Side", width=5)
    table.add_column("Outcome", width=20)
    table.add_column("PnL", justify="right", width=10)
    table.add_column("Capital", justify="right", width=12)

    running = initial_capital
    for r in results:
        running += r["pnl"]
        ts_str = datetime.fromtimestamp(r["window_ts"], tz=timezone.utc).strftime("%H:%M")
        if r["filled"]:
            pnl_str = f"${r['pnl']:+.2f}"
            color = "green" if r["pnl"] > 0 else ("red" if r["pnl"] < 0 else "dim")
        else:
            pnl_str = "—"
            color = "dim"
        table.add_row(
            ts_str,
            r["filled_side"] or "—",
            r["outcome"],
            f"[{color}]{pnl_str}[/{color}]",
            f"${running:,.2f}",
        )

    console.print(table)
    total_pnl = capital - initial_capital
    color = "green" if total_pnl >= 0 else "red"
    fills = sum(1 for r in results if r["filled"])
    console.print(
        f"[bold]Sessions: {len(results)} | Fills: {fills} | "
        f"Total PnL: [{color}]${total_pnl:+.2f}[/{color}] | "
        f"Capital: ${capital:,.2f}[/bold]"
    )


async def run(asset: str, buy: float, sell: float, capital: float, position_size_pct: float) -> None:
    market_ws = MarketChannel()
    ws_task = asyncio.create_task(market_ws.run(), name="market-ws")

    initial_capital = capital
    results: list[dict] = []

    console.print(f"\n[bold cyan]Paper Trading {asset}[/bold cyan]")
    console.print(f"Buy ≤ {buy} | Sell ≥ {sell} | Capital: ${capital:,.2f} | Position: {position_size_pct:.0%}/window")
    console.print(f"[dim]Run collect_prices.py separately to keep recording all 7 coins.[/dim]\n")

    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            window_ts = next_window_start()
            secs_until = window_ts - time.time()
            console.print(f"[dim]Next window in {secs_until:.0f}s — fetching market…[/dim]")

            # Pre-fetch market while waiting for the first window
            market = await gamma_client.get_current_window_market(
                asset, window_ts, http, retries=12, retry_delay=5.0
            )

            while True:
                if market is None:
                    console.print(f"[yellow]No market found for {asset} @ {window_ts}, skipping.[/yellow]")
                    window_ts += config.WINDOW_SECS
                    await sleep_until(float(window_ts) - 10)
                    market = await gamma_client.get_current_window_market(
                        asset, window_ts, http, retries=12, retry_delay=5.0
                    )
                    continue

                # Subscribe and get fresh WS snapshots
                token_ids = [market.up_token.token_id, market.down_token.token_id]
                await market_ws.subscribe(*token_ids)
                await market_ws.reconnect()
                await asyncio.sleep(2.0)
                await asyncio.gather(
                    *[_seed_price(tid, http, market_ws) for tid in token_ids],
                    return_exceptions=True,
                )

                await sleep_until(float(window_ts))

                result = await run_window(
                    asset=asset,
                    window_ts=window_ts,
                    market_ws=market_ws,
                    up_token_id=market.up_token.token_id,
                    down_token_id=market.down_token.token_id,
                    buy=buy,
                    sell=sell,
                    capital=capital,
                    position_size_pct=position_size_pct,
                )
                capital += result["pnl"]
                results.append(result)

                print_summary(results, capital, initial_capital)

                # Prefetch next window's market while we have time between windows
                next_ts = window_ts + config.WINDOW_SECS
                market = await gamma_client.get_current_window_market(
                    asset, next_ts, http, retries=12, retry_delay=5.0
                )
                window_ts = next_ts

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow]")
        if results:
            print_summary(results, capital, initial_capital)
    finally:
        ws_task.cancel()
        await asyncio.gather(ws_task, return_exceptions=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live paper trading simulation for one asset")
    p.add_argument("--asset", help="Asset to trade (e.g. BTC, ETH, SOL)")
    p.add_argument("--buy", type=float, help="Buy threshold (e.g. 0.36)")
    p.add_argument("--sell", type=float, help="Sell threshold (e.g. 0.56)")
    p.add_argument("--capital", type=float, default=CAPITAL_DEFAULT,
                   help=f"Starting capital in USD (default: {CAPITAL_DEFAULT})")
    p.add_argument("--position-size", type=float, default=POSITION_SIZE_PCT,
                   help=f"Position size as fraction of capital (default: {POSITION_SIZE_PCT})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.asset and args.buy and args.sell:
        asset = args.asset.upper()
        buy = args.buy
        sell = args.sell
    else:
        asset, buy, sell = prompt_setup()

    try:
        asyncio.run(run(asset, buy, sell, args.capital, args.position_size))
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow]")
