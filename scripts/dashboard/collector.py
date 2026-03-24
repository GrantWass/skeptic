"""
LiveCollector — background price-collection thread for the dashboard.

Runs the same logic as collect_prices.py but inside a daemon thread so the
Streamlit UI stays responsive. Terminal logging (rich) is preserved exactly
as in the standalone script.

Shared state (written by background thread, read by UI):
    running, status, assets, window_ts, window_elapsed, window_remaining,
    rows_this_window, prices, history, error
"""
import asyncio
import csv
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import httpx
import streamlit as st
from rich.console import Console

from skeptic import config
from skeptic.clients import gamma as gamma_client
from skeptic.clients.ws import MarketChannel
from skeptic.utils.time import next_window_start, sleep_until

_PRICES_DIR = Path("data/prices")
_rich_console = Console()  # routes to terminal, not Streamlit


def _prices_csv_path() -> Path:
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    return _PRICES_DIR / f"prices_{date_str}.csv"


def _ensure_csv(path: Path) -> None:
    _PRICES_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(["ts", "window_ts", "asset", "up_price", "down_price"])


class LiveCollector:
    """
    Runs the price-collection loop in a daemon background thread so the
    Streamlit UI stays responsive.  Terminal logging (rich) is preserved
    exactly as in collect_prices.py — Streamlit only reads the shared state.
    """

    def __init__(self) -> None:
        self.running: bool = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Shared state — written by background thread, read by UI
        self.status: str = "Stopped"
        self.assets: list[str] = list(config.ASSETS)
        self.window_ts: int = 0
        self.window_elapsed: int = 0
        self.window_remaining: int = 0
        self.rows_this_window: int = 0
        self.prices: dict[str, dict] = {}   # asset → {"up": float|None, "down": float|None}
        self.history: list[tuple] = []       # (elapsed_s, asset, up, down) for current window
        self.error: str = ""

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, assets: list[str] | None = None) -> None:
        if self.running:
            return
        if assets:
            self.assets = assets
        self.running = True
        self.status = "Starting…"
        self.error = ""
        self._thread = threading.Thread(target=self._run, daemon=True, name="live-collector")
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        self.status = "Stopped"
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ── Background thread ─────────────────────────────────────────────────────

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._collect())
        except Exception as exc:
            self.error = str(exc)
            self.status = f"Error: {exc}"
        finally:
            self.running = False
            self._loop.close()

    async def _collect(self) -> None:
        market_ws = MarketChannel()
        ws_task = asyncio.create_task(market_ws.run(), name="market-ws")

        _rich_console.print(
            f"[bold cyan]Live Collector[/bold cyan] — assets: {', '.join(self.assets)}"
        )
        _rich_console.print("Logging to [green]data/prices/[/green]\n")

        try:
            async with httpx.AsyncClient(timeout=30.0) as http:
                window_ts = next_window_start()
                secs = window_ts - time.time()
                self.status = f"Waiting {secs:.0f}s for window {window_ts}…"
                _rich_console.print(
                    f"[dim]Window {window_ts} in {secs:.1f}s — fetching markets…[/dim]"
                )
                prefetch_task = asyncio.create_task(
                    self._fetch_and_subscribe(self.assets, window_ts, http, market_ws)
                )

                while self.running:
                    await sleep_until(float(window_ts))
                    markets = await prefetch_task

                    next_ts = window_ts + config.WINDOW_SECS
                    prefetch_task = asyncio.create_task(
                        self._fetch_and_subscribe(self.assets, next_ts, http, market_ws)
                    )

                    if not markets:
                        _rich_console.print(
                            f"[yellow]No markets for window {window_ts}, skipping[/yellow]"
                        )
                        window_ts = next_ts
                        continue

                    _rich_console.print(f"  [dim]Reconnecting WS for fresh snapshots…[/dim]")
                    await market_ws.reconnect()
                    await asyncio.sleep(2.0)

                    csv_path = _prices_csv_path()
                    _ensure_csv(csv_path)

                    _rich_console.print(
                        f"[green]▶ Window {window_ts}[/green] — "
                        f"logging {', '.join(m.asset for m in markets)} "
                        f"→ {csv_path.name}"
                    )

                    self.window_ts = window_ts
                    self.rows_this_window = 0
                    self.history = []
                    self.status = "Collecting"

                    n = await self._collect_window(markets, market_ws, window_ts, csv_path)
                    _rich_console.print(f"  [dim]✓ {n} rows written[/dim]")
                    window_ts = next_ts
        finally:
            ws_task.cancel()
            await asyncio.gather(ws_task, return_exceptions=True)

    async def _collect_window(
        self,
        markets: list,
        market_ws: MarketChannel,
        window_ts: int,
        csv_path: Path,
    ) -> int:
        window_end = window_ts + config.WINDOW_SECS
        rows_written = 0
        last_log = time.time()
        log_interval = 15

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)

            while self.running:
                now = time.time()
                if now >= window_end:
                    break

                self.window_elapsed = int(now - window_ts)
                self.window_remaining = int(window_end - now)
                ts = int(now)

                for market in markets:
                    up = market_ws.price_cache.get(market.up_token.token_id)
                    dn = market_ws.price_cache.get(market.down_token.token_id)
                    self.prices[market.asset] = {"up": up, "down": dn}

                    if up is not None or dn is not None:
                        writer.writerow([
                            ts, window_ts, market.asset,
                            f"{up:.4f}" if up is not None else "",
                            f"{dn:.4f}" if dn is not None else "",
                        ])
                        rows_written += 1
                        self.rows_this_window = rows_written
                        self.history.append((self.window_elapsed, market.asset, up, dn))

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
                    _rich_console.print(
                        f"  [dim]t+{elapsed:03d}s ({remaining}s left) | "
                        f"{' | '.join(price_lines)} | {rows_written} rows[/dim]"
                    )
                    last_log = now

                await asyncio.sleep(1.0 - (time.time() % 1.0))

            f.flush()
        return rows_written

    async def _fetch_and_subscribe(self, assets, window_ts, http, market_ws) -> list:
        _rich_console.print(f"  [dim]Fetching markets for window {window_ts}…[/dim]")
        tasks = [gamma_client.get_current_window_market(a, window_ts, http) for a in assets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        markets = []
        token_ids = []
        for asset, result in zip(assets, results):
            if isinstance(result, Exception) or result is None:
                _rich_console.print(
                    f"  [yellow]⚠ No market for {asset} window {window_ts}[/yellow]"
                )
                continue
            markets.append(result)
            token_ids += [result.up_token.token_id, result.down_token.token_id]

        if token_ids:
            await market_ws.subscribe(*token_ids)
            await asyncio.gather(
                *[self._seed_price(tid, http, market_ws) for tid in token_ids],
                return_exceptions=True,
            )
            _rich_console.print(
                f"  [dim]Subscribed + seeded {len(markets)} markets[/dim]"
            )
        return markets

    async def _seed_price(self, token_id: str, http: httpx.AsyncClient, market_ws: MarketChannel) -> None:
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


@st.cache_resource
def get_collector() -> LiveCollector:
    """Module-level singleton so the background thread survives Streamlit reruns."""
    return LiveCollector()
