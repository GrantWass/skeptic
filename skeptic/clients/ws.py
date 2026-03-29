"""
Async WebSocket client for Polymarket CLOB subscriptions.

Two channels:
  - User channel  (wss://.../ws/user)   — fill/trade events for our orders
  - Market channel (wss://.../ws/market) — live price snapshots per token

Each channel runs in its own asyncio task and reconnects with exponential backoff.
"""
import asyncio
import json
import logging
import time
from typing import Callable, Awaitable

import websockets
from websockets.exceptions import ConnectionClosed

from skeptic import config
from skeptic.models.order import Fill

logger = logging.getLogger(__name__)

# Type alias for fill callback
FillCallback = Callable[[Fill], Awaitable[None]]


class PriceCache:
    """
    Per-token order book maintained from book snapshots + price_change deltas.

    _bids / _asks: token_id → {price_float → size_float}
    Size == 0 means the level is removed from the book.
    """

    def __init__(self) -> None:
        self._bids: dict[str, dict[float, float]] = {}
        self._asks: dict[str, dict[float, float]] = {}

    def snapshot(self, token_id: str, bids: list[dict], asks: list[dict]) -> None:
        """Replace the full book from a book snapshot message."""
        self._bids[token_id] = {
            float(b["price"]): float(b["size"])
            for b in bids
            if float(b.get("size", 0)) > 0
        }
        self._asks[token_id] = {
            float(a["price"]): float(a["size"])
            for a in asks
            if float(a.get("size", 0)) > 0
        }

    def apply_change(self, token_id: str, side: str, price: str, size: float) -> None:
        """Apply a single price-level delta from a price_change message."""
        book = self._bids if side.upper() == "BUY" else self._asks
        levels = book.setdefault(token_id, {})
        fp = float(price)
        if size <= 0:
            levels.pop(fp, None)
        else:
            levels[fp] = size

    def get_bid(self, token_id: str) -> float | None:
        levels = self._bids.get(token_id)
        return max(levels) if levels else None

    def get_ask(self, token_id: str) -> float | None:
        levels = self._asks.get(token_id)
        return min(levels) if levels else None

    def get_mid(self, token_id: str) -> float | None:
        bid = self.get_bid(token_id)
        ask = self.get_ask(token_id)
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        return ask or bid


class UserChannel:
    """
    Subscribes to the Polymarket user WebSocket channel and emits Fill objects
    into an asyncio.Queue whenever a trade is matched.
    """

    def __init__(self, api_key: str, secret: str, passphrase: str) -> None:
        self._api_key = api_key
        self._secret = secret
        self._passphrase = passphrase
        self.fill_queue: asyncio.Queue[Fill] = asyncio.Queue()
        self._subscribed_markets: set[str] = set()
        self._ws = None
        self._running = False

    async def subscribe(self, condition_id: str) -> None:
        self._subscribed_markets.add(condition_id)
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({
                    "type": "subscribe",
                    "markets": [condition_id],
                }))
            except Exception:
                pass  # Will re-subscribe on next reconnect

    async def unsubscribe(self, condition_id: str) -> None:
        self._subscribed_markets.discard(condition_id)
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({
                    "type": "unsubscribe",
                    "markets": [condition_id],
                }))
            except Exception:
                pass

    async def run(self) -> None:
        self._running = True
        backoff = 1.0
        while self._running:
            try:
                async with websockets.connect(config.WS_USER_URL, ping_interval=20, ping_timeout=10, open_timeout=15) as ws:
                    self._ws = ws
                    backoff = 1.0
                    # Authenticate
                    await ws.send(json.dumps({
                        "type": "auth",
                        "apiKey": self._api_key,
                        "secret": self._secret,
                        "passphrase": self._passphrase,
                    }))
                    # Re-subscribe to any active markets
                    for cid in self._subscribed_markets:
                        await ws.send(json.dumps({"type": "subscribe", "markets": [cid]}))

                    async for raw in ws:
                        await self._handle(raw)
            except ConnectionClosed as e:
                logger.warning("User WS closed: %s — reconnecting in %.1fs", e, backoff)
            except Exception as e:
                logger.error("User WS error: %s — reconnecting in %.1fs", e, backoff)
            finally:
                self._ws = None

            if self._running:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _handle(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return

        # Polymarket sends messages as a list of events
        msgs = payload if isinstance(payload, list) else [payload]

        for msg in msgs:
            if not isinstance(msg, dict):
                continue

            event_type = msg.get("event_type") or msg.get("type", "")
            if event_type in ("trade", "TRADE"):
                status = msg.get("status", "")
                if status not in ("MATCHED", "matched", ""):
                    continue
                order_id = msg.get("taker_order_id") or msg.get("maker_order_id") or msg.get("order_id", "")
                try:
                    price = float(msg.get("price") or 0)
                    size = float(msg.get("size") or msg.get("size_matched") or 0)
                except (ValueError, TypeError):
                    continue
                if order_id and size > 0:
                    fill = Fill(
                        order_id=order_id,
                        outcome="",  # resolved by executor from order_id → outcome mapping
                        price=price,
                        size=size,
                        ts=time.time(),
                    )
                    await self.fill_queue.put(fill)
                    logger.info("Fill received: order=%s price=%.4f size=%.4f", order_id, price, size)

    def stop(self) -> None:
        self._running = False


class MarketChannel:
    """
    Subscribes to the Polymarket market WebSocket channel for live price data.
    Maintains a PriceCache with the latest mid-prices per token.
    """

    def __init__(self) -> None:
        self.price_cache = PriceCache()
        self._subscribed_tokens: set[str] = set()
        self._ws = None
        self._running = False

    def get_price(self, token_id: str) -> float | None:
        """Return book mid-price (for display)."""
        return self.price_cache.get_mid(token_id)

    def get_ask(self, token_id: str) -> float | None:
        """Return best ask (what you'd actually pay to buy)."""
        return self.price_cache.get_ask(token_id)

    async def subscribe(self, *token_ids: str) -> None:
        for tid in token_ids:
            self._subscribed_tokens.add(tid)
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({
                    "type": "subscribe",
                    "assets_ids": list(token_ids),
                }))
            except Exception:
                pass

    async def unsubscribe(self, *token_ids: str) -> None:
        for tid in token_ids:
            self._subscribed_tokens.discard(tid)

    async def reconnect(self) -> None:
        """Close the current connection so the run loop reconnects fresh.

        On a fresh connection Polymarket sends book snapshots for all
        subscribed tokens — the only reliable way to get real-time deltas
        for tokens that were subscribed mid-connection.
        """
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass

    async def run(self) -> None:
        self._running = True
        backoff = 1.0
        while self._running:
            try:
                async with websockets.connect(config.WS_MARKET_URL, ping_interval=20, ping_timeout=10, open_timeout=15) as ws:
                    self._ws = ws
                    backoff = 1.0
                    # Re-subscribe to all tracked tokens
                    if self._subscribed_tokens:
                        await ws.send(json.dumps({
                            "type": "subscribe",
                            "assets_ids": list(self._subscribed_tokens),
                        }))
                    async for raw in ws:
                        await self._handle(raw)
            except ConnectionClosed as e:
                logger.warning("Market WS closed: %s — reconnecting in %.1fs", e, backoff)
            except Exception as e:
                logger.error("Market WS error: %s — reconnecting in %.1fs", e, backoff)
            finally:
                self._ws = None

            if self._running:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _handle(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return

        # Polymarket sends messages as a list of events
        msgs = payload if isinstance(payload, list) else [payload]

        for msg in msgs:
            if not isinstance(msg, dict):
                continue

            event_type = msg.get("event_type") or msg.get("type", "")
            asset_id = msg.get("asset_id", "")

            if event_type == "book":
                if not asset_id:
                    continue
                bids = msg.get("bids") or []
                asks = msg.get("asks") or []
                self.price_cache.snapshot(asset_id, bids, asks)

            elif event_type == "price_change":
                if not asset_id:
                    continue
                for change in msg.get("changes") or []:
                    price = change.get("price", "")
                    side  = change.get("side", "")
                    try:
                        size = float(change.get("size", 0))
                    except (ValueError, TypeError):
                        size = 0.0
                    if price and side:
                        self.price_cache.apply_change(asset_id, side, price, size)

            # last_trade_price is ignored: arrives independently for each token at
            # different times, causing UP+DOWN sums far from 1.0.

    def stop(self) -> None:
        self._running = False
