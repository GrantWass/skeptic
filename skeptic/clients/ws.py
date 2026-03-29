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
from typing import Callable, Awaitable, NamedTuple

import websockets
from websockets.exceptions import ConnectionClosed

from skeptic import config
from skeptic.models.order import Fill

logger = logging.getLogger(__name__)

# Type alias for fill callback
FillCallback = Callable[[Fill], Awaitable[None]]


class BookData(NamedTuple):
    bid: float | None
    ask: float | None
    bid_volume: float   # sum of all bid sizes in the book
    ask_volume: float   # sum of all ask sizes in the book


class PriceCache:
    """Thread-safe (asyncio-safe) price cache keyed by token_id."""

    def __init__(self) -> None:
        self._prices: dict[str, float] = {}
        self._books: dict[str, BookData] = {}

    def update(self, token_id: str, price: float) -> None:
        self._prices[token_id] = price

    def update_book(self, token_id: str, book: BookData) -> None:
        self._books[token_id] = book

    def get(self, token_id: str) -> float | None:
        return self._prices.get(token_id)

    def get_book(self, token_id: str) -> BookData | None:
        return self._books.get(token_id)

    def midpoint(self, token_id: str, bid: float | None, ask: float | None) -> float | None:
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        return bid or ask or self._prices.get(token_id)


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

            if event_type in ("book", "price_change", "last_trade_price"):
                asset_id = msg.get("asset_id", "")
                if not asset_id:
                    continue

                def to_float(v) -> float | None:
                    try:
                        return float(v) if v not in (None, "", "0") else None
                    except (ValueError, TypeError):
                        return None

                # "book" messages send full bids/asks arrays sorted ascending by price.
                # Best bid = highest bid = last element; best ask = lowest ask = last element.
                bids = msg.get("bids") or []
                asks = msg.get("asks") or []
                bid = to_float(bids[-1]["price"]) if bids else to_float(msg.get("best_bid"))
                ask = to_float(asks[-1]["price"]) if asks else to_float(msg.get("best_ask"))
                price = to_float(msg.get("price") or msg.get("last_trade_price"))

                mid = self.price_cache.midpoint(asset_id, bid, ask)
                if mid is not None:
                    self.price_cache.update(asset_id, mid)
                elif price is not None:
                    self.price_cache.update(asset_id, price)

                # Store book depth for spread / imbalance whenever full book arrives
                if bids or asks:
                    try:
                        bid_vol = sum(float(b.get("size", 0)) for b in bids)
                        ask_vol = sum(float(a.get("size", 0)) for a in asks)
                        self.price_cache.update_book(asset_id, BookData(bid, ask, bid_vol, ask_vol))
                    except Exception:
                        pass

    def stop(self) -> None:
        self._running = False
