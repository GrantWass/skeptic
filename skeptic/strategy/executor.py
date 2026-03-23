"""
SessionExecutor: core strategy logic for a single 5-minute window.

Flow:
  1. Snapshot prices at T=0
  2. Place limit BUY orders on UP and DOWN at BUY_PRICE
  3. Monitor fills via WebSocket for MONITOR_SECS (60s)
     - Any fill (including partial) triggers the cascade
  4a. Fill → cancel the other order → place SELL at SELL_PRICE
  4b. Timeout → cancel both orders
  5. Snapshot prices at T=60 (minute-1 mark)
  6. Log everything to the DB
"""
import asyncio
import logging
import time

from py_clob_client.client import ClobClient

from skeptic import config
from skeptic.clients import clob as clob_client
from skeptic.clients.ws import UserChannel, MarketChannel
from skeptic.models.market import Market
from skeptic.models.order import Fill, Order, OrderPair
from skeptic.models.session import TradingSession
from skeptic.storage import logger as session_logger

log = logging.getLogger(__name__)


class SessionExecutor:
    def __init__(
        self,
        client: ClobClient,
        user_ws: UserChannel,
        market_ws: MarketChannel,
        dry_run: bool = False,
    ) -> None:
        self._client = client
        self._user_ws = user_ws
        self._market_ws = market_ws
        self._dry_run = dry_run

    async def run(self, market: Market, capital: float) -> TradingSession:
        # --- Snapshot prices at T=0 ---
        up_open = self._market_ws.price_cache.get(market.up_token.token_id)
        down_open = self._market_ws.price_cache.get(market.down_token.token_id)

        shares = (capital * config.POSITION_SIZE_PCT) / config.BUY_PRICE

        session = TradingSession(
            asset=market.asset,
            condition_id=market.condition_id,
            window_start_ts=market.start_ts,
            buy_price_used=config.BUY_PRICE,
            sell_price_used=config.SELL_PRICE,
            capital_deployed=capital * config.POSITION_SIZE_PCT,
            up_price_open=up_open,
            down_price_open=down_open,
        )
        session_logger.insert_session(session)

        if up_open is not None:
            session_logger.insert_price_snapshot(
                session.session_id, market.up_token.token_id, "UP", up_open, minute_mark=0
            )
        if down_open is not None:
            session_logger.insert_price_snapshot(
                session.session_id, market.down_token.token_id, "DOWN", down_open, minute_mark=0
            )

        # --- Subscribe to user channel for this market ---
        await self._user_ws.subscribe(market.condition_id)

        try:
            # --- Place orders ---
            up_order, down_order = await self._place_orders(market, shares, session)
            pair = OrderPair(up_order=up_order, down_order=down_order, placed_at=time.time())

            # --- Build order_id → outcome lookup ---
            id_to_outcome = {
                up_order.order_id: "UP",
                down_order.order_id: "DOWN",
            }

            # --- Start minute-1 snapshot task ---
            m1_task = asyncio.create_task(
                self._capture_m1_snapshot(market, session.session_id)
            )

            # --- Monitor fills for MONITOR_SECS ---
            fill = await self._monitor_fills(pair, id_to_outcome)

            if fill is not None:
                # Cascade: cancel opposite, place sell
                await self._handle_fill(fill, pair, id_to_outcome, market, shares, session)
            else:
                # Timeout — cancel both
                log.info("[%s] No fill in %ds — cancelling both orders", market.asset, config.MONITOR_SECS)
                await self._cancel_pair(pair)

            # Ensure minute-1 snapshot completes
            try:
                await asyncio.wait_for(m1_task, timeout=5.0)
            except asyncio.TimeoutError:
                pass

        finally:
            await self._user_ws.unsubscribe(market.condition_id)

        return session

    # -----------------------------------------------------------------------

    async def _place_orders(
        self, market: Market, shares: float, session: TradingSession
    ) -> tuple[Order, Order]:
        if self._dry_run:
            log.info("[DRY RUN][%s] Would place BUY UP @ %.2f x %.4f", market.asset, config.BUY_PRICE, shares)
            log.info("[DRY RUN][%s] Would place BUY DOWN @ %.2f x %.4f", market.asset, config.BUY_PRICE, shares)
            fake_up = Order("dry-up-" + session.session_id[:8], market.up_token.token_id, "UP", "BUY", config.BUY_PRICE, shares)
            fake_down = Order("dry-dn-" + session.session_id[:8], market.down_token.token_id, "DOWN", "BUY", config.BUY_PRICE, shares)
            return fake_up, fake_down

        up_order = clob_client.place_limit_order(
            self._client, market.up_token.token_id, "UP", "BUY", config.BUY_PRICE, shares
        )
        down_order = clob_client.place_limit_order(
            self._client, market.down_token.token_id, "DOWN", "BUY", config.BUY_PRICE, shares
        )
        session_logger.insert_order(up_order, session.session_id)
        session_logger.insert_order(down_order, session.session_id)
        return up_order, down_order

    async def _monitor_fills(
        self, pair: OrderPair, id_to_outcome: dict[str, str]
    ) -> Fill | None:
        """
        Wait for any fill on either order within MONITOR_SECS.
        Also polls REST at T=55s as a fallback.
        Returns the first Fill, or None on timeout.
        """
        deadline = asyncio.get_event_loop().time() + config.MONITOR_SECS
        poll_task = asyncio.create_task(self._rest_poll_fallback(pair, id_to_outcome, deadline - 5))

        try:
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    return None
                try:
                    raw_fill: Fill = await asyncio.wait_for(
                        self._user_ws.fill_queue.get(),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    return None

                if raw_fill.order_id in id_to_outcome:
                    raw_fill.outcome = id_to_outcome[raw_fill.order_id]
                    return raw_fill
                # Fill is for a different order — put it back and keep waiting
                await self._user_ws.fill_queue.put(raw_fill)
                await asyncio.sleep(0.01)
        finally:
            poll_task.cancel()

    async def _rest_poll_fallback(
        self, pair: OrderPair, id_to_outcome: dict[str, str], fire_at: float
    ) -> None:
        """At T≈55s, poll REST to catch fills the WS may have missed."""
        wait = fire_at - asyncio.get_event_loop().time()
        if wait > 0:
            await asyncio.sleep(wait)
        if self._dry_run:
            return
        try:
            open_ids = {
                o["id"] for o in clob_client.get_open_orders(self._client)
            }
            for order_id, outcome in id_to_outcome.items():
                if order_id not in open_ids:
                    # Order is no longer open — treat as filled
                    fill = Fill(
                        order_id=order_id,
                        outcome=outcome,
                        price=config.BUY_PRICE,
                        size=pair.up_order.size,
                        ts=time.time(),
                    )
                    await self._user_ws.fill_queue.put(fill)
                    log.info("REST fallback detected fill for order %s", order_id)
        except Exception as e:
            log.warning("REST fallback poll failed: %s", e)

    async def _handle_fill(
        self,
        fill: Fill,
        pair: OrderPair,
        id_to_outcome: dict[str, str],
        market: Market,
        shares: float,
        session: TradingSession,
    ) -> None:
        """Cancel the opposite order and place a sell."""
        log.info("[%s] Fill on %s — size=%.4f", market.asset, fill.outcome, fill.size)
        session_logger.update_fill(session.session_id, fill.outcome)

        # Cancel the unfilled side
        if fill.outcome == "UP":
            opposite = pair.down_order
            sell_token = market.up_token
        else:
            opposite = pair.up_order
            sell_token = market.down_token

        if not self._dry_run:
            if clob_client.cancel_order(self._client, opposite.order_id):
                session_logger.update_order_status(opposite.order_id, "CANCELLED")

        # Place sell order
        if self._dry_run:
            log.info("[DRY RUN][%s] Would place SELL %s @ %.2f x %.4f", market.asset, fill.outcome, config.SELL_PRICE, fill.size)
            session_logger.update_sell_placed(session.session_id)
            return

        try:
            sell_order = clob_client.place_limit_order(
                self._client, sell_token.token_id, fill.outcome, "SELL", config.SELL_PRICE, fill.size
            )
            session_logger.insert_order(sell_order, session.session_id)
            session_logger.update_sell_placed(session.session_id)
            log.info("[%s] Sell order placed %s", market.asset, sell_order.order_id)

            # Note: sell monitoring happens asynchronously in the background.
            # The runner monitors for sell fills via the user WS after this method returns.
            # PnL is recorded when the sell fills or at resolution.
        except Exception as e:
            log.error("[%s] Failed to place sell: %s", market.asset, e)

    async def _cancel_pair(self, pair: OrderPair) -> None:
        if self._dry_run:
            return
        clob_client.cancel_orders(
            self._client, [pair.up_order.order_id, pair.down_order.order_id]
        )
        session_logger.update_order_status(pair.up_order.order_id, "CANCELLED")
        session_logger.update_order_status(pair.down_order.order_id, "CANCELLED")

    async def _capture_m1_snapshot(self, market: Market, session_id: str) -> None:
        """Capture price snapshot at the 1-minute mark."""
        await asyncio.sleep(config.MONITOR_SECS)
        up_m1 = self._market_ws.price_cache.get(market.up_token.token_id)
        down_m1 = self._market_ws.price_cache.get(market.down_token.token_id)

        if up_m1 is not None:
            session_logger.insert_price_snapshot(
                session_id, market.up_token.token_id, "UP", up_m1, minute_mark=1
            )
        if down_m1 is not None:
            session_logger.insert_price_snapshot(
                session_id, market.down_token.token_id, "DOWN", down_m1, minute_mark=1
            )

        if up_m1 is not None or down_m1 is not None:
            session_logger.update_m1_prices(
                session_id,
                up_m1 if up_m1 is not None else 0.0,
                down_m1 if down_m1 is not None else 0.0,
            )
