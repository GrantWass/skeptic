"""
Skeptic Trading Dashboard
Run: streamlit run dashboard.py

Controls: Asset, Buy Threshold, Sell Threshold, Fill Window
Displays:  Live price chart, fill/sell markers, P&L, trade log
Position:  5% of wallet balance per bet
"""

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from skeptic import config
from skeptic.clients import clob as clob_client
from skeptic.clients import gamma as gamma_client
from skeptic.clients.ws import UserChannel, MarketChannel
from skeptic.models.market import Market
from skeptic.models.order import Fill
from skeptic.utils.time import (
    next_window_start,
    seconds_until_next_window,
    sleep_until,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

ASSETS = ["BTC", "ETH", "SOL", "DOGE", "XRP", "BNB", "HYPE"]
POSITION_SIZE = 0.05          # 5% of balance per bet
PAPER_START_BALANCE = 500.0   # simulated balance for dry-run mode


# ─────────────────────────────── Data classes ────────────────────────────────

@dataclass
class PricePoint:
    ts: float
    up: Optional[float]
    down: Optional[float]
    window_ts: int


@dataclass
class TradeRecord:
    trade_id: str
    asset: str
    outcome: str        # "UP" or "DOWN"
    buy_price: float
    buy_size: float     # shares
    buy_time: float     # unix ts
    cost: float         # dollars = buy_price * buy_size
    window_ts: int
    sell_price: Optional[float] = None
    sell_time: Optional[float] = None
    pnl: Optional[float] = None
    # OPEN | SELL_PLACED | SOLD | TIMEOUT
    status: str = "OPEN"


# ──────────────────────────── Thread-safe state ──────────────────────────────

class TradingState:
    """Shared state between the async trading engine and Streamlit UI."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.running: bool = False
        self.balance: float = 0.0
        self.balance_start: float = 0.0
        self.prices: list[PricePoint] = []
        self.trades: list[TradeRecord] = []
        self.status_msg: str = "Idle"
        self.error: Optional[str] = None
        self.total_pnl: float = 0.0
        self.wins: int = 0
        self.losses: int = 0
        self.current_asset: Optional[str] = None
        self.current_window_ts: int = 0
        self.debug: dict = {}

    # ── writers (called from engine thread) ─────────────────────────────────

    def add_price(self, ts: float, up: Optional[float], down: Optional[float], window_ts: int) -> None:
        with self._lock:
            self.prices.append(PricePoint(ts=ts, up=up, down=down, window_ts=window_ts))
            if len(self.prices) > 3100:           # 0.1s × 3100 ≈ 310s — full window
                self.prices = self.prices[-3100:]

    def add_trade(self, trade: TradeRecord) -> None:
        with self._lock:
            self.trades.append(trade)

    def update_trade(self, trade_id: str, **kwargs) -> None:
        with self._lock:
            for t in self.trades:
                if t.trade_id == trade_id:
                    for k, v in kwargs.items():
                        setattr(t, k, v)
                    if "pnl" in kwargs and kwargs["pnl"] is not None:
                        self.total_pnl += kwargs["pnl"]
                        if kwargs["pnl"] >= 0:
                            self.wins += 1
                        else:
                            self.losses += 1
                    break

    def set_balance(self, b: float) -> None:
        with self._lock:
            self.balance = b

    def set_status(self, msg: str) -> None:
        with self._lock:
            self.status_msg = msg

    def set_debug(self, **kwargs) -> None:
        with self._lock:
            self.debug.update(kwargs)

    def set_error(self, msg: str) -> None:
        with self._lock:
            self.error = msg
            self.running = False

    def stop(self) -> None:
        with self._lock:
            self.running = False
            self.status_msg = "Stopped"

    # ── reader (called from Streamlit thread) ────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "balance": self.balance,
                "balance_start": self.balance_start,
                "prices": list(self.prices),
                "trades": list(self.trades),
                "status_msg": self.status_msg,
                "error": self.error,
                "total_pnl": self.total_pnl,
                "wins": self.wins,
                "losses": self.losses,
                "current_asset": self.current_asset,
                "current_window_ts": self.current_window_ts,
                "debug": dict(self.debug),
            }


# ────────────────────────────── Trading engine ───────────────────────────────

class TradingEngine:
    """
    Runs the Polymarket 5-min UP/DOWN strategy in a background daemon thread.
    Communicates with the UI only through TradingState.
    """

    def __init__(
        self,
        state: TradingState,
        asset: str,
        buy_threshold: float,
        sell_threshold: float,
        fill_window: int,
        dry_run: bool = True,
    ) -> None:
        self.state = state
        self.asset = asset
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.fill_window = fill_window
        self.dry_run = dry_run

        # Set by _execute_window so the price recorder knows which tokens to watch
        self._up_token_id: Optional[str] = None
        self._down_token_id: Optional[str] = None
        self._current_window_ts: int = 0

    # ── lifecycle ───────────────────────────────────────────────────────────

    def start(self) -> None:
        with self.state._lock:
            self.state.running = True
            self.state.current_asset = self.asset
            self.state.error = None
            self.state.status_msg = "Starting…"
        threading.Thread(target=self._run_in_thread, daemon=True).start()

    def stop(self) -> None:
        self.state.stop()

    def _run_in_thread(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._main())
        except Exception as exc:
            logger.exception("Engine crashed")
            self.state.set_error(str(exc))
        finally:
            loop.close()
            with self.state._lock:
                if self.state.error is None:
                    self.state.running = False
                    self.state.status_msg = "Stopped"

    # ── main async loop ──────────────────────────────────────────────────────

    async def _main(self) -> None:
        self.state.set_status("Connecting…")

        # ── Credential check ─────────────────────────────────────────────────
        pk_set = bool(getattr(config, "PRIVATE_KEY", ""))
        wa_set = bool(getattr(config, "WALLET_ADDRESS", ""))
        self.state.set_debug(
            dry_run=self.dry_run,
            private_key_set=pk_set,
            wallet_address_set=wa_set,
            clob_api_key_set=bool(getattr(config, "CLOB_API_KEY", "")),
            market_ws_connected=False,
            user_ws_connected=False,
            up_token_id=None,
            down_token_id=None,
            price_cache_up=None,
            price_cache_down=None,
            prices_recorded=0,
            last_market_slug=None,
            last_market_found=None,
        )

        # ── CLOB client (skipped in dry-run) ────────────────────────────────
        clob = None
        if not self.dry_run:
            try:
                clob = clob_client.build_client()
                balance = clob_client.get_usdc_balance(clob)
                self.state.set_balance(balance)
                self.state.set_debug(clob_connected=True, balance_fetched=balance)
                with self.state._lock:
                    self.state.balance_start = balance
            except Exception as exc:
                self.state.set_debug(clob_connected=False, clob_error=str(exc))
                self.state.set_error(f"CLOB connect failed: {exc}")
                return
        else:
            self.state.set_debug(clob_connected="skipped (dry run)")
            with self.state._lock:
                self.state.balance = PAPER_START_BALANCE
                self.state.balance_start = PAPER_START_BALANCE

        # ── WebSocket channels ───────────────────────────────────────────────
        user_ws: Optional[UserChannel] = None
        if not self.dry_run:
            user_ws = UserChannel(
                api_key=config.CLOB_API_KEY,
                secret=config.CLOB_SECRET,
                passphrase=config.CLOB_PASSPHRASE,
            )

        market_ws = MarketChannel()

        bg: list[asyncio.Task] = [asyncio.create_task(market_ws.run())]
        if user_ws:
            bg.append(asyncio.create_task(user_ws.run()))
        bg.append(asyncio.create_task(self._price_recorder(market_ws)))

        # Give WebSockets a moment to connect
        await asyncio.sleep(1.0)
        self.state.set_debug(
            market_ws_connected=market_ws._ws is not None,
            user_ws_connected=(user_ws._ws is not None) if user_ws else "skipped (dry run)",
        )

        try:
            async with httpx.AsyncClient() as http:
                while self._is_running():
                    win_ts = next_window_start()
                    secs = seconds_until_next_window()
                    self.state.set_status(
                        f"Next window in {secs:.0f}s — pre-fetching {self.asset} market…"
                    )

                    # Fetch market while waiting for window boundary
                    from skeptic.utils.time import market_slug
                    slug = market_slug(self.asset, win_ts)
                    self.state.set_debug(last_market_slug=slug, last_market_found=None)
                    market = await gamma_client.get_current_window_market(
                        self.asset, win_ts, http
                    )
                    self.state.set_debug(last_market_found=market is not None)

                    # Sleep until window start (checking running every second)
                    if not await self._sleep_until(float(win_ts)):
                        break

                    if not self._is_running():
                        break

                    if market is None:
                        self.state.set_status("No market found — skipping window")
                        if not await self._sleep(5.0):
                            break
                        continue

                    # New window opening — clear old prices and set new window ts
                    with self.state._lock:
                        self.state.prices = []
                        self.state.current_window_ts = win_ts
                    self._up_token_id = market.up_token.token_id
                    self._down_token_id = market.down_token.token_id
                    self._current_window_ts = win_ts
                    self.state.set_debug(
                        up_token_id=market.up_token.token_id,
                        down_token_id=market.down_token.token_id,
                        condition_id=market.condition_id,
                    )
                    await market_ws.subscribe(market.up_token.token_id, market.down_token.token_id)
                    self.state.set_debug(market_ws_connected=market_ws._ws is not None)

                    if user_ws:
                        await user_ws.subscribe(market.condition_id)
                        self.state.set_debug(user_ws_connected=user_ws._ws is not None)

                    # Refresh live balance
                    if clob:
                        try:
                            bal = clob_client.get_usdc_balance(clob)
                            self.state.set_balance(bal)
                            self.state.set_debug(balance_fetched=bal)
                        except Exception as exc:
                            self.state.set_debug(balance_error=str(exc))

                    await self._execute_window(clob, user_ws, market_ws, market)

                    if user_ws:
                        await user_ws.unsubscribe(market.condition_id)

                    # Pause price recorder between windows; keep current_window_ts
                    # so the chart keeps showing the last window's data.
                    self._up_token_id = None
                    self._down_token_id = None

        finally:
            for t in bg:
                t.cancel()
            await asyncio.gather(*bg, return_exceptions=True)

    # ── window execution ─────────────────────────────────────────────────────

    async def _execute_window(
        self,
        clob,
        user_ws: Optional[UserChannel],
        market_ws: MarketChannel,
        market: Market,
    ) -> None:
        balance = self.state.snapshot()["balance"]
        bet = balance * POSITION_SIZE
        shares = bet / self.buy_threshold

        self.state.set_status(
            f"Window open — placing {self.asset} UP+DOWN @ ${self.buy_threshold:.2f} "
            f"({shares:.2f} sh, ${bet:.2f})"
        )

        trade_id = str(uuid.uuid4())

        # ── Place BUY orders ────────────────────────────────────────────────
        if self.dry_run:
            up_oid = f"dry-up-{trade_id[:6]}"
            dn_oid = f"dry-dn-{trade_id[:6]}"
        else:
            try:
                up_ord = clob_client.place_limit_order(
                    clob, market.up_token.token_id, "UP", "BUY",
                    self.buy_threshold, shares
                )
                dn_ord = clob_client.place_limit_order(
                    clob, market.down_token.token_id, "DOWN", "BUY",
                    self.buy_threshold, shares
                )
                up_oid, dn_oid = up_ord.order_id, dn_ord.order_id
            except Exception as exc:
                self.state.set_status(f"Order error: {exc}")
                return

        id_to_outcome = {up_oid: "UP", dn_oid: "DOWN"}
        self.state.set_status(f"Monitoring fills for {self.fill_window}s…")

        # ── Wait for a fill ─────────────────────────────────────────────────
        fill = await self._wait_for_fill(user_ws, market_ws, id_to_outcome)

        if fill is None:
            # No fill — cancel both and move on
            if not self.dry_run and clob:
                try:
                    clob_client.cancel_orders(clob, [up_oid, dn_oid])
                except Exception:
                    pass
            self.state.set_status(f"No fill in {self.fill_window}s — cancelled")
            return

        # ── Fill received ───────────────────────────────────────────────────
        cost = self.buy_threshold * fill.size
        record = TradeRecord(
            trade_id=trade_id,
            asset=self.asset,
            outcome=fill.outcome,
            buy_price=self.buy_threshold,
            buy_size=fill.size,
            buy_time=fill.ts,
            cost=cost,
            window_ts=self._current_window_ts,
            status="SELL_PLACED",
        )
        self.state.add_trade(record)
        self.state.set_status(f"Filled {fill.outcome}! Placing sell @ ${self.sell_threshold:.2f}…")

        # Cancel the unfilled side
        opp_oid = dn_oid if fill.outcome == "UP" else up_oid
        if not self.dry_run and clob:
            try:
                clob_client.cancel_order(clob, opp_oid)
            except Exception:
                pass

        # ── Place SELL order ────────────────────────────────────────────────
        sell_token = market.up_token if fill.outcome == "UP" else market.down_token
        if self.dry_run:
            sell_oid = f"dry-sell-{trade_id[:6]}"
        else:
            try:
                sell_ord = clob_client.place_limit_order(
                    clob, sell_token.token_id, fill.outcome, "SELL",
                    self.sell_threshold, fill.size
                )
                sell_oid = sell_ord.order_id
            except Exception as exc:
                self.state.set_status(f"Sell error: {exc}")
                return

        self.state.set_status(f"Sell live — waiting for ${self.sell_threshold:.2f}…")

        # ── Wait for sell fill (remaining window time, max ~4 min) ──────────
        elapsed = time.time() - self._current_window_ts
        remaining = max(10.0, 270.0 - elapsed)
        sell_fill = await self._wait_for_sell(user_ws, market_ws, sell_oid, fill.outcome, remaining)

        if sell_fill is not None:
            pnl = (self.sell_threshold - self.buy_threshold) * fill.size
            if self.dry_run:
                with self.state._lock:
                    self.state.balance = max(0.0, self.state.balance + pnl)
            self.state.update_trade(
                trade_id,
                sell_price=self.sell_threshold,
                sell_time=sell_fill.ts,
                pnl=pnl,
                status="SOLD",
            )
            self.state.set_status(f"Sold! P&L: ${pnl:+.4f}")
        else:
            # Sell order still live — window ended without a sell hit
            self.state.update_trade(trade_id, status="SELL_PLACED")
            self.state.set_status("Sell order live — awaiting fill or resolution")

    # ── fill monitoring helpers ──────────────────────────────────────────────

    async def _wait_for_fill(
        self,
        user_ws: Optional[UserChannel],
        market_ws: MarketChannel,
        id_to_outcome: dict[str, str],
    ) -> Optional[Fill]:
        """Wait up to fill_window seconds for a fill on either order."""
        deadline = asyncio.get_event_loop().time() + self.fill_window

        if self.dry_run or user_ws is None:
            # Simulate: fill when market price touches buy_threshold
            while asyncio.get_event_loop().time() < deadline:
                if not self._is_running():
                    return None
                for outcome, token_id in [
                    ("UP", self._up_token_id),
                    ("DOWN", self._down_token_id),
                ]:
                    if token_id is None:
                        continue
                    p = market_ws.price_cache.get(token_id)
                    if p is not None and p <= self.buy_threshold:
                        size = (self.state.snapshot()["balance"] * POSITION_SIZE) / self.buy_threshold
                        return Fill(
                            order_id=list(id_to_outcome.keys())[0],
                            outcome=outcome,
                            price=self.buy_threshold,
                            size=size,
                            ts=time.time(),
                        )
                await asyncio.sleep(0.5)
            return None

        # Live: drain the WebSocket fill queue
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0 or not self._is_running():
                return None
            try:
                fill = await asyncio.wait_for(
                    user_ws.fill_queue.get(), timeout=min(1.0, remaining)
                )
                if fill.order_id in id_to_outcome:
                    fill.outcome = id_to_outcome[fill.order_id]
                    return fill
                # Not our order — put it back
                await user_ws.fill_queue.put(fill)
                await asyncio.sleep(0.01)
            except asyncio.TimeoutError:
                continue

    async def _wait_for_sell(
        self,
        user_ws: Optional[UserChannel],
        market_ws: MarketChannel,
        sell_oid: str,
        outcome: str,
        timeout: float,
    ) -> Optional[Fill]:
        """Wait up to timeout seconds for the sell order to fill."""
        deadline = asyncio.get_event_loop().time() + timeout
        token_id = self._up_token_id if outcome == "UP" else self._down_token_id

        if self.dry_run or user_ws is None:
            # Simulate: fill when price reaches sell_threshold
            while asyncio.get_event_loop().time() < deadline:
                if not self._is_running():
                    return None
                p = market_ws.price_cache.get(token_id or "")
                if p is not None and p >= self.sell_threshold:
                    size = (self.state.snapshot()["balance"] * POSITION_SIZE) / self.buy_threshold
                    return Fill(
                        order_id=sell_oid,
                        outcome=outcome,
                        price=self.sell_threshold,
                        size=size,
                        ts=time.time(),
                    )
                await asyncio.sleep(0.5)
            return None

        # Live
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0 or not self._is_running():
                return None
            try:
                fill = await asyncio.wait_for(
                    user_ws.fill_queue.get(), timeout=min(1.0, remaining)
                )
                if fill.order_id == sell_oid:
                    fill.outcome = outcome
                    return fill
                await user_ws.fill_queue.put(fill)
                await asyncio.sleep(0.01)
            except asyncio.TimeoutError:
                continue

    # ── price recorder ───────────────────────────────────────────────────────

    async def _price_recorder(self, market_ws: MarketChannel) -> None:
        """Sample live prices as fast as possible."""
        n = 0
        while True:
            up_id = self._up_token_id
            dn_id = self._down_token_id
            wts = self._current_window_ts
            if up_id and dn_id:
                up_p = market_ws.price_cache.get(up_id)
                dn_p = market_ws.price_cache.get(dn_id)
                self.state.add_price(time.time(), up_p, dn_p, wts)
                n += 1
                self.state.set_debug(
                    price_cache_up=up_p,
                    price_cache_down=dn_p,
                    prices_recorded=n,
                    market_ws_connected=market_ws._ws is not None,
                )
            await asyncio.sleep(0.1)

    # ── utilities ────────────────────────────────────────────────────────────

    def _is_running(self) -> bool:
        with self.state._lock:
            return self.state.running

    async def _sleep(self, secs: float) -> bool:
        """Sleep secs, returning False early if engine should stop."""
        deadline = asyncio.get_event_loop().time() + secs
        while asyncio.get_event_loop().time() < deadline:
            if not self._is_running():
                return False
            await asyncio.sleep(min(1.0, deadline - asyncio.get_event_loop().time()))
        return True

    async def _sleep_until(self, target_ts: float) -> bool:
        """Sleep until target_ts, returning False if stopped early."""
        coarse = target_ts - time.time() - 0.1
        if coarse > 0:
            if not await self._sleep(coarse):
                return False
        # Tight poll for the last 100 ms
        while time.time() < target_ts:
            if not self._is_running():
                return False
            await asyncio.sleep(0.01)
        return True


# ──────────────────────────── Streamlit UI ───────────────────────────────────

def _init_state() -> None:
    if "state" not in st.session_state:
        st.session_state.state = TradingState()


def _render_sidebar(snap: dict) -> tuple[str, float, float, int, bool]:
    st.sidebar.title("⚙️ Skeptic")
    st.sidebar.markdown("---")

    running = snap["running"]

    asset = st.sidebar.selectbox(
        "Market", ASSETS, index=3, disabled=running
    )
    buy_threshold = st.sidebar.slider(
        "Buy Threshold", 0.01, 0.99, 0.40, 0.01, disabled=running,
        help="Place limit BUY when price ≤ this value"
    )
    sell_threshold = st.sidebar.slider(
        "Sell Threshold", 0.01, 0.99, 0.93, 0.01, disabled=running,
        help="Place limit SELL when price ≥ this value"
    )
    fill_window = st.sidebar.slider(
        "Fill Window (seconds)", 10, 300, 30, 5, disabled=running,
        help="Cancel unfilled buy orders after this many seconds"
    )
    dry_run = st.sidebar.checkbox(
        "Paper Trading (dry run)", value=True, disabled=running,
        help="Simulate orders without real capital"
    )

    st.sidebar.markdown("---")

    if not running:
        if st.sidebar.button("▶  Start Trading", type="primary", width='stretch'):
            state: TradingState = st.session_state.state
            engine = TradingEngine(
                state=state,
                asset=asset,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                fill_window=fill_window,
                dry_run=dry_run,
            )
            st.session_state.engine = engine
            engine.start()
            st.rerun()
    else:
        if st.sidebar.button("■  Stop Trading", type="secondary", width='stretch'):
            if "engine" in st.session_state:
                st.session_state.engine.stop()
            else:
                st.session_state.state.stop()
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption("Position size: 5% of wallet per bet")

    return asset, buy_threshold, sell_threshold, fill_window, dry_run


def _render_metrics(snap: dict) -> None:
    balance = snap["balance"]
    pnl = snap["total_pnl"]
    wins = snap["wins"]
    losses = snap["losses"]
    n = wins + losses
    win_rate = wins / n * 100 if n > 0 else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Balance", f"${balance:.2f}")
    pnl_delta = f"{pnl:+.2f}" if pnl != 0 else None
    c2.metric("Total P&L", f"${pnl:+.2f}", delta=pnl_delta)
    c3.metric("Wins", wins)
    c4.metric("Losses", losses)
    c5.metric("Win Rate", f"{win_rate:.0f}%")


def _render_chart(snap: dict, fill_window: int) -> None:
    prices: list[PricePoint] = snap["prices"]
    trades: list[TradeRecord] = snap["trades"]
    asset = snap.get("current_asset") or "—"
    window_ts = snap.get("current_window_ts", 0)
    buy_th = st.session_state.get("_buy_th", 0.40)
    sell_th = st.session_state.get("_sell_th", 0.93)

    # Only show prices for the current window
    window_prices = [p for p in prices if p.window_ts == window_ts and 0 <= (p.ts - window_ts) <= 305] if window_ts else []

    now_elapsed = time.time() - window_ts if window_ts else 0
    now_elapsed = max(0.0, min(now_elapsed, 300.0))

    # Seconds remaining in fill window (clamp to 0)
    fill_remaining = max(0.0, fill_window - now_elapsed)
    in_fill_window = fill_remaining > 0

    fig = go.Figure()

    # ── Fill window shading ──────────────────────────────────────────────────
    # Green zone: the fill window (0 → fill_window)
    fig.add_vrect(
        x0=0, x1=fill_window,
        fillcolor="rgba(0,224,150,0.07)",
        line_width=0,
        layer="below",
    )
    # Fill window deadline — solid line, color shifts red as time runs out
    urgency = 1.0 - (fill_remaining / fill_window) if fill_window > 0 else 1.0
    r = int(255 * urgency)
    g = int(224 * (1 - urgency))
    deadline_color = f"rgba({r},{g},80,0.9)"
    fig.add_vline(
        x=fill_window,
        line=dict(color=deadline_color, width=2, dash="solid"),
        annotation_text=f"fill window",
        annotation_position="top",
        annotation_font=dict(size=10, color=deadline_color),
    )

    # ── "Now" cursor ────────────────────────────────────────────────────────
    if window_ts and now_elapsed <= 300:
        fig.add_vline(
            x=now_elapsed,
            line=dict(color="rgba(255,255,255,0.5)", width=1, dash="dot"),
            annotation_text=f"{now_elapsed:.0f}s",
            annotation_position="bottom",
            annotation_font=dict(size=9, color="rgba(255,255,255,0.6)"),
        )

    # ── Price lines ──────────────────────────────────────────────────────────
    if window_prices:
        xs = [p.ts - window_ts for p in window_prices]
        up_vals = [p.up for p in window_prices]
        dn_vals = [p.down for p in window_prices]

        fig.add_trace(go.Scatter(
            x=xs, y=up_vals,
            name="UP",
            line=dict(color="#00e096", width=3),
            connectgaps=False,
            hovertemplate="t+%{x:.0f}s  %{y:.3f}<extra>UP</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=dn_vals,
            name="DOWN",
            line=dict(color="#ff4f6e", width=3),
            connectgaps=False,
            hovertemplate="t+%{x:.0f}s  %{y:.3f}<extra>DOWN</extra>",
        ))

    # ── Trade markers (current window only) ─────────────────────────────────
    for t in trades:
        if t.window_ts != window_ts:
            continue
        bx = t.buy_time - window_ts
        color = "#00e096" if t.outcome == "UP" else "#ff4f6e"
        fig.add_trace(go.Scatter(
            x=[bx], y=[t.buy_price],
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=16, color=color,
                        line=dict(width=1, color="white")),
            text=[f"BUY {t.outcome}"],
            textposition="top center",
            showlegend=False,
            hovertemplate=f"BUY {t.outcome} @ {t.buy_price:.3f}<extra></extra>",
        ))
        if t.sell_time and t.sell_price is not None:
            sx = t.sell_time - window_ts
            pnl_str = f"${t.pnl:+.3f}" if t.pnl is not None else "SELL"
            fig.add_trace(go.Scatter(
                x=[sx], y=[t.sell_price],
                mode="markers+text",
                marker=dict(symbol="triangle-down", size=16, color="#ffd700",
                            line=dict(width=1, color="white")),
                text=[pnl_str],
                textposition="bottom center",
                showlegend=False,
                hovertemplate=f"SELL @ {t.sell_price:.3f}<extra></extra>",
            ))

    # ── Threshold lines ──────────────────────────────────────────────────────
    fig.add_hline(y=buy_th, line=dict(dash="dot", color="rgba(255,255,255,0.3)", width=1),
                  annotation_text=f"buy {buy_th:.2f}", annotation_position="right")
    fig.add_hline(y=sell_th, line=dict(dash="dot", color="rgba(255,215,0,0.4)", width=1),
                  annotation_text=f"sell {sell_th:.2f}", annotation_position="right")

    # ── Title suffix ─────────────────────────────────────────────────────────
    if not window_ts:
        title_suffix = " — ⏳ waiting for next window"
    elif now_elapsed > 300:
        title_suffix = " — ⏳ window closed, waiting for next"
    elif in_fill_window:
        title_suffix = f" — ⏱ {fill_remaining:.0f}s until fill window closes"
    else:
        title_suffix = " — fill window closed"

    fig.update_layout(
        title=dict(text=f"{asset} UP / DOWN{title_suffix}", x=0, font=dict(color="#e0e0e0")),
        xaxis=dict(
            title="seconds into window",
            range=[0, 300],
            tickvals=list(range(0, 301, 30)),
            ticktext=[f"{s}s" for s in range(0, 301, 30)],
            color="#aaaaaa",
            gridcolor="#333333",
        ),
        yaxis=dict(
            title="Price",
            range=[0, 1],
            tickformat=".2f",
            color="#aaaaaa",
            gridcolor="#333333",
        ),
        height=380,
        margin=dict(l=10, r=80, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color="#e0e0e0")),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f1a",
        font=dict(color="#e0e0e0"),
    )

    st.plotly_chart(fig, width='stretch')


def _render_trade_log(snap: dict) -> None:
    trades: list[TradeRecord] = snap["trades"]

    if not trades:
        st.info("No trades yet. Start the engine to begin.")
        return

    STATUS_ICON = {
        "OPEN": "⏳ Open",
        "SELL_PLACED": "🎯 Sell live",
        "SOLD": "✅ Sold",
        "TIMEOUT": "❌ Timeout",
    }

    rows = []
    for t in reversed(trades):
        rows.append({
            "Time": datetime.fromtimestamp(t.buy_time).strftime("%H:%M:%S"),
            "Asset": t.asset,
            "Side": t.outcome,
            "Buy @": f"${t.buy_price:.3f}",
            "Shares": f"{t.buy_size:.2f}",
            "Cost": f"${t.cost:.2f}",
            "Sell @": f"${t.sell_price:.3f}" if t.sell_price else "—",
            "Sold at": datetime.fromtimestamp(t.sell_time).strftime("%H:%M:%S") if t.sell_time else "—",
            "P&L": f"${t.pnl:+.3f}" if t.pnl is not None else "—",
            "Status": STATUS_ICON.get(t.status, t.status),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, width='stretch', hide_index=True)


# ────────────────────────────── App entrypoint ───────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Skeptic | Trading Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_state()

    state: TradingState = st.session_state.state
    snap = state.snapshot()

    asset, buy_th, sell_th, fill_win, dry_run = _render_sidebar(snap)

    # Stash thresholds so the chart can draw reference lines
    st.session_state["_buy_th"] = buy_th
    st.session_state["_sell_th"] = sell_th

    # ── Header ────────────────────────────────────────────────────────────────
    col_h, col_badge = st.columns([4, 1])
    with col_h:
        mode_tag = "📄 Paper" if dry_run else "💰 Live"
        st.title(f"📈 Skeptic — {mode_tag} Trading")
    with col_badge:
        if snap["running"]:
            st.success("● RUNNING", icon=None)
        else:
            st.warning("● STOPPED", icon=None)

    if snap["error"]:
        st.error(f"🚨 {snap['error']}")

    st.caption(f"**Status:** {snap['status_msg']}")

    st.markdown("---")

    # ── Metrics ───────────────────────────────────────────────────────────────
    _render_metrics(snap)

    st.markdown("---")

    # ── Price chart ───────────────────────────────────────────────────────────
    st.subheader("Live Price Chart")
    _render_chart(snap, fill_win)

    st.markdown("---")

    # ── Trade log ─────────────────────────────────────────────────────────────
    st.subheader("Trade Log")
    _render_trade_log(snap)

    # ── Debug panel ───────────────────────────────────────────────────────────
    with st.expander("🔧 Debug", expanded=not snap["running"]):
        dbg = snap.get("debug", {})

        if not dbg:
            st.info("Start the engine to see debug info.")
        else:
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Credentials (.env)**")
                st.write({
                    "PRIVATE_KEY set": dbg.get("private_key_set"),
                    "WALLET_ADDRESS set": dbg.get("wallet_address_set"),
                    "CLOB_API_KEY set": dbg.get("clob_api_key_set"),
                    "dry_run": dbg.get("dry_run"),
                })
                st.markdown("**CLOB**")
                st.write({
                    "connected": dbg.get("clob_connected"),
                    "balance fetched": dbg.get("balance_fetched"),
                    "error": dbg.get("clob_error"),
                    "balance_error": dbg.get("balance_error"),
                })

            with c2:
                st.markdown("**WebSockets**")
                st.write({
                    "market WS connected": dbg.get("market_ws_connected"),
                    "user WS connected": dbg.get("user_ws_connected"),
                })
                st.markdown("**Market**")
                st.write({
                    "slug fetched": dbg.get("last_market_slug"),
                    "market found": dbg.get("last_market_found"),
                    "condition_id": dbg.get("condition_id"),
                    "up_token_id": (dbg.get("up_token_id") or "")[:20] + "…" if dbg.get("up_token_id") else None,
                    "down_token_id": (dbg.get("down_token_id") or "")[:20] + "…" if dbg.get("down_token_id") else None,
                })

            n_prices = dbg.get("prices_recorded", 0)
            up_p = dbg.get("price_cache_up")
            dn_p = dbg.get("price_cache_down")

            st.markdown("**Price Cache**")
            st.write({
                "UP price": round(up_p, 3) if up_p is not None else None,
                "DOWN price": round(dn_p, 3) if dn_p is not None else None,
                "prices recorded (total)": n_prices,
            })

            prices_in_snap = snap.get("prices", [])
            win_ts_snap = snap.get("current_window_ts", 0)
            matching = [p for p in prices_in_snap if p.window_ts == win_ts_snap and 0 <= (p.ts - win_ts_snap) <= 300] if win_ts_snap else []
            st.markdown("**Chart diagnostics**")
            st.write({
                "prices in snapshot": len(prices_in_snap),
                "current_window_ts": win_ts_snap,
                "prices matching window+range": len(matching),
                "now elapsed (s)": round(time.time() - win_ts_snap, 1) if win_ts_snap else None,
            })

            # ── Health diagnostics ───────────────────────────────────────────
            if dbg.get("last_market_found") is False:
                st.error(
                    f"Market not found for slug `{dbg.get('last_market_slug')}`. "
                    "The window may not have opened yet, or the asset isn't available."
                )
            elif n_prices > 0 and up_p is not None:
                st.success(
                    f"✓ Data flowing — {n_prices:,} price points recorded. "
                    f"UP={up_p:.3f}  DOWN={dn_p:.3f}. "
                    "If the chart looks flat, the market is genuinely near 50/50."
                )
            elif n_prices == 0 and dbg.get("market_ws_connected"):
                st.warning(
                    "Market WS connected but no prices yet. "
                    "Waiting for the next window to open so tokens can be subscribed."
                )

            if not dbg.get("private_key_set") and not dbg.get("dry_run"):
                st.error("PRIVATE_KEY is not set in .env — live trading will fail.")
            if not dbg.get("wallet_address_set") and not dbg.get("dry_run"):
                st.error("WALLET_ADDRESS is not set in .env — live trading will fail.")

    # ── Auto-refresh while engine is running ──────────────────────────────────
    if snap["running"]:
        time.sleep(1.5)
        st.rerun()


if __name__ == "__main__":
    main()
