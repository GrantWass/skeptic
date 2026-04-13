"""
Live executor for the coin-momentum buy strategy.

Watches the real coin price (via Binance aggTrade WebSocket) within each 5-minute window.
When the coin moves >= sigma_entry * sigma_value from the window open,
AND the Polymarket price for that side is <= max_pm_price, it buys.

Parameters (set at construction):
  asset         — single asset to trade (e.g. "BTC")
  sigma_value   — pre-computed sigma for that asset (std dev of 5-min window moves in $)
  sigma_entry   — multiplier: how many sigmas the coin must move to trigger (e.g. 1.0)
  max_pm_price  — maximum Polymarket price we're willing to pay (our edge threshold)
  direction     — "up", "down", or "both"
"""
import asyncio
import csv
import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
import httpx
import websockets

from skeptic import config
from skeptic.clients import clob as clob_client

_BALANCE_PHRASES = ("insufficient", "not enough", "balance", "funds")


class InsufficientBalanceError(RuntimeError):
    """Raised when an order is rejected due to insufficient USDC balance."""
from skeptic.clients import ctf as ctf_client
from skeptic.clients import gamma
from skeptic.clients.ws import MarketChannel
from skeptic.models.market import Market
from skeptic.utils.kelly import kelly_usdc, MOMENTUM_EDGE_THRESHOLD
from skeptic.utils.time import current_window_start, next_window_start, sleep_until

log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

LIVE_DIR = os.path.join("data", "live")

BINANCE_KLINES_URL  = "https://api.binance.com/api/v3/klines"
BINANCE_WS_BASE     = "wss://stream.binance.com:9443/ws"

class BinanceCoinStream:
    """
    Subscribes to Binance's aggTrade WebSocket stream for a single symbol and
    caches the latest trade price. Reconnects automatically on any error.

    Use get_price() to read the latest price (returns None until first message).
    """

    def __init__(self, symbol: str) -> None:
        self._symbol       = symbol.lower()
        self._price:       float | None = None
        self._last_update: float = 0.0
        self._event        = asyncio.Event()
        # (timestamp, quantity) for rolling volume sums — matches 1s OHLCV volume column
        self._vol_history: deque[tuple[float, float]] = deque(maxlen=500)

    def get_price(self) -> float | None:
        return self._price

    def get_vol_10s_log(self) -> float:
        """log1p of total trade quantity in the last 10 seconds — matches vol_10s_log training feature."""
        cutoff = time.time() - 10.0
        total  = sum(q for ts, q in self._vol_history if ts >= cutoff)
        return math.log1p(total)

    @property
    def stale(self) -> bool:
        """True if no price received yet or last update was >10 s ago."""
        return self._last_update == 0.0 or (time.time() - self._last_update) > 10.0

    @property
    def age_ms(self) -> float:
        """Milliseconds since last price update (∞ if never received)."""
        if self._last_update == 0.0:
            return float("inf")
        return (time.time() - self._last_update) * 1000.0

    async def next_price(self, timeout: float = 5.0) -> float | None:
        """
        Block until the next trade price arrives, then return it.
        Returns None on timeout (stream stale / disconnected).
        Multiple rapid prices between calls collapse to one wake-up —
        we always read the latest cached value.
        """
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        self._event.clear()
        return self._price

    async def run(self) -> None:
        url     = f"{BINANCE_WS_BASE}/{self._symbol}@aggTrade"
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=20
                ) as ws:
                    backoff = 1.0
                    async for raw in ws:
                        msg               = json.loads(raw)
                        self._price       = float(msg["p"])
                        self._last_update = time.time()
                        self._vol_history.append((self._last_update, float(msg["q"])))
                        self._event.set()
            except Exception as exc:
                log.warning(
                    "BinanceCoinStream %s disconnected: %s — reconnect in %.0fs",
                    self._symbol.upper(), exc, backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)


ASSET_TO_SYMBOL = {
    "BTC":  "BTCUSDT",
    "ETH":  "ETHUSDT",
    "SOL":  "SOLUSDT",
    "DOGE": "DOGEUSDT",
    "XRP":  "XRPUSDT",
    "BNB":  "BNBUSDT",
}

TRADE_FIELDS = [
    "ts", "asset", "side", "token_id",
    "fill_price", "fill_size", "fill_usdc", "fee_usdc",
    "sigma_value", "sigma_entry", "max_pm_price",
    "elapsed_second", "coin_open", "coin_trigger", "coin_move",
    "slippage", "window_start_ts", "window_end_ts",
    "sign_ms", "post_ms", "order_ms",
    "resolution", "pnl_usdc", "status", "order_id",
]

SLIPPAGE = 0.075
BUY_FEE_RATE = 0.015
MODEL_EDGE_THRESHOLD = 0.20   # min predicted_win - pm_ask to fire a model dry-run trade
EWMA_LAMBDA = 0.95             # decay factor for walk-forward sigma estimate

MODEL_TRADE_FIELDS = [
    "ts", "asset", "side", "token_id",
    "fill_price", "fill_size", "fill_usdc", "fee_usdc",
    "predicted_win", "edge",
    "elapsed_second", "coin_open", "coin_trigger", "coin_move",
    "window_start_ts", "window_end_ts",
    "sign_ms", "post_ms", "order_ms",
    "resolution", "pnl_usdc", "status", "order_id", "slippage",
]


@dataclass
class ModelTrade:
    ts: float
    asset: str
    side: str
    token_id: str
    fill_price: float
    fill_size: float
    fill_usdc: float
    fee_usdc: float
    predicted_win: float
    edge: float
    elapsed_second: int
    coin_open: float
    coin_trigger: float
    coin_move: float
    window_start_ts: int
    window_end_ts: int
    sign_ms: float | None = None   # ms for sign+POST combined (None for dry-run)
    post_ms: float | None = None   # always None — model never uses presigned path
    order_ms: float | None = None  # total ms from _execute_model_buy entry to order confirmed
    resolution: float | None = None
    pnl_usdc: float | None = None
    status: str = "open"
    order_id: str = ""
    slippage: float = 0.0


def _ensure_model_csv(trades_csv: str) -> None:
    os.makedirs(LIVE_DIR, exist_ok=True)
    if not os.path.exists(trades_csv):
        with open(trades_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=MODEL_TRADE_FIELDS).writeheader()
        return
    with open(trades_csv, newline="") as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames or []
        missing = [c for c in MODEL_TRADE_FIELDS if c not in existing_fields]
        if not missing:
            return
        rows = list(reader)
    tmp = trades_csv + ".migrate.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MODEL_TRADE_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            for col in missing:
                row.setdefault(col, "")
            w.writerow(row)
    os.replace(tmp, trades_csv)
    log.info("Migrated %s — added columns: %s", trades_csv, missing)


_model_csv_ensured: set[str] = set()


def _write_model_trade(trade: ModelTrade, trades_csv: str) -> None:
    if trades_csv not in _model_csv_ensured:
        _ensure_model_csv(trades_csv)
        _model_csv_ensured.add(trades_csv)
    with open(trades_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MODEL_TRADE_FIELDS)
        w.writerow(asdict(trade))


@dataclass
class MomentumTrade:
    ts: float
    asset: str
    side: str
    token_id: str
    fill_price: float
    fill_size: float
    fill_usdc: float
    fee_usdc: float
    sigma_value: float
    sigma_entry: float
    max_pm_price: float
    elapsed_second: int
    coin_open: float
    coin_trigger: float
    coin_move: float
    slippage: float
    window_start_ts: int
    window_end_ts: int
    sign_ms: float | None = None   # ms to sign the order (None if presigned path or dry-run)
    post_ms: float | None = None   # ms for the HTTP POST to return (None on fresh-sign path)
    order_ms: float | None = None  # total ms from _execute_buy entry to order confirmed (None if dry-run)
    resolution: float | None = None
    pnl_usdc: float | None = None
    status: str = "open"
    order_id: str = ""


def _ensure_live_dir(trades_csv: str) -> None:
    os.makedirs(LIVE_DIR, exist_ok=True)
    if not os.path.exists(trades_csv):
        with open(trades_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=TRADE_FIELDS).writeheader()
        return

    # Migrate existing file if it's missing any columns from TRADE_FIELDS.
    with open(trades_csv, newline="") as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames or []
        missing = [c for c in TRADE_FIELDS if c not in existing_fields]
        if not missing:
            return
        rows = list(reader)

    tmp = trades_csv + ".migrate.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRADE_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            for col in missing:
                row.setdefault(col, "")
            w.writerow(row)
    os.replace(tmp, trades_csv)
    log.info("Migrated %s — added columns: %s", trades_csv, missing)


def _write_trade(trade: MomentumTrade, trades_csv: str) -> None:
    _ensure_live_dir(trades_csv)
    with open(trades_csv, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=TRADE_FIELDS).writerow(asdict(trade))


def _write_status(status: dict, path: str) -> None:
    os.makedirs(LIVE_DIR, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(status, f)
    os.replace(tmp, path)


class MomentumBuyExecutor:
    """
    Buys a Polymarket UP/DOWN side when:
      1. Coin price moves >= sigma_entry * sigma_value from window open
      2. Polymarket ask for that side is <= max_pm_price
    """

    def __init__(
        self,
        asset: str,
        sigma_value: float,
        sigma_entry: float,
        max_pm_price: float,
        direction: str = "both",
        wallet_pct: float = 0.10,
        fixed_usdc: float | None = None,
        dry_run: bool = False,
        name: str = "momentum",
        model_cfg: dict | None = None,
        momentum_cfg: dict | None = None,
    ) -> None:
        assert direction in ("up", "down", "both"), "direction must be 'up', 'down', or 'both'"
        self.asset        = asset
        self.sigma_value  = sigma_value
        self.sigma_entry  = sigma_entry
        self.max_pm_price = max_pm_price
        self.direction    = direction
        self.wallet_pct   = wallet_pct
        self.fixed_usdc      = fixed_usdc
        self.dry_run         = dry_run
        _mom = momentum_cfg or {}
        self._momentum_enabled: bool = bool(_mom.get("enabled", True))
        _mc = model_cfg or {}
        self._model_cfg = _mc
        self._model_enabled:         bool        = bool(_mc.get("enabled", True))
        self._model_window_start:    int         = int(_mc.get("window_start", 0))
        self._model_window_end:      int         = int(_mc.get("window_end",   300))
        self._MODEL_EDGE_THRESHOLD:  float       = float(_mc.get("edge_threshold", MODEL_EDGE_THRESHOLD))
        self._model_fixed_usdc:         float | None = _mc.get("fixed_usdc")

        self._model_multi_trade_cooldown: int        = int(_mc.get("multi_trade_cooldown", 0))
        self._model_max_trades_per_window: int       = int(_mc.get("max_trades_per_window", 3))
        self._model_max_slippage:          float     = float(_mc.get("max_slippage", 0.10))

        self._symbol = ASSET_TO_SYMBOL.get(asset.upper())
        if self._symbol is None:
            raise ValueError(f"No Binance symbol for asset '{asset}'")

        self._trades_csv       = os.path.join(LIVE_DIR, f"trades_{name}.csv")
        self._model_trades_csv = os.path.join(LIVE_DIR, f"trades_model_{name}.csv")
        self._status_json      = os.path.join(LIVE_DIR, f"status_{name}.json")

        self._clob        = clob_client.build_client()
        self._ws          = MarketChannel()
        self._coin_stream = BinanceCoinStream(self._symbol)
        self._http: httpx.AsyncClient | None = None

        # Load model if available — used for non-blocking inference in status loop
        model_path = os.path.join("data", "models", f"{asset.lower()}.joblib")
        self._model: dict[str, Any] | None = None
        if os.path.exists(model_path):
            try:
                self._model = joblib.load(model_path)
            except Exception as exc:
                log.warning("Could not load model for %s: %s", asset, exc)

        # EWMA sigma state — updated after every completed window
        # Initialize from config sigma^2 so the first live update treats config sigma as prior state.
        self._sigma_initial: float = sigma_value
        self._ewma_var: float = float(sigma_value) ** 2

        # Per-window state
        self._market:   Market | None = None
        self._filled:   bool = False
        self._trade:    MomentumTrade | None = None
        self._presigned: dict[str, object] = {}
        self._tp_sell_order_id: str | None = None   # open take-profit sell order
        self._window_start: int = 0
        self._window_end:   int = 0
        self._coin_open:    float | None = None
        self._coin_current: float | None = None
        self._position_usdc: float = 0.0
        # (timestamp, price) ring buffer — used to compute velocity/acceleration features
        self._price_history: deque[tuple[float, float]] = deque(maxlen=60)
        # Model dry-run state — separate from live trading state
        self._model_last_fire_ts: float | None = None   # None = not fired this window
        self._model_trades: list[ModelTrade]   = []     # all trades this window
        self._model_trade:  ModelTrade | None  = None   # most recent trade (for status display)

    # ── Public ────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        _ensure_live_dir(self._trades_csv)
        _ensure_model_csv(self._model_trades_csv)
        _model_csv_ensured.add(self._model_trades_csv)
        async with httpx.AsyncClient(http2=True, limits=httpx.Limits(max_connections=50, max_keepalive_connections=12)) as http:
            self._http = http
            ws_task          = asyncio.create_task(self._ws.run())
            coin_stream_task = asyncio.create_task(self._coin_stream.run())
            status_task      = asyncio.create_task(self._status_loop())

            # Warm up HTTP connection and crypto library on the dedicated signing executor.
            from skeptic.clients.clob import _signing_executor
            from py_clob_client.clob_types import MarketOrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY

            def _dummy_sign() -> None:
                try:
                    self._clob.create_market_order(MarketOrderArgs(
                        token_id="0x" + "aa" * 32, amount=1.0, side=BUY,
                        price=0.50, order_type=OrderType.FOK,  # type: ignore[arg-type]
                    ))
                except Exception:
                    pass

            loop = asyncio.get_event_loop()
            await asyncio.gather(
                loop.run_in_executor(_signing_executor, _dummy_sign),
                http.get(f"{self._clob.host}/time"),
                asyncio.sleep(1.0),
                return_exceptions=True,
            )
            log.info("Warmup complete (signing thread + HTTP connection)")

            try:
                while True:
                    ws  = current_window_start()
                    we  = ws + config.WINDOW_SECS
                    now = time.time()

                    if now > we - 5:
                        nw = next_window_start()
                        await sleep_until(nw)
                        ws, we = nw, nw + config.WINDOW_SECS

                    await self._run_window(ws, we)
                    await sleep_until(we)
            finally:
                ws_task.cancel()
                coin_stream_task.cancel()
                status_task.cancel()

    def _disable_strategy_on_balance_error(self, strategy: str, side: str, exc: Exception) -> None:
        """Disable one strategy for this asset after insufficient-balance rejection; keep process running."""
        if strategy == "momentum":
            self._momentum_enabled = False
            self._filled = False
        elif strategy == "model":
            self._model_enabled = False
            self._model_last_fire_ts = None

        log.error(
            "Insufficient balance on %s %s (%s): %s — disabling %s and switching it to DRY-RUN; executor will continue",
            self.asset,
            side,
            strategy.upper(),
            exc,
            strategy,
        )

    # ── Window ────────────────────────────────────────────────────────────────

    async def _run_window(self, window_start: int, window_end: int) -> None:
        self._filled        = False
        self._trade         = None
        self._market        = None
        self._presigned     = {}
        self._tp_sell_order_id = None
        self._coin_open     = None
        self._coin_current  = None
        self._window_start  = window_start
        self._window_end    = window_end
        self._price_history.clear()
        self._model_last_fire_ts = None
        self._model_trades       = []
        self._model_trade        = None

        # Apply EWMA sigma for this window (estimated from all prior windows)
        old_sigma = self.sigma_value
        if self._ewma_var is not None:
            new_sigma = float(self._ewma_var ** 0.5)
            self.sigma_value = new_sigma
            log.info(
                "%s EWMA σ update: %.6g → %.6g  (λ=%.2f)",
                self.asset, old_sigma, new_sigma, EWMA_LAMBDA,
            )
        else:
            log.info(
                "%s EWMA σ: using config seed %.6g (no prior windows yet)",
                self.asset, self.sigma_value,
            )
        balance = await asyncio.to_thread(clob_client.get_usdc_balance, self._clob)
        if self.fixed_usdc is not None:
            self._position_usdc = self.fixed_usdc
        else:
            self._position_usdc = round(balance * self.wallet_pct, 4)

        # Prefer the live WebSocket price; fall back to REST kline if stream is stale
        stream_price = self._coin_stream.get_price()
        if stream_price is not None and not self._coin_stream.stale:
            self._coin_open = stream_price
        else:
            self._coin_open = await self._fetch_coin_price_at(window_start)

        if self._coin_open is None:
            log.error("Could not fetch coin open price for %s — skipping window", self._symbol)
            return

        # Discover Polymarket market
        market = await gamma.get_current_window_market(
            self.asset, window_start, self._http, retries=24, retry_delay=5.0)
        if market is None:
            log.warning("No Polymarket market for %s this window — skipping", self.asset)
            return
        self._market = market

        # Subscribe to WebSocket and pre-sign orders
        all_tokens = [market.up_token.token_id, market.down_token.token_id]
        await self._ws.subscribe(*all_tokens)
        await self._ws.reconnect()
        # Wait for first PM price instead of a fixed sleep
        for _ in range(10):
            if any(self._ws.get_ask(t) is not None for t in all_tokens):
                break
            await asyncio.sleep(0.1)

        if not self.dry_run and self._momentum_enabled:
            for token_id in all_tokens:
                try:
                    presigned_tuple = await asyncio.to_thread(
                        clob_client.presign_market_order,
                        self._clob, token_id, self._position_usdc, 0.90,
                    )
                    self._presigned[token_id] = presigned_tuple
                except Exception as exc:
                    log.warning("Pre-sign failed for %s: %s", token_id[:10], exc)


        # Watch until 8s before end
        watch_until = window_end - 8
        now = time.time()
        if now < watch_until:
            try:
                await asyncio.wait_for(
                    self._momentum_watch_loop(),
                    timeout=watch_until - now,
                )
            except asyncio.TimeoutError:
                pass

        # Snapshot state now — _run_window will reset these at the next window start
        trade_snap        = self._trade
        model_trades_snap = list(self._model_trades)
        market_snap       = self._market
        coin_open_snap    = self._coin_open
        coin_close_snap   = self._coin_current

        # Cancel any open take-profit sell before resolution
        if self._tp_sell_order_id:
            tp_oid = self._tp_sell_order_id
            self._tp_sell_order_id = None
            await asyncio.to_thread(clob_client.cancel_order, self._clob, tp_oid)

        if trade_snap:
            asyncio.create_task(self._resolve_bg(
                trade_snap, market_snap, coin_open_snap, coin_close_snap, delay=8.0
            ))

        for _mt in model_trades_snap:
            if _mt.status in ("open", "fok_killed"):
                asyncio.create_task(self._resolve_model_bg(
                    _mt, market_snap, coin_open_snap, coin_close_snap, delay=8.0
                ))

        await self._ws.unsubscribe(*all_tokens)

        # Update EWMA with this window's realized move (used as sigma for the NEXT window)
        if coin_open_snap is not None and coin_close_snap is not None:
            move = coin_close_snap - coin_open_snap
            move_sq = move ** 2
            if self._ewma_var is None:
                self._ewma_var = move_sq   # seed on first completed window
            else:
                self._ewma_var = EWMA_LAMBDA * self._ewma_var + (1.0 - EWMA_LAMBDA) * move_sq

    async def _momentum_watch_loop(self) -> None:
        """
        Check threshold once per second using the latest aggTrade price —
        matching the 1-second candle close used in backtesting.

        aggTrades arrive many times per second; we accumulate them via
        BinanceCoinStream and sample the latest price at each 1s tick.
        This means a mid-second spike that reverts before the tick fires
        will not trigger, exactly as in the training data.

        In dry-run mode, logs a per-tick DEBUG line and a 30-second INFO heartbeat
        so the stream health can be verified without placing real orders.
        """
        threshold  = self.sigma_entry * self.sigma_value
        _skipped: set[str] = set()
        _stale_warns    = 0

        while True:
            # Sleep until the next 1-second boundary, then sample the latest price.
            # This matches the cadence of the 1s OHLCV candles used in training.
            now = time.time()
            await asyncio.sleep(1.0 - (now % 1.0))

            coin_price = self._coin_stream.get_price()

            if coin_price is None or self._coin_stream.stale:
                _stale_warns += 1
                if _stale_warns % 3 == 1:
                    log.warning(
                        "BinanceCoinStream stale (age=%.0f ms) — no price",
                        self._coin_stream.age_ms,
                    )
                continue

            _stale_warns       = 0
            self._coin_current = coin_price
            self._price_history.append((time.time(), coin_price))

            if self._filled:
                continue

            move   = coin_price - self._coin_open
            sigmas = move / self.sigma_value if self.sigma_value else 0.0

            if self.direction in ("up", "both") and move >= threshold:
                pm_price = self._ws.get_ask(self._market.up_token.token_id)
                if pm_price is not None and pm_price <= self.max_pm_price:
                    log.info(
                        "TRIGGER UP  coin_open=%g  coin_now=%g  move=%+g (%.2fσ)  pm_ask=%.4f%s",
                        self._coin_open, coin_price, move, sigmas, pm_price,
                        "  [MOMENTUM DISABLED]" if not self._momentum_enabled else "",
                    )
                    self._filled = True
                    _skipped.discard("up")
                    await self._execute_buy("UP", self._market.up_token.token_id,
                                            pm_price, coin_price, move)
                elif pm_price is not None:
                    _skipped.add("up")
            else:
                _skipped.discard("up")

            if self.direction in ("down", "both") and move <= -threshold:
                pm_price = self._ws.get_ask(self._market.down_token.token_id)
                if pm_price is not None and pm_price <= self.max_pm_price:
                    log.info(
                        "TRIGGER DOWN  coin_open=%g  coin_now=%g  move=%+g (%.2fσ)  pm_ask=%.4f%s",
                        self._coin_open, coin_price, move, sigmas, pm_price,
                        "  [MOMENTUM DISABLED]" if not self._momentum_enabled else "",
                    )
                    self._filled = True
                    _skipped.discard("down")
                    await self._execute_buy("DOWN", self._market.down_token.token_id,
                                            pm_price, coin_price, move)
                elif pm_price is not None:
                    _skipped.add("down")
            else:
                _skipped.discard("down")

    # ── Execution ─────────────────────────────────────────────────────────────

    def _kelly_stake(self, edge: float) -> float:
        """Scale momentum stake from fixed_usdc up to KELLY_MAX_USDC. fixed_usdc is the floor."""
        fixed = self.fixed_usdc if self.fixed_usdc is not None else self._position_usdc
        return kelly_usdc(edge=edge, edge_threshold=MOMENTUM_EDGE_THRESHOLD, fixed_usdc=fixed)

    def _model_kelly_stake(self, edge: float) -> float:
        """Scale model stake from fixed_usdc up to KELLY_MAX_USDC. fixed_usdc is the floor."""
        fixed = self._model_fixed_usdc if self._model_fixed_usdc is not None else self._position_usdc
        return kelly_usdc(edge=edge, edge_threshold=self._MODEL_EDGE_THRESHOLD, fixed_usdc=fixed)

    async def _execute_buy(
        self,
        side: str,
        token_id: str,
        trigger_pm_price: float,
        coin_trigger: float,
        coin_move: float,
    ) -> None:
        t0 = time.perf_counter()

        # Kelly sizing: edge = max_pm_price - ask
        stake    = self._kelly_stake(self.max_pm_price - trigger_pm_price)
        est_size = round(stake / trigger_pm_price, 2)
        fee_usdc = round(stake * BUY_FEE_RATE, 4)

        trade = MomentumTrade(
            ts=time.time(),
            asset=self.asset,
            side=side,
            token_id=token_id,
            fill_price=trigger_pm_price,
            fill_size=est_size,
            fill_usdc=round(stake + fee_usdc, 4),
            fee_usdc=fee_usdc,
            sigma_value=self.sigma_value,
            sigma_entry=self.sigma_entry,
            max_pm_price=self.max_pm_price,
            elapsed_second=int(time.time() - self._window_start),
            coin_open=self._coin_open,
            coin_trigger=coin_trigger,
            coin_move=coin_move,
            slippage=SLIPPAGE,
            window_start_ts=self._window_start,
            window_end_ts=self._window_end,
        )

        if self.dry_run or not self._momentum_enabled:
            log.info(
                "[%s] BUY %s %s  $%.2f USDC  pm=%.4f  coin_open=%g  coin_now=%g  move=%+g",
                "DRY RUN" if self.dry_run else "MOMENTUM DISABLED",
                self.asset, side, stake, trigger_pm_price,
                self._coin_open, coin_trigger, coin_move,
            )
            trade.order_id = "DRY_RUN"
        else:
            try:
                presigned = self._presigned.get(token_id)
                fixed = self.fixed_usdc if self.fixed_usdc is not None else self._position_usdc
                if presigned is not None and stake <= fixed:
                    # Fast path: Kelly didn't scale up — use pre-signed order.
                    # Hot path cost: HMAC headers (~0.3ms) + network POST only.
                    _, serialized_body, body_dict, pre_usdc = presigned
                    t_post_start = time.perf_counter()
                    order = await clob_client.post_preserialized_order_async(
                        self._http, self._clob, serialized_body, body_dict,
                        token_id, side, pre_usdc,
                    )
                    trade.sign_ms = None  # presigned — no sign cost at trigger time
                    trade.post_ms = round((time.perf_counter() - t_post_start) * 1000, 1)
                    stake = pre_usdc
                    trade.fee_usdc = round(pre_usdc * BUY_FEE_RATE, 4)
                    trade.fill_usdc = round(pre_usdc + trade.fee_usdc, 4)
                else:
                    # Kelly scaled up (or no pre-signed order) — sign fresh at trigger time.
                    order, trade.sign_ms, trade.post_ms = await clob_client.sign_and_post_async(
                        self._http, self._clob, token_id, side, stake, price_cap=0.90,
                    )

                trade.order_ms   = round((time.perf_counter() - t0) * 1000, 1)
                trade.order_id   = order.order_id
                trade.fill_price = order.price if order.price > 0 else trigger_pm_price
                trade.fill_size  = order.size  if order.size  > 0 else est_size
                notional_usdc    = round(trade.fill_price * trade.fill_size, 4)
                trade.fee_usdc   = round(notional_usdc * BUY_FEE_RATE, 4)
                trade.fill_usdc  = round(notional_usdc + trade.fee_usdc, 4)
                trade.slippage   = round(trade.fill_price - trigger_pm_price, 4)
                _fmt_ms = lambda v: f"{v:.0f}ms" if v is not None else "—"
                log.info(
                    "FILLED  %s %s  %.4f shares @ %.4f  ($%.2f incl fee %.4f)"
                    "  slippage=%+.4f  order=%s  sign=%s post=%s total=%s",
                    self.asset, side, trade.fill_size, trade.fill_price,
                    trade.fill_usdc, trade.fee_usdc, trade.slippage, order.order_id[:16],
                    _fmt_ms(trade.sign_ms), _fmt_ms(trade.post_ms), _fmt_ms(trade.order_ms),
                )
            except Exception as exc:
                trade.order_ms = round((time.perf_counter() - t0) * 1000, 1)
                err_str = str(exc).lower()
                if any(p in err_str for p in _BALANCE_PHRASES):
                    trade.status = "insufficient_balance_disabled"
                    self._disable_strategy_on_balance_error("momentum", side, exc)
                else:
                    if "fully filled" in err_str or "fok" in err_str:
                        # FOK killed — log as hypothetical with assumed slippage, resolve for analysis
                        FOK_SLIPPAGE = 0.25
                        trade.fill_price = round(trigger_pm_price + FOK_SLIPPAGE, 4)
                        trade.fill_size  = round(stake / trade.fill_price, 2) if trade.fill_price > 0 else 0.0
                        notional         = round(trade.fill_price * trade.fill_size, 4)
                        trade.fee_usdc   = round(notional * BUY_FEE_RATE, 4)
                        trade.fill_usdc  = round(notional + trade.fee_usdc, 4)
                        trade.slippage   = FOK_SLIPPAGE
                        trade.status     = "fok_killed"
                        trade.order_id   = "FOK_KILLED"
                        self._filled     = False  # allow retry in same window
                        log.warning("FOK KILLED %s %s  hypothetical fill=%.4f (+%.2f slip)", self.asset, side, trade.fill_price, FOK_SLIPPAGE)
                    else:
                        log.error("Order failed %s %s: %s", self.asset, side, exc)
                        trade.status = "order_failed"
                        # bad token / orderbook gone — don't retry

        self._trade = trade
        await asyncio.to_thread(_write_trade, trade, self._trades_csv)

        # Place take-profit sell immediately after fill — sits in book at 0.98
        if not self.dry_run and trade.status == "open" and trade.fill_size > 0:
            asyncio.create_task(self._take_profit_bg(trade, token_id))

    # ── Take-profit sell ──────────────────────────────────────────────────────

    TAKE_PROFIT_PRICE = 0.98

    async def _take_profit_bg(self, trade: "MomentumTrade", token_id: str) -> None:
        """
        Place a GTC SELL limit order at 0.98 shortly after fill — shares need a
        few seconds to settle before the CLOB sees the balance.
        Cancelled at window end if unfilled.
        """
        await asyncio.sleep(5.0)  # wait for shares to settle in CLOB balance

        # Use actual token balance so we never try to sell more than we hold
        MIN_SELL_SIZE = 5.0
        sell_size = await asyncio.to_thread(clob_client.get_token_balance, self._clob, token_id)
        if sell_size < MIN_SELL_SIZE:
            log.info("Take-profit: skipping %s — size %.4f below min %.0f shares", self.asset, sell_size, MIN_SELL_SIZE)
            return
        sell_size = round(sell_size, 4)

        sell_order = None
        try:
            sell_order = await asyncio.to_thread(
                clob_client.place_limit_order,
                self._clob, token_id, trade.side,
                "SELL", self.TAKE_PROFIT_PRICE, sell_size,
            )
        except Exception as exc:
            log.error("Take-profit order placement failed %s: %s", self.asset, exc)
            return

        if sell_order is None:
            return

        self._tp_sell_order_id = sell_order.order_id
        log.info(
            "TAKE-PROFIT order placed %s %s  @ %.2f  size=%.4f  order=%s",
            self.asset, trade.side, self.TAKE_PROFIT_PRICE,
            trade.fill_size, sell_order.order_id[:16],
        )

        # Poll for fill confirmation until window end
        poll_until = self._window_end - 5
        while time.time() < poll_until:
            # If cancel cleared the ID, the order was cancelled — stop polling
            if self._tp_sell_order_id is None:
                return
            open_orders = await asyncio.to_thread(
                clob_client.get_open_orders, self._clob, None,
            )
            open_ids = {o.get("id") or o.get("order_id") for o in open_orders}
            if self._tp_sell_order_id not in open_ids:
                # Order gone from open orders — it filled
                payout = self.TAKE_PROFIT_PRICE * trade.fill_size
                pnl = payout - trade.fill_usdc
                trade.pnl_usdc   = round(pnl, 4)
                trade.status     = "won"
                trade.resolution = self.TAKE_PROFIT_PRICE
                self._tp_sell_order_id = None
                log.info(
                    "TAKE-PROFIT FILLED %s %s  @ %.2f  PnL=$%.4f",
                    self.asset, trade.side, self.TAKE_PROFIT_PRICE, pnl,
                )
                await asyncio.to_thread(_write_trade, trade, self._trades_csv)
                return
            await asyncio.sleep(3.0)

    # ── Resolution ────────────────────────────────────────────────────────────

    async def _resolve_bg(
        self,
        trade: "MomentumTrade",
        market: "Market | None",
        coin_open: "float | None",
        coin_close: "float | None",
        delay: float = 0.0,
    ) -> None:
        if not trade or trade.status not in ("open", "fok_killed"):
            return
        is_fok = trade.status == "fok_killed"
        await asyncio.sleep(delay)
        if trade.status not in ("open", "fok_killed"):  # take-profit may have resolved it during the delay
            return

        # Poll PM up to 60s for settlement
        pm_resolved_up: bool | None = None
        if market:
            for _ in range(12):
                p = self._ws.get_price(market.up_token.token_id)
                if p is not None:
                    if p >= 0.95:   pm_resolved_up = True;  break
                    elif p <= 0.05: pm_resolved_up = False; break
                await asyncio.sleep(5.0)

        coin_resolved_up: bool | None = None
        if coin_close is not None and coin_open is not None:
            diff = coin_close - coin_open
            if diff > 0:   coin_resolved_up = True
            elif diff < 0: coin_resolved_up = False

        if pm_resolved_up is not None:
            resolved_up = pm_resolved_up
        elif coin_resolved_up is not None:
            log.warning("RESOLVE  %s  PM ambiguous — using coin direction", self.asset)
            resolved_up = coin_resolved_up
        else:
            log.warning("Cannot resolve %s: PM ambiguous and coin is flat", self.asset)
            trade.status = "unresolved"
            await asyncio.to_thread(_write_trade, trade, self._trades_csv)
            return

        win = (resolved_up and trade.side.upper() == "UP") or (not resolved_up and trade.side.upper() == "DOWN")
        payout = trade.fill_size if win else 0.0
        pnl = payout - trade.fill_usdc
        trade.resolution = 1.0 if resolved_up else 0.0
        trade.pnl_usdc   = round(pnl, 4)
        if is_fok:
            trade.status = "fok_won" if win else "fok_lost"
        else:
            trade.status = "won" if win else "lost"
        log.info("RESOLVED  %s %s  PnL=$%.4f  %s", self.asset, trade.side, pnl, trade.status.upper())
        await asyncio.to_thread(_write_trade, trade, self._trades_csv)

    async def _execute_model_buy(
        self,
        side: str,
        token_id: str,
        trigger_ask: float,
        edge: float,
        predicted_win: float,
        elapsed: int,
    ) -> None:
        _trigger_ts = time.perf_counter()
        paper_run = not self._model_enabled
        model_usdc = self._model_kelly_stake(edge)
        fill_price = trigger_ask + SLIPPAGE if paper_run else trigger_ask
        est_size   = round(model_usdc / fill_price, 2) if fill_price > 0 else 0.0
        fee_usdc   = round(model_usdc * BUY_FEE_RATE, 4)
        coin_move  = (self._coin_current or self._coin_open or 0.0) - (self._coin_open or 0.0)

        mtrade = ModelTrade(
            ts=time.time(),
            asset=self.asset,
            side=side,
            token_id=token_id,
            fill_price=fill_price,
            fill_size=est_size,
            fill_usdc=round(model_usdc + fee_usdc, 4),
            fee_usdc=fee_usdc,
            predicted_win=round(predicted_win, 4),
            edge=round(edge, 4),
            elapsed_second=elapsed,
            coin_open=self._coin_open or 0.0,
            coin_trigger=self._coin_current or self._coin_open or 0.0,
            coin_move=coin_move,
            window_start_ts=self._window_start,
            window_end_ts=self._window_end,
        )

        if paper_run:
            mtrade.order_id = "DRY_RUN"
            mtrade.slippage = SLIPPAGE
            # sign_ms / post_ms / order_ms intentionally left None — no real order placed
            log.info("[MODEL] DRY-RUN ENTRY %s %s  predicted=%.1f%%  ask=%.3f  fill=%.3f  edge=%+.3f  elapsed=%ds",
                     self.asset, side, predicted_win * 100, trigger_ask, fill_price, edge, elapsed)
        else:
            try:
                order, mtrade.sign_ms, mtrade.post_ms = await clob_client.sign_and_post_async(
                    self._http, self._clob, token_id, side, model_usdc,
                    price_cap=trigger_ask + self._model_max_slippage,
                )
                mtrade.order_ms   = round((time.perf_counter() - _trigger_ts) * 1000, 1)
                mtrade.order_id   = order.order_id
                mtrade.fill_price = order.price if order.price > 0 else trigger_ask
                mtrade.fill_size  = order.size  if order.size  > 0 else est_size
                notional_usdc     = round(mtrade.fill_price * mtrade.fill_size, 4)
                mtrade.fee_usdc   = round(notional_usdc * BUY_FEE_RATE, 4)
                mtrade.fill_usdc  = round(notional_usdc + mtrade.fee_usdc, 4)
                mtrade.slippage   = round(mtrade.fill_price - trigger_ask, 4)
                log.info("[MODEL] FILLED %s %s  predicted=%.1f%%  %.4f shares @ %.4f  ($%.2f)  edge=%+.3f  elapsed=%ds  order=%.0fms",
                         self.asset, side, predicted_win * 100, mtrade.fill_size,
                         mtrade.fill_price, mtrade.fill_usdc, edge, elapsed, mtrade.order_ms)
            except Exception as exc:
                mtrade.order_ms = round((time.perf_counter() - _trigger_ts) * 1000, 1)
                log.error("[MODEL] Order failed %s %s: %s", self.asset, side, exc)
                mtrade.status = "order_failed"
                err_str = str(exc).lower()
                if any(p in err_str for p in _BALANCE_PHRASES):
                    mtrade.status = "insufficient_balance_disabled"
                    self._disable_strategy_on_balance_error("model", side, exc)
                else:
                    is_fok_failure = "fully filled" in err_str or "fok" in err_str
                    if is_fok_failure:
                        FOK_SLIPPAGE = 0.25
                        mtrade.fill_price = round(trigger_ask + FOK_SLIPPAGE, 4)
                        mtrade.fill_size  = round(model_usdc / mtrade.fill_price, 2) if mtrade.fill_price > 0 else 0.0
                        notional          = round(mtrade.fill_price * mtrade.fill_size, 4)
                        mtrade.fee_usdc   = round(notional * BUY_FEE_RATE, 4)
                        mtrade.fill_usdc  = round(notional + mtrade.fee_usdc, 4)
                        mtrade.slippage   = FOK_SLIPPAGE
                        mtrade.status     = "fok_killed"
                        mtrade.order_id   = "FOK_KILLED"
                        self._model_last_fire_ts = None  # allow retry this window
                        log.warning("[MODEL] FOK KILLED %s %s  hypothetical fill=%.4f (+%.2f slip)", self.asset, side, mtrade.fill_price, FOK_SLIPPAGE)
                        self._model_trades.append(mtrade)
                        self._model_trade = mtrade
                        await asyncio.to_thread(_write_model_trade, mtrade, self._model_trades_csv)
                        return  # don't count against per-window trade limit
                    # else: bad token / orderbook gone — don't retry

        self._model_trades.append(mtrade)
        self._model_trade = mtrade
        await asyncio.to_thread(_write_model_trade, mtrade, self._model_trades_csv)

    async def _resolve_model_bg(
        self,
        trade: "ModelTrade",
        market: "Market | None",
        coin_open: "float | None",
        coin_close: "float | None",
        delay: float = 0.0,
    ) -> None:
        if not trade or trade.status not in ("open", "fok_killed"):
            return
        is_fok = trade.status == "fok_killed"
        await asyncio.sleep(delay)

        # Poll PM up to 60s for settlement
        pm_resolved_up: bool | None = None
        if market:
            for _ in range(12):
                p = self._ws.get_price(market.up_token.token_id)
                if p is not None:
                    if p >= 0.95:   pm_resolved_up = True;  break
                    elif p <= 0.05: pm_resolved_up = False; break
                await asyncio.sleep(5.0)

        coin_resolved_up: bool | None = None
        if coin_close is not None and coin_open is not None:
            diff = coin_close - coin_open
            if diff > 0:   coin_resolved_up = True
            elif diff < 0: coin_resolved_up = False

        if pm_resolved_up is not None:
            resolved_up = pm_resolved_up
        elif coin_resolved_up is not None:
            log.warning("[MODEL] RESOLVE %s  PM ambiguous — using coin direction", self.asset)
            resolved_up = coin_resolved_up
        else:
            trade.status = "unresolved"
            await asyncio.to_thread(_write_model_trade, trade, self._model_trades_csv)
            return

        win = (resolved_up and trade.side.upper() == "UP") or (not resolved_up and trade.side.upper() == "DOWN")
        payout = trade.fill_size if win else 0.0
        pnl = payout - trade.fill_usdc
        trade.resolution = 1.0 if resolved_up else 0.0
        trade.pnl_usdc   = round(pnl, 4)
        if is_fok:
            trade.status = "fok_won" if win else "fok_lost"
        else:
            trade.status = "won" if win else "lost"
        log.info("[MODEL] RESOLVED %s %s  predicted=%.1f%%  fill=%.3f  PnL=$%.4f  %s",
                 self.asset, trade.side, trade.predicted_win * 100, trade.fill_price, pnl,
                 trade.status.upper())
        await asyncio.to_thread(_write_model_trade, trade, self._model_trades_csv)

    # ── Binance REST fallback (window open price only) ────────────────────────

    async def _fetch_coin_price_at(self, ts: int) -> float | None:
        """
        Fallback: fetch the coin open price at a specific Unix timestamp using
        Binance 1s klines REST API. Only called when the WebSocket stream is stale
        at window start.
        """
        try:
            resp = await self._http.get(
                BINANCE_KLINES_URL,
                params={
                    "symbol":    self._symbol,
                    "interval":  "1s",
                    "startTime": ts * 1000,
                    "limit":     1,
                },
                timeout=5.0,
            )
            resp.raise_for_status()
            rows = resp.json()
            if rows:
                return float(rows[0][1])  # open price of that 1s candle
        except Exception as exc:
            log.warning("Binance kline fetch failed for ts=%d: %s", ts, exc)

        log.error("REST kline fallback also failed — cannot determine window open price")
        return None

    def _compute_features(self) -> dict:
        """Compute model features from current price history. Mirrors build_window_features in train_model.py."""
        now   = time.time()
        p0    = self._coin_current
        sigma = self.sigma_value
        if p0 is None or not sigma:
            return {}

        def _px_at(offset: float) -> float | None:
            """Last price at or before now-offset seconds (matches 1s candle close lookback)."""
            target = now - offset
            result = None
            for ts, px in self._price_history:
                if ts <= target:
                    result = px
            return result

        p2  = _px_at(2)
        p4  = _px_at(4)
        p5  = _px_at(5)
        p10 = _px_at(10)

        def _vel(pa: float | None, pb: float | None) -> float | None:
            return (pa - pb) / sigma if pa is not None and pb is not None else None  # type: ignore[operator]

        vel_2s  = _vel(p0, p2)
        vel_5s  = _vel(p0, p5)
        vel_10s = _vel(p0, p10)

        # acc_4s  = _acc(2): (p0 - 2*p2 + p4) / sigma
        acc_4s: float | None = None
        if p0 is not None and p2 is not None and p4 is not None:
            acc_4s = (p0 - 2 * p2 + p4) / sigma  # type: ignore[operator]

        # acc_10s = _acc(5): (p0 - 2*p5 + p10) / sigma
        acc_10s: float | None = None
        if p0 is not None and p5 is not None and p10 is not None:
            acc_10s = (p0 - 2 * p5 + p10) / sigma  # type: ignore[operator]

        vel_ratio: float | None = None
        if vel_2s is not None and vel_10s is not None and vel_10s != 0.0:
            vel_ratio = abs(vel_2s) / abs(vel_10s)

        vel_decay: float | None = None
        if vel_2s is not None and vel_10s is not None:
            vel_decay = abs(vel_10s) - abs(vel_2s)

        def _r(v: float | None, dp: int = 4) -> float | None:
            return round(v, dp) if v is not None else None

        coin_open   = self._coin_open or p0
        move_sigmas = (p0 - coin_open) / sigma

        elapsed_second = int(now - self._window_start)
        vol_10s_log    = self._coin_stream.get_vol_10s_log()

        _hour_utc = datetime.fromtimestamp(self._window_start, tz=timezone.utc).hour
        _hour_rad = _hour_utc * (2 * 3.141592653589793 / 24)

        feats: dict[str, float | None] = {
            "move_sigmas":    _r(move_sigmas, 4),
            "elapsed_second": elapsed_second,
            "hour_sin":       round(math.sin(_hour_rad), 8),
            "hour_cos":       round(math.cos(_hour_rad), 8),
            "vel_2s":         _r(vel_2s),
            "vel_5s":         _r(vel_5s),
            "vel_10s":        _r(vel_10s),
            "acc_4s":         _r(acc_4s),
            "acc_10s":        _r(acc_10s),
            "vel_ratio":      _r(vel_ratio),
            "vel_decay":      _r(vel_decay),
            "vol_10s_log":    _r(vol_10s_log),
            "move_x_elapsed": _r(move_sigmas * elapsed_second),
            "move_x_vol":     _r(move_sigmas * vol_10s_log) if vol_10s_log is not None else None,
        }

        return feats

    def _run_inference(self, features: dict) -> float | None:
        """Synchronous sklearn inference — call via asyncio.to_thread."""
        if self._model is None or not features:
            return None
        try:
            feat_order = self._model["features"]
            # Preserve training feature names to avoid sklearn/lightgbm name warnings.
            X = pd.DataFrame(
                [[features.get(f) for f in feat_order]],
                columns=feat_order,
                dtype=float,
            )
            return float(self._model["pipe"].predict_proba(X)[0, 1])
        except Exception as exc:
            log.warning("Inference failed: %s", exc)
            return None

    async def _status_loop(self) -> None:
        while True:
            market = self._market
            up_price = dn_price = up_ask = dn_ask = None
            if market:
                up_price = self._ws.get_price(market.up_token.token_id)
                dn_price = self._ws.get_price(market.down_token.token_id)
                up_ask   = self._ws.get_ask(market.up_token.token_id)
                dn_ask   = self._ws.get_ask(market.down_token.token_id)

            features      = self._compute_features()
            predicted_win: float | None = None
            inference_ms: float | None = None
            try:
                t0 = time.perf_counter()
                predicted_win = await asyncio.to_thread(self._run_inference, features)
                inference_ms = (time.perf_counter() - t0) * 1000.0
                # if self._model is not None and features:
                #     log.info(
                #         "Model inference latency: %.2f ms | predicted_win=%s",
                #         inference_ms,
                #         f"{predicted_win:.4f}" if predicted_win is not None else "None",
                #     )
            except Exception as exc:
                log.warning("Inference failed: %s", exc)

            # Model trade: fire when edge >= threshold
            # once-per-window (cooldown=0) or every N seconds (cooldown=N)
            try:
                _cd = self._model_multi_trade_cooldown
                _model_gate = (
                    self._model_last_fire_ts is None
                    if _cd == 0
                    else (self._model_last_fire_ts is None or
                          time.time() - self._model_last_fire_ts >= _cd)
                )
                # Check max trades per window limit
                _max_trades = self._model_max_trades_per_window
                _trades_limit_ok = (_max_trades == 0 or len(self._model_trades) < _max_trades)

                if (
                    predicted_win is not None
                    and _model_gate
                    and _trades_limit_ok
                    and market is not None
                    and self._coin_open is not None
                    and self._coin_current is not None
                ):
                    up_ask_now = self._ws.get_ask(market.up_token.token_id)
                    dn_ask_now = self._ws.get_ask(market.down_token.token_id)
                    up_edge = (predicted_win - up_ask_now) if up_ask_now is not None else None
                    dn_edge = ((1.0 - predicted_win) - dn_ask_now) if dn_ask_now is not None else None

                    fire_side: str | None = None
                    fire_ask:  float | None = None
                    fire_edge: float | None = None
                    fire_token: str | None = None

                    elapsed_now = int(time.time() - self._window_start)
                    in_model_window = (
                        self._model_window_start <= elapsed_now <= self._model_window_end
                    )

                    if in_model_window and up_edge is not None and up_edge >= self._MODEL_EDGE_THRESHOLD:
                        if dn_edge is None or up_edge >= dn_edge:
                            fire_side, fire_ask, fire_edge, fire_token = (
                                "UP", up_ask_now, up_edge, market.up_token.token_id
                            )
                    if in_model_window and dn_edge is not None and dn_edge >= self._MODEL_EDGE_THRESHOLD:
                        if fire_side is None or dn_edge > (fire_edge or 0):
                            fire_side, fire_ask, fire_edge, fire_token = (
                                "DOWN", dn_ask_now, dn_edge, market.down_token.token_id
                            )

                    if (
                        fire_side is not None and fire_ask is not None
                        and fire_token is not None and fire_edge is not None
                    ):
                        self._model_last_fire_ts = time.time()
                        asyncio.create_task(self._execute_model_buy(
                            side=fire_side,
                            token_id=fire_token,
                            trigger_ask=fire_ask,
                            edge=fire_edge,
                            predicted_win=predicted_win,
                            elapsed=int(time.time() - self._window_start),
                        ))
            except Exception as exc:
                log.warning("Model fire logic failed: %s", exc)

            try:
                payload = {
                    "updated_at":     time.time(),
                    "window_start":   self._window_start,
                    "window_end":     self._window_end,
                    "elapsed_secs":   round(time.time() - self._window_start),
                    "remaining_secs": round(self._window_end - time.time()),
                    "asset":          self.asset,
                    "sigma_value":    self.sigma_value,
                    "sigma_entry":    self.sigma_entry,
                    "max_pm_price":   self.max_pm_price,
                    "direction":      self.direction,
                    "coin_open":      self._coin_open,
                    "coin_current":   self._coin_current,
                    "coin_move":      (self._coin_current or self._coin_open or 0) - (self._coin_open or 0),
                    "threshold_move": self.sigma_entry * self.sigma_value,
                    "up_price":       up_price,
                    "down_price":     dn_price,
                    "up_ask":         up_ask,
                    "down_ask":       dn_ask,
                    "filled":         self._filled,
                    "trade":          asdict(self._trade) if self._trade else None,
                    "dry_run":        self.dry_run,
                    "features":       {k: (float(v) if v is not None else None) for k, v in features.items()},
                    "predicted_win":        predicted_win,
                    "MODEL_EDGE_THRESHOLD": self._MODEL_EDGE_THRESHOLD,
                    "model_trade":          asdict(self._model_trade) if self._model_trade else None,
                    "model_trades":         [asdict(t) for t in self._model_trades],
                    "momentum_enabled":     self._momentum_enabled,
                    "model_enabled":        self._model_enabled,
                    "price_history":        [px for _, px in self._price_history],
                }
                await asyncio.to_thread(_write_status, payload, self._status_json)
            except Exception as exc:
                log.warning("Status write failed: %s", exc, exc_info=True)
            await asyncio.sleep(0.25)


