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
import warnings
import yaml
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
from skeptic.clients import gamma
from skeptic.clients.ws import MarketChannel
from skeptic.models.market import Market
from skeptic.utils.kelly import kelly_usdc, MOMENTUM_EDGE_THRESHOLD, KELLY_MAX_USDC, imbalance_kelly_multiplier
from skeptic.utils.time import current_window_start, next_window_start, sleep_until

log = logging.getLogger("momentum_buy")
logging.getLogger("httpx").setLevel(logging.WARNING)

LIVE_DIR = os.path.join("data", "live")

BINANCE_KLINES_URL  = "https://api.binance.com/api/v3/klines"
BINANCE_WS_COMBINED = "wss://stream.binance.com:9443/stream?streams={streams}"

class BinanceCoinStream:
    """
    Subscribes to Binance's aggTrade + depth5 WebSocket streams for a single symbol.
    Caches the latest trade price and top-5 orderbook bid/ask volumes.
    Reconnects automatically on any error.

    Use get_price() to read the latest price (returns None until first message).
    Use get_coin_ob_imbalance() to read bid/(bid+ask) ratio from top-5 depth.
    """

    def __init__(self, symbol: str) -> None:
        self._symbol       = symbol.lower()
        self._price:       float | None = None
        self._last_update: float = 0.0
        self._event        = asyncio.Event()
        # (timestamp, quantity) for rolling volume sums — matches 1s OHLCV volume column
        self._vol_history: deque[tuple[float, float]] = deque(maxlen=500)
        # Top-5 orderbook bid/ask volume totals (updated from @depth5@100ms)
        self._coin_bid_vol: float = 0.0
        self._coin_ask_vol: float = 0.0
        # Rolling (timestamp, imbalance) history — ~100ms updates, 200 entries ≈ 20s
        self._ob_history: deque[tuple[float, float]] = deque(maxlen=200)

    def get_price(self) -> float | None:
        return self._price

    def get_vol_10s_log(self) -> float:
        """log1p of total trade quantity in the last 10 seconds — matches vol_10s_log training feature."""
        cutoff = time.time() - 10.0
        total  = sum(q for ts, q in self._vol_history if ts >= cutoff)
        return math.log1p(total)

    def get_coin_ob_imbalance(self) -> float | None:
        """bid_vol / (bid_vol + ask_vol) from top-5 depth levels. None if no depth data yet."""
        total = self._coin_bid_vol + self._coin_ask_vol
        if total <= 0:
            return None
        return self._coin_bid_vol / total

    def get_ob_metrics(self) -> tuple[float | None, float | None, float | None]:
        """
        Returns (snapshot, mean_5s, trend) at the current moment.

        snapshot — imbalance of the most recent depth update
        mean_5s  — mean imbalance over the last 5 seconds (~50 depth updates)
        trend    — snapshot − mean_5s (positive = book getting more bid-heavy into entry)

        All three are None if fewer than 3 depth samples exist in the last 5 seconds.
        """
        if not self._ob_history:
            return None, None, None
        now     = time.time()
        snap    = self._ob_history[-1][1]
        window  = [imb for ts, imb in self._ob_history if ts >= now - 5.0]
        if len(window) < 3:
            return snap, None, None
        mean_5s = sum(window) / len(window)
        trend   = snap - mean_5s
        return snap, mean_5s, trend

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
        sym     = self._symbol
        streams = f"{sym}@aggTrade/{sym}@depth5@100ms"
        url     = BINANCE_WS_COMBINED.format(streams=streams)
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=20
                ) as ws:
                    backoff = 1.0
                    async for raw in ws:
                        envelope = json.loads(raw)
                        # Combined stream wraps each message: {"stream": "...", "data": {...}}
                        stream_name = envelope.get("stream", "")
                        msg         = envelope.get("data", envelope)
                        if "@aggTrade" in stream_name:
                            self._price       = float(msg["p"])
                            self._last_update = time.time()
                            self._vol_history.append((self._last_update, float(msg["q"])))
                            self._event.set()
                        elif "@depth5" in stream_name:
                            bids = msg.get("bids") or []
                            asks = msg.get("asks") or []
                            self._coin_bid_vol = sum(float(b[1]) for b in bids)
                            self._coin_ask_vol = sum(float(a[1]) for a in asks)
                            _total = self._coin_bid_vol + self._coin_ask_vol
                            if _total > 0:
                                self._ob_history.append((time.time(), self._coin_bid_vol / _total))
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
    "coin_ob_imbalance", "coin_ob_imb_5s", "coin_ob_trend",
]

SLIPPAGE = 0.075
FEE_GAMMA = 0.072  # Polymarket CLOB fee coefficient: fee = C × γ × p × (1 − p)
BUY_FEE_RATE = FEE_GAMMA  # deprecated alias


def _calc_fee_usdc(fill_size: float, fill_price: float) -> float:
    """CLOB fee in USDC: C × FEE_GAMMA × p × (1 − p)."""
    return fill_size * FEE_GAMMA * fill_price * (1.0 - fill_price)
EWMA_LAMBDA = 0.95             # decay factor for walk-forward sigma estimate
SESSION_LOSS_LIMIT = 10.0      # dollars: disable a strategy if it loses this much from its session peak
LOSS_LIMIT_COOLDOWN_WINDOWS = 6  # dry-run windows (~30 min) before checking paper PnL to re-enable
BAD_TRADE_CSV_CAP = 3  # stop logging FOK/error/OB-filtered trades to CSV after this many per window

MODEL_TRADE_FIELDS = [
    "ts", "asset", "side", "token_id",
    "fill_price", "fill_size", "fill_usdc", "fee_usdc",
    "predicted_win", "edge",
    "elapsed_second", "coin_open", "coin_trigger", "coin_move",
    "window_start_ts", "window_end_ts",
    "sign_ms", "post_ms", "order_ms",
    "resolution", "pnl_usdc", "status", "order_id", "slippage",
    "coin_ob_imbalance", "coin_ob_imb_5s", "coin_ob_trend",
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
    coin_ob_imbalance: float | None = None
    coin_ob_imb_5s: float | None = None
    coin_ob_trend: float | None = None


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
    sign_ms: float | None = None   # ms to sign the order (None if dry-run)
    post_ms: float | None = None   # ms for the HTTP POST to return (None on fresh-sign path)
    order_ms: float | None = None  # total ms from _execute_buy entry to order confirmed (None if dry-run)
    resolution: float | None = None
    pnl_usdc: float | None = None
    status: str = "open"
    order_id: str = ""
    coin_ob_imbalance: float | None = None
    coin_ob_imb_5s: float | None = None
    coin_ob_trend: float | None = None


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
        name: str = "momentum",
        model_cfg: dict | None = None,
        momentum_cfg: dict | None = None,
        config_path: str = "config/assets.yaml",
    ) -> None:
        assert direction in ("up", "down", "both"), "direction must be 'up', 'down', or 'both'"
        self.asset        = asset
        self.sigma_value  = sigma_value
        self.sigma_entry  = sigma_entry
        self.max_pm_price = max_pm_price
        self.direction    = direction
        self.wallet_pct   = wallet_pct
        self.fixed_usdc      = fixed_usdc
        self._config_path = config_path
        _mom = momentum_cfg or {}
        self._momentum_enabled: bool              = bool(_mom.get("enabled", True))
        self._momentum_multi_trade_cooldown: int  = int(_mom.get("multi_trade_cooldown", 0))
        self._momentum_max_trades_per_window: int = int(_mom.get("max_trades_per_window", 0))
        _mfi = _mom.get("min_fav_imbalance")
        self._momentum_min_fav_imbalance: float | None = float(_mfi) if _mfi is not None else None
        _mfi_mean = _mom.get("min_fav_imbalance_mean")
        self._momentum_min_fav_imbalance_mean: float | None = float(_mfi_mean) if _mfi_mean is not None else None
        _mc = model_cfg or {}
        self._model_cfg = _mc
        self._model_enabled:         bool        = bool(_mc.get("enabled", True))
        self._model_window_start:    int         = int(_mc.get("window_start", 0))
        self._model_window_end:      int         = int(_mc.get("window_end",   300))
        self._MODEL_EDGE_THRESHOLD:  float       = float(_mc.get("edge_threshold", .20))
        self._model_fixed_usdc:         float | None = _mc.get("fixed_usdc")

        self._model_multi_trade_cooldown: int        = int(_mc.get("multi_trade_cooldown", 0))
        self._model_max_trades_per_window: int       = int(_mc.get("max_trades_per_window", 3))
        self._model_max_slippage:          float     = float(_mc.get("max_slippage", 0.10))
        _mdl_mfi = _mc.get("min_fav_imbalance")
        self._model_min_fav_imbalance: float | None  = float(_mdl_mfi) if _mdl_mfi is not None else None
        _mdl_mfi_mean = _mc.get("min_fav_imbalance_mean")
        self._model_min_fav_imbalance_mean: float | None = float(_mdl_mfi_mean) if _mdl_mfi_mean is not None else None

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
        self._model_path = os.path.join("data", "models", f"{asset.lower()}.joblib")
        self._model: dict[str, Any] | None = None
        self._model_mtime: float = 0.0
        if os.path.exists(self._model_path):
            try:
                self._model = joblib.load(self._model_path)
                self._model_mtime = os.path.getmtime(self._model_path)
            except Exception as exc:
                log.warning("Could not load model for %s: %s", asset, exc)

        # EWMA sigma state — updated after every completed window
        # Initialize from config sigma^2 so the first live update treats config sigma as prior state.
        self._sigma_initial: float = sigma_value
        self._ewma_var: float = float(sigma_value) ** 2

        # CLOB availability backoff — set when 425 persists after retry
        self._clob_backoff_until: float = 0.0

        # Session-level PnL accumulators for the loss circuit breaker.
        # Only real fills count (DRY_RUN and FOK_KILLED trades are excluded).
        self._session_momentum_pnl:      float = 0.0
        self._session_model_pnl:         float = 0.0
        # High-water marks — loss limit triggers on drawdown from peak, not from zero.
        self._session_momentum_peak_pnl: float = 0.0
        self._session_model_peak_pnl:    float = 0.0
        # Cooldown state — tracks dry-run windows and paper PnL while waiting to re-enable.
        self._momentum_in_cooldown:      bool  = False
        self._model_in_cooldown:         bool  = False
        self._momentum_cooldown_count:   int   = 0   # windows elapsed in current cooldown cycle
        self._model_cooldown_count:      int   = 0
        self._momentum_cooldown_pnl:     float = 0.0  # paper PnL accumulated this cooldown cycle
        self._model_cooldown_pnl:        float = 0.0

        # Per-window state
        self._market:   Market | None = None
        self._filled:   bool = False
        self._momentum_trade_count:  int          = 0
        self._momentum_bad_count:    int          = 0  # FOK + errors + OB-filtered this window
        self._momentum_last_fire_ts: float | None = None
        self._trade:      MomentumTrade | None = None
        self._momentum_trades: list[MomentumTrade] = []
        self._tp_sell_order_id: str | None = None   # open take-profit sell order
        self._window_start: int = 0
        self._window_end:   int = 0
        self._coin_open:    float | None = None
        self._coin_current: float | None = None
        self._position_usdc: float = 0.0
        # (timestamp, price) ring buffer — used to compute velocity/acceleration features
        self._price_history: deque[tuple[float, float]] = deque(maxlen=60)
        # Cached hour features — computed once per window, reused for all inferences
        self._cached_hour_sin: float = 0.0
        self._cached_hour_cos: float = 0.0
        # Cached book data — fetched once per status loop
        self._cached_up_book: Any = None
        self._cached_dn_book: Any = None
        # Model dry-run state — separate from live trading state
        self._model_last_fire_ts: float | None = None   # None = not fired this window
        self._model_trades: list[ModelTrade]   = []     # all trades this window
        self._model_trade:  ModelTrade | None  = None   # most recent trade (for status display)
        self._model_bad_count:   int           = 0      # FOK + errors + OB-filtered this window

    # ── Public ────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        _ensure_live_dir(self._trades_csv)
        _ensure_model_csv(self._model_trades_csv)
        _model_csv_ensured.add(self._model_trades_csv)
        async with httpx.AsyncClient(
            http2=True,
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=12,
                keepalive_expiry=None,   # never proactively expire idle connections
            ),
        ) as http:
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
                asyncio.sleep(1),
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
            self._momentum_trade_count  = 0
            self._momentum_last_fire_ts = None
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
        self._tp_sell_order_id = None
        self._coin_open     = None
        self._coin_current  = None
        self._window_start  = window_start
        self._window_end    = window_end
        self._price_history.clear()
        self._momentum_trade_count:  int          = 0
        self._momentum_bad_count:    int          = 0
        self._momentum_last_fire_ts: float | None = None
        self._momentum_trades        = []
        self._model_last_fire_ts = None
        self._model_trades       = []
        self._model_trade        = None
        self._model_bad_count    = 0
        # Clear cached book data at window start
        self._cached_up_book = None
        self._cached_dn_book = None

        # Pre-compute hour features once per window
        _hour_utc = datetime.fromtimestamp(window_start, tz=timezone.utc).hour
        _hour_rad = _hour_utc * (2 * 3.141592653589793 / 24)
        self._cached_hour_sin = round(math.sin(_hour_rad), 8)
        self._cached_hour_cos = round(math.cos(_hour_rad), 8)

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
        # Hot-reload enabled flags from config each window so changes take effect without restart
        try:
            with open(self._config_path) as _f:
                _cfg = yaml.safe_load(_f) or {}
            _global_mom = _cfg.get("MOMENTUM", {})
            _global_mdl = _cfg.get("MODEL", {})
            _asset_cfg  = _cfg.get(self.asset.upper(), {})
            _mom_cfg    = {**_global_mom, **_asset_cfg.get("momentum", {})}
            _mdl_cfg    = {**_global_mdl, **_asset_cfg.get("model", {})}
            self._momentum_enabled = bool(_mom_cfg.get("enabled", True))
            self._model_enabled    = bool(_mdl_cfg.get("enabled", True))
            _mfi = _mom_cfg.get("min_fav_imbalance")
            self._momentum_min_fav_imbalance = float(_mfi) if _mfi is not None else None
            _mfi_mean = _mom_cfg.get("min_fav_imbalance_mean")
            self._momentum_min_fav_imbalance_mean = float(_mfi_mean) if _mfi_mean is not None else None
            _mdl_mfi = _mdl_cfg.get("min_fav_imbalance")
            self._model_min_fav_imbalance    = float(_mdl_mfi) if _mdl_mfi is not None else None
            _mdl_mfi_mean = _mdl_cfg.get("min_fav_imbalance_mean")
            self._model_min_fav_imbalance_mean = float(_mdl_mfi_mean) if _mdl_mfi_mean is not None else None
        except Exception as exc:
            log.warning("%s config reload failed: %s — keeping existing enabled flags", self.asset, exc)

        # Hot-reload model if the .joblib file has been updated since last load
        try:
            if os.path.exists(self._model_path):
                _mtime = os.path.getmtime(self._model_path)
                if _mtime > self._model_mtime:
                    self._model = joblib.load(self._model_path)
                    self._model_mtime = _mtime
                    log.info("%s model hot-reloaded from %s", self.asset, self._model_path)
        except Exception as exc:
            log.warning("%s model reload failed: %s — keeping existing model", self.asset, exc)

        # Session loss circuit breaker takes precedence over config reload.
        # Cooldown mode: run dry for LOSS_LIMIT_COOLDOWN_WINDOWS windows, then re-enable only
        # if the paper PnL during that cycle was profitable. Otherwise, reset and wait again.
        if self._momentum_in_cooldown:
            self._momentum_cooldown_count += 1
            if self._momentum_cooldown_count >= LOSS_LIMIT_COOLDOWN_WINDOWS:
                if self._momentum_cooldown_pnl > 0:
                    log.info(
                        "MOMENTUM cooldown complete  %s  paper_pnl=$%.4f — profitable, re-enabling per config",
                        self.asset, self._momentum_cooldown_pnl,
                    )
                    self._session_momentum_pnl      = 0.0
                    self._session_momentum_peak_pnl = 0.0
                    self._momentum_in_cooldown       = False
                    self._momentum_cooldown_count    = 0
                    self._momentum_cooldown_pnl      = 0.0
                    # _momentum_enabled already set by config reload above — let it stand
                else:
                    log.info(
                        "MOMENTUM cooldown  %s  paper_pnl=$%.4f not profitable — extending cooldown (%d more windows)",
                        self.asset, self._momentum_cooldown_pnl, LOSS_LIMIT_COOLDOWN_WINDOWS,
                    )
                    self._momentum_cooldown_count = 0
                    self._momentum_cooldown_pnl   = 0.0
                    self._momentum_enabled         = False
            else:
                self._momentum_enabled = False
                log.info(
                    "MOMENTUM cooldown  %s  window %d/%d  paper_pnl=$%.4f — staying in dry run",
                    self.asset, self._momentum_cooldown_count, LOSS_LIMIT_COOLDOWN_WINDOWS,
                    self._momentum_cooldown_pnl,
                )
        elif self._session_momentum_pnl <= self._session_momentum_peak_pnl - SESSION_LOSS_LIMIT:
            self._momentum_enabled = False
            log.info(
                "MOMENTUM loss limit active  %s  session_pnl=$%.4f  peak=$%.4f — staying in dry run",
                self.asset, self._session_momentum_pnl, self._session_momentum_peak_pnl,
            )

        if self._model_in_cooldown:
            self._model_cooldown_count += 1
            if self._model_cooldown_count >= LOSS_LIMIT_COOLDOWN_WINDOWS:
                if self._model_cooldown_pnl > 0:
                    log.info(
                        "[MODEL] cooldown complete  %s  paper_pnl=$%.4f — profitable, re-enabling per config",
                        self.asset, self._model_cooldown_pnl,
                    )
                    self._session_model_pnl      = 0.0
                    self._session_model_peak_pnl = 0.0
                    self._model_in_cooldown       = False
                    self._model_cooldown_count    = 0
                    self._model_cooldown_pnl      = 0.0
                else:
                    log.info(
                        "[MODEL] cooldown  %s  paper_pnl=$%.4f not profitable — extending cooldown (%d more windows)",
                        self.asset, self._model_cooldown_pnl, LOSS_LIMIT_COOLDOWN_WINDOWS,
                    )
                    self._model_cooldown_count = 0
                    self._model_cooldown_pnl   = 0.0
                    self._model_enabled         = False
            else:
                self._model_enabled = False
                log.info(
                    "[MODEL] cooldown  %s  window %d/%d  paper_pnl=$%.4f — staying in dry run",
                    self.asset, self._model_cooldown_count, LOSS_LIMIT_COOLDOWN_WINDOWS,
                    self._model_cooldown_pnl,
                )
        elif self._session_model_pnl <= self._session_model_peak_pnl - SESSION_LOSS_LIMIT:
            self._model_enabled = False
            log.info(
                "[MODEL] loss limit active  %s  session_pnl=$%.4f  peak=$%.4f — staying in dry run",
                self.asset, self._session_model_pnl, self._session_model_peak_pnl,
            )

        balance = await asyncio.to_thread(clob_client.get_usdc_balance, self._clob)
        if balance < 2.0:
            log.warning(
                "%s balance $%.4f < $2.00 — disabling momentum and model for this window",
                self.asset, balance,
            )
            self._momentum_enabled = False
            self._model_enabled    = False
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

        # Subscribe to WebSocket and warm CLOB connection
        all_tokens = [market.up_token.token_id, market.down_token.token_id]
        await self._ws.subscribe(*all_tokens)
        await self._ws.reconnect()
        # Warm the TCP+TLS connection to the CLOB so the trade POST reuses an open socket
        assert self._http is not None
        await clob_client.warm_connection_async(self._http, self._clob)
        # Wait for first PM price instead of a fixed sleep
        for _ in range(10):
            if any(self._ws.get_ask(t) is not None for t in all_tokens):
                break
            await asyncio.sleep(0.1)

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
        momentum_trades_snap = list(self._momentum_trades)
        model_trades_snap    = list(self._model_trades)
        market_snap          = self._market
        coin_open_snap       = self._coin_open
        coin_close_snap      = self._coin_current

        # Cancel any open take-profit sell before resolution
        if self._tp_sell_order_id:
            tp_oid = self._tp_sell_order_id
            self._tp_sell_order_id = None
            await asyncio.to_thread(clob_client.cancel_order, self._clob, tp_oid)

        for _mom in momentum_trades_snap:
            if _mom.status in ("open", "fok_killed"):
                asyncio.create_task(self._resolve_bg(
                    _mom, market_snap, coin_open_snap, coin_close_snap, delay=8.0
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

    MOMENTUM_TICK_SECS = 0.5  # check interval for the momentum threshold loop

    async def _momentum_watch_loop(self) -> None:
        """
        Check threshold twice per second using the latest aggTrade price.

        aggTrades arrive many times per second; we accumulate them via
        BinanceCoinStream and sample the latest price at each tick.

        In dry-run mode, logs a per-tick DEBUG line and a 30-second INFO heartbeat
        so the stream health can be verified without placing real orders.
        """
        threshold  = self.sigma_entry * self.sigma_value
        _skipped: set[str] = set()
        _stale_warns    = 0

        while True:
            # Sleep until the next 0.5-second boundary, then sample the latest price.
            now = time.time()
            await asyncio.sleep(self.MOMENTUM_TICK_SECS - (now % self.MOMENTUM_TICK_SECS))

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

            # ── Multi-trade gate ──────────────────────────────────────────────
            _cd  = self._momentum_multi_trade_cooldown
            _max = self._momentum_max_trades_per_window
            if _cd == 0:
                # Classic single-entry: block after first fill
                if self._filled:
                    continue
            else:
                # Cooldown mode: respect max-trades cap and per-trade cooldown
                if _max > 0 and self._momentum_trade_count >= _max:
                    continue
                if (self._momentum_last_fire_ts is not None
                        and time.time() - self._momentum_last_fire_ts < _cd):
                    continue

            move   = coin_price - self._coin_open
            sigmas = move / self.sigma_value if self.sigma_value else 0.0
            market = self._market
            if market is None:
                continue

            if self.direction in ("up", "both") and move >= threshold:
                pm_price = self._ws.get_ask(market.up_token.token_id)
                if pm_price is not None and pm_price <= self.max_pm_price:
                    log.info(
                        "TRIGGER UP  coin_open=%g  coin_now=%g  move=%+g (%.2fσ)  pm_ask=%.4f%s",
                        self._coin_open, coin_price, move, sigmas, pm_price,
                        "  [MOMENTUM DISABLED]" if not self._momentum_enabled else "",
                    )
                    self._filled = True
                    self._momentum_trade_count  += 1
                    self._momentum_last_fire_ts  = time.time()
                    _skipped.discard("up")
                    await self._execute_buy("UP", market.up_token.token_id,
                                            pm_price, coin_price, move)
                elif pm_price is not None:
                    _skipped.add("up")
            else:
                _skipped.discard("up")

            if self.direction in ("down", "both") and move <= -threshold:
                pm_price = self._ws.get_ask(market.down_token.token_id)
                if pm_price is not None and pm_price <= self.max_pm_price:
                    log.info(
                        "TRIGGER DOWN  coin_open=%g  coin_now=%g  move=%+g (%.2fσ)  pm_ask=%.4f%s",
                        self._coin_open, coin_price, move, sigmas, pm_price,
                        "  [MOMENTUM DISABLED]" if not self._momentum_enabled else "",
                    )
                    self._filled = True
                    self._momentum_trade_count  += 1
                    self._momentum_last_fire_ts  = time.time()
                    _skipped.discard("down")
                    await self._execute_buy("DOWN", market.down_token.token_id,
                                            pm_price, coin_price, move)
                elif pm_price is not None:
                    _skipped.add("down")
            else:
                _skipped.discard("down")

    # ── Execution ─────────────────────────────────────────────────────────────

    def _kelly_stake(self, edge: float, imbalance: float | None = None) -> float:
        """Scale momentum stake from fixed_usdc up to KELLY_MAX_USDC, adjusted for orderbook imbalance."""
        fixed = self.fixed_usdc if self.fixed_usdc is not None else self._position_usdc
        stake = kelly_usdc(edge=edge, edge_threshold=MOMENTUM_EDGE_THRESHOLD, fixed_usdc=fixed)
        if imbalance is not None:
            stake = round(stake * imbalance_kelly_multiplier(imbalance), 4)
            stake = max(fixed, min(KELLY_MAX_USDC, stake))
            log.info("Kelly $%.2f USDC  (imbalance=%.4f)", stake, imbalance)
        return stake

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

        # Orderbook imbalance: bid_volume / (bid_volume + ask_volume) on the traded token
        # Use cached book if available, otherwise fetch
        if side.upper() == "UP" and self._cached_up_book is not None:
            _book = self._cached_up_book
        elif side.upper() == "DOWN" and self._cached_dn_book is not None:
            _book = self._cached_dn_book
        else:
            _book = self._ws.price_cache.get_book(token_id)
        
        imbalance: float | None = None
        if _book is not None:
            _total_vol = _book.bid_volume + _book.ask_volume
            if _total_vol > 0:
                imbalance = _book.bid_volume / _total_vol

        # Kelly sizing: edge = max_pm_price - ask, scaled by imbalance
        stake    = self._kelly_stake(self.max_pm_price - trigger_pm_price, imbalance=imbalance)
        est_size = round(stake / trigger_pm_price, 2)
        fee_usdc = round(_calc_fee_usdc(est_size, trigger_pm_price), 4)

        # Coin orderbook metrics at trigger time
        _ob_snap, _ob_mean_5s, _ob_trend = self._coin_stream.get_ob_metrics()

        # Direction-adjusted favorable imbalance: bid-heavy (high) is good for UP,
        # ask-heavy (low raw = high favorable) is good for DOWN.
        _fav_imb: float | None = None
        if _ob_snap is not None:
            _fav_imb = _ob_snap if side.upper() == "UP" else 1.0 - _ob_snap
        _fav_imb_mean: float | None = None
        if _ob_mean_5s is not None:
            _fav_imb_mean = _ob_mean_5s if side.upper() == "UP" else 1.0 - _ob_mean_5s
        _snap_filtered = (
            self._momentum_min_fav_imbalance is not None
            and _fav_imb is not None
            and _fav_imb < self._momentum_min_fav_imbalance
        )
        _mean_filtered = (
            self._momentum_min_fav_imbalance_mean is not None
            and _fav_imb_mean is not None
            and _fav_imb_mean < self._momentum_min_fav_imbalance_mean
        )
        _ob_filtered = _snap_filtered or _mean_filtered
        _ob_filter_str = ""
        if _ob_filtered:
            _snap_str = f"snap={_fav_imb:.3f}<{self._momentum_min_fav_imbalance}" if _snap_filtered else (f"snap={_fav_imb:.3f}" if _fav_imb is not None else "snap=n/a")
            _mean_str = f"  mean={_fav_imb_mean:.3f}<{self._momentum_min_fav_imbalance_mean}" if _mean_filtered else (f"  mean={_fav_imb_mean:.3f}" if _fav_imb_mean is not None else "  mean=n/a")
            _ob_filter_str = f"  {_snap_str}{_mean_str}"

        # Get book once and cache for potential take-profit order
        _book = self._ws.price_cache.get_book(token_id)
        imbalance: float | None = None
        if _book is not None:
            _total_vol = _book.bid_volume + _book.ask_volume
            if _total_vol > 0:
                imbalance = _book.bid_volume / _total_vol

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
            coin_ob_imbalance=round(_ob_snap, 4)    if _ob_snap    is not None else None,
            coin_ob_imb_5s=round(_ob_mean_5s, 4)   if _ob_mean_5s is not None else None,
            coin_ob_trend=round(_ob_trend, 4)       if _ob_trend   is not None else None,
        )
        # Register immediately so the status loop shows this trade while the order is in-flight.
        # The dataclass is mutated in-place as order results arrive below.
        self._trade = trade
        self._momentum_trades.append(trade)

        _imb_str = f"  imbalance={imbalance:.3f}" if imbalance is not None else "  imbalance=n/a"
        _fav_str = _ob_filter_str if _ob_filtered else (f"  fav_imb={_fav_imb:.3f}" if _fav_imb is not None else "  fav_imb=n/a")
        if not self._momentum_enabled or _ob_filtered:
            _reason = "OB FILTER" if _ob_filtered else "MOMENTUM DISABLED"
            _log_fn = log.debug if (_ob_filtered and self._momentum_bad_count >= BAD_TRADE_CSV_CAP) else log.info
            _log_fn(
                "[%s] BUY %s %s  $%.2f USDC  pm=%.4f%s%s",
                _reason, self.asset, side, stake, trigger_pm_price, _imb_str, _fav_str,
            )
            trade.order_id = "DRY_RUN_OB_FILTERED" if _ob_filtered else "DRY_RUN"
            if _ob_filtered:
                self._momentum_trade_count -= 1
                self._filled = False
                self._momentum_bad_count += 1
        elif time.time() < self._clob_backoff_until:
            _remaining = self._clob_backoff_until - time.time()
            log.info("CLOB backoff active %s %s — %.0fs remaining, skipping order", self.asset, side, _remaining)
            trade.order_id = "DRY_RUN"
            self._filled = False
        else:
            try:
                assert self._http is not None
                _mom_retry_delay = 2.0
                order = None
                for _attempt in range(2):
                    try:
                        order, trade.sign_ms, trade.post_ms = await clob_client.sign_and_post_async(
                            self._http, self._clob, token_id, side, stake, price_cap=0.90,
                        )
                        break
                    except Exception as _inner_exc:
                        _exc_str = str(_inner_exc)
                        if _attempt == 0 and "425" in _exc_str:
                            log.warning(
                                "425 service not ready %s %s — retrying in %.0fs",
                                self.asset, side, _mom_retry_delay,
                            )
                            await asyncio.sleep(_mom_retry_delay)
                            continue
                        raise
                assert order is not None

                trade.order_ms   = round((time.perf_counter() - t0) * 1000, 1)
                trade.order_id   = order.order_id
                trade.fill_price = order.price if order.price > 0 else trigger_pm_price
                trade.fill_size  = order.size  if order.size  > 0 else est_size
                notional_usdc    = round(trade.fill_price * trade.fill_size, 4)
                trade.fee_usdc   = round(_calc_fee_usdc(trade.fill_size, trade.fill_price), 4)
                trade.fill_usdc  = round(notional_usdc + trade.fee_usdc, 4)
                trade.slippage   = round(trade.fill_price - trigger_pm_price, 4)
                _fmt_ms = lambda v: f"{v:.0f}ms" if v is not None else "—"
                log.info(
                    "FILLED  %s %s  %.4f shares @ %.4f  ($%.2f incl fee %.4f)"
                    "  slippage=%+.4f  order=%s  total=%s",
                    self.asset, side, trade.fill_size, trade.fill_price,
                    trade.fill_usdc, trade.fee_usdc, trade.slippage, order.order_id[:16],
                    _fmt_ms(trade.order_ms),
                )
            except Exception as exc:
                trade.order_ms = round((time.perf_counter() - t0) * 1000, 1)
                if "425" in str(exc):
                    _backoff = 30.0
                    self._clob_backoff_until = time.time() + _backoff
                    log.warning(
                        "425 persisted after retry %s %s — CLOB backoff %.0fs",
                        self.asset, side, _backoff,
                    )
                    self._filled = False
                    self._momentum_trade_count -= 1
                    self._momentum_last_fire_ts = None
                    self._momentum_trades.remove(trade)
                    self._trade = None
                    return
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
                        trade.fee_usdc   = round(_calc_fee_usdc(trade.fill_size, trade.fill_price), 4)
                        trade.fill_usdc  = round(notional + trade.fee_usdc, 4)
                        trade.slippage   = FOK_SLIPPAGE
                        trade.status     = "fok_killed"
                        trade.order_id   = "FOK_KILLED"
                        self._filled     = False  # allow retry in same window
                        self._momentum_trade_count  -= 1
                        self._momentum_bad_count    += 1
                        log.warning("FOK KILLED %s %s  hypothetical fill=%.4f (+%.2f slip)", self.asset, side, trade.fill_price, FOK_SLIPPAGE)
                    else:
                        log.error("Order failed %s %s: %s", self.asset, side, exc)
                        trade.status = "order_failed"
                        self._filled = False
                        self._momentum_trade_count -= 1
                        self._momentum_bad_count   += 1

        _is_bad_trade = trade.status in ("fok_killed", "order_failed") or "OB_FILTERED" in (trade.order_id or "")
        if not _is_bad_trade or self._momentum_bad_count <= BAD_TRADE_CSV_CAP:
            await asyncio.to_thread(_write_trade, trade, self._trades_csv)

        # Place take-profit sell immediately after fill — sits in book at 0.98
        if trade.status == "open" and trade.fill_size > 0:
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
        
        # Cache resolved value (used in model resolution too)
        resolved_up = resolved_up

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

        # Session loss circuit breaker — only real fills count.
        if not is_fok and trade.order_id and not trade.order_id.startswith("DRY_RUN") and trade.order_id != "FOK_KILLED":
            self._session_momentum_pnl      += pnl
            self._session_momentum_peak_pnl  = max(self._session_momentum_peak_pnl, self._session_momentum_pnl)
            log.info(
                "MOMENTUM session PnL  %s  cumulative=$%.4f  peak=$%.4f",
                self.asset, self._session_momentum_pnl, self._session_momentum_peak_pnl,
            )
            if (self._session_momentum_pnl <= self._session_momentum_peak_pnl - SESSION_LOSS_LIMIT
                    and self._momentum_enabled):
                self._momentum_enabled       = False
                self._momentum_in_cooldown   = True
                self._momentum_cooldown_count = 0
                self._momentum_cooldown_pnl   = 0.0
                log.warning(
                    "MOMENTUM LOSS LIMIT HIT  %s  session_pnl=$%.4f  peak=$%.4f  drawdown=$%.4f"
                    " — entering %d-window cooldown",
                    self.asset, self._session_momentum_pnl, self._session_momentum_peak_pnl,
                    self._session_momentum_pnl - self._session_momentum_peak_pnl,
                    LOSS_LIMIT_COOLDOWN_WINDOWS,
                )
        elif self._momentum_in_cooldown and trade.order_id and trade.order_id.startswith("DRY_RUN"):
            self._momentum_cooldown_pnl += pnl
            log.info(
                "MOMENTUM cooldown paper PnL  %s  +$%.4f  cycle_total=$%.4f",
                self.asset, pnl, self._momentum_cooldown_pnl,
            )

    async def _execute_model_buy(
        self,
        side: str,
        token_id: str,
        trigger_ask: float,
        edge: float,
        predicted_win: float,
        elapsed: int,
    ) -> None:
        if time.time() < self._clob_backoff_until:
            _remaining = self._clob_backoff_until - time.time()
            log.info("[MODEL] CLOB backoff active %s %s — %.0fs remaining, skipping order", self.asset, side, _remaining)
            return
        _trigger_ts = time.perf_counter()
        model_usdc = self._model_kelly_stake(edge)
        coin_move  = (self._coin_current or self._coin_open or 0.0) - (self._coin_open or 0.0)

        _ob_snap, _ob_mean_5s, _ob_trend = self._coin_stream.get_ob_metrics()

        # Direction-adjusted favorable imbalance filter
        _model_fav_imb: float | None = None
        if _ob_snap is not None:
            _model_fav_imb = _ob_snap if side.upper() == "UP" else 1.0 - _ob_snap
        _model_fav_imb_mean: float | None = None
        if _ob_mean_5s is not None:
            _model_fav_imb_mean = _ob_mean_5s if side.upper() == "UP" else 1.0 - _ob_mean_5s
        _model_snap_filtered = (
            self._model_min_fav_imbalance is not None
            and _model_fav_imb is not None
            and _model_fav_imb < self._model_min_fav_imbalance
        )
        _model_mean_filtered = (
            self._model_min_fav_imbalance_mean is not None
            and _model_fav_imb_mean is not None
            and _model_fav_imb_mean < self._model_min_fav_imbalance_mean
        )
        _model_ob_filtered = _model_snap_filtered or _model_mean_filtered
        _model_ob_filter_str = ""
        if _model_ob_filtered:
            _snap_str = f"snap={_model_fav_imb:.3f}<{self._model_min_fav_imbalance}" if _model_snap_filtered else (f"snap={_model_fav_imb:.3f}" if _model_fav_imb is not None else "snap=n/a")
            _mean_str = f"  mean={_model_fav_imb_mean:.3f}<{self._model_min_fav_imbalance_mean}" if _model_mean_filtered else (f"  mean={_model_fav_imb_mean:.3f}" if _model_fav_imb_mean is not None else "  mean=n/a")
            _model_ob_filter_str = f"  {_snap_str}{_mean_str}"

        paper_run = not self._model_enabled or _model_ob_filtered
        fill_price = trigger_ask + SLIPPAGE if paper_run else trigger_ask
        est_size   = round(model_usdc / fill_price, 2) if fill_price > 0 else 0.0
        fee_usdc   = round(_calc_fee_usdc(est_size, fill_price), 4)

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
            coin_ob_imbalance=round(_ob_snap, 4)    if _ob_snap    is not None else None,
            coin_ob_imb_5s=round(_ob_mean_5s, 4)   if _ob_mean_5s is not None else None,
            coin_ob_trend=round(_ob_trend, 4)       if _ob_trend   is not None else None,
        )

        if paper_run:
            mtrade.order_id = "DRY_RUN_OB_FILTERED" if _model_ob_filtered else "DRY_RUN"
            mtrade.slippage = SLIPPAGE
            # sign_ms / post_ms / order_ms intentionally left None — no real order placed
            if not self._model_enabled and _model_ob_filtered:
                _paper_reason = "MODEL DISABLED + OB FILTERED"
            elif _model_ob_filtered:
                _paper_reason = "OB FILTERED"
            else:
                _paper_reason = "MODEL DISABLED"
            _fav_str = _model_ob_filter_str if _model_ob_filtered else (f"  snap={_model_fav_imb:.3f}" if _model_fav_imb is not None else "") + (f"  mean={_model_fav_imb_mean:.3f}" if _model_fav_imb_mean is not None else "")
            _log_fn = log.debug if (_model_ob_filtered and self._model_bad_count >= BAD_TRADE_CSV_CAP) else log.info
            _log_fn("[%s] ENTRY %s %s  predicted=%.1f%%  ask=%.3f  fill=%.3f  edge=%+.3f  elapsed=%ds%s",
                    _paper_reason, self.asset, side, predicted_win * 100,
                    trigger_ask, fill_price, edge, elapsed, _fav_str)
            if _model_ob_filtered:
                self._model_bad_count += 1
                self._model_last_fire_ts = None
                self._model_trade = mtrade
                if self._model_bad_count <= BAD_TRADE_CSV_CAP:
                    await asyncio.to_thread(_write_model_trade, mtrade, self._model_trades_csv)
                return  # don't count against per-window trade limit
        else:
            try:
                assert self._http is not None
                _retry_delay = 2.0
                order = None
                for _attempt in range(2):
                    try:
                        order, mtrade.sign_ms, mtrade.post_ms = await clob_client.sign_and_post_async(
                            self._http, self._clob, token_id, side, model_usdc,
                            price_cap=trigger_ask + self._model_max_slippage,
                        )
                        break
                    except Exception as _inner_exc:
                        if _attempt == 0 and "425" in str(_inner_exc):
                            log.warning(
                                "[MODEL] 425 service not ready %s %s — retrying in %.0fs",
                                self.asset, side, _retry_delay,
                            )
                            await asyncio.sleep(_retry_delay)
                            continue
                        raise
                assert order is not None
                mtrade.order_ms   = round((time.perf_counter() - _trigger_ts) * 1000, 1)
                mtrade.order_id   = order.order_id
                mtrade.fill_price = order.price if order.price > 0 else trigger_ask
                mtrade.fill_size  = order.size  if order.size  > 0 else est_size
                notional_usdc     = round(mtrade.fill_price * mtrade.fill_size, 4)
                mtrade.fee_usdc   = round(_calc_fee_usdc(mtrade.fill_size, mtrade.fill_price), 4)
                mtrade.fill_usdc  = round(notional_usdc + mtrade.fee_usdc, 4)
                mtrade.slippage   = round(mtrade.fill_price - trigger_ask, 4)
                log.info("[MODEL] FILLED %s %s  predicted=%.1f%%  %.4f shares @ %.4f  ($%.2f)  edge=%+.3f  elapsed=%ds  order=%.0fms",
                         self.asset, side, predicted_win * 100, mtrade.fill_size,
                         mtrade.fill_price, mtrade.fill_usdc, edge, elapsed, mtrade.order_ms)
            except Exception as exc:
                mtrade.order_ms = round((time.perf_counter() - _trigger_ts) * 1000, 1)
                if "425" in str(exc):
                    _backoff = 30.0
                    self._clob_backoff_until = time.time() + _backoff
                    log.warning(
                        "[MODEL] 425 persisted after retry %s %s — CLOB backoff %.0fs",
                        self.asset, side, _backoff,
                    )
                    self._model_last_fire_ts = None
                    self._model_bad_count += 1
                    return  # don't record as a trade or count against limit
                mtrade.status = "order_failed"
                err_str = str(exc).lower()
                is_fok_failure = "fully filled" in err_str or "fok" in err_str
                if not is_fok_failure:
                    log.error("[MODEL] Order failed %s %s: %s", self.asset, side, exc)
                if any(p in err_str for p in _BALANCE_PHRASES):
                    mtrade.status = "insufficient_balance_disabled"
                    self._disable_strategy_on_balance_error("model", side, exc)
                else:
                    if is_fok_failure:
                        FOK_SLIPPAGE = 0.25
                        mtrade.fill_price = round(trigger_ask + FOK_SLIPPAGE, 4)
                        mtrade.fill_size  = round(model_usdc / mtrade.fill_price, 2) if mtrade.fill_price > 0 else 0.0
                        notional          = round(mtrade.fill_price * mtrade.fill_size, 4)
                        mtrade.fee_usdc   = round(_calc_fee_usdc(mtrade.fill_size, mtrade.fill_price), 4)
                        mtrade.fill_usdc  = round(notional + mtrade.fee_usdc, 4)
                        mtrade.slippage   = FOK_SLIPPAGE
                        mtrade.status     = "fok_killed"
                        mtrade.order_id   = "FOK_KILLED"
                        log.warning("[MODEL] FOK KILLED %s %s  hypothetical fill=%.4f (+%.2f slip)", self.asset, side, mtrade.fill_price, FOK_SLIPPAGE)
                        self._model_last_fire_ts = None
                        self._model_bad_count    += 1
                        self._model_trade = mtrade
                        if self._model_bad_count <= BAD_TRADE_CSV_CAP:
                            await asyncio.to_thread(_write_model_trade, mtrade, self._model_trades_csv)
                        return  # don't count against per-window trade limit
                    # else: bad token / orderbook gone — don't retry
                    self._model_last_fire_ts = None
                    self._model_bad_count    += 1
                    self._model_trade = mtrade
                    if self._model_bad_count <= BAD_TRADE_CSV_CAP:
                        await asyncio.to_thread(_write_model_trade, mtrade, self._model_trades_csv)
                    return  # don't count against per-window trade limit

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

        # Session loss circuit breaker — only real fills count.
        if not is_fok and trade.order_id and not trade.order_id.startswith("DRY_RUN") and trade.order_id != "FOK_KILLED":
            self._session_model_pnl      += pnl
            self._session_model_peak_pnl  = max(self._session_model_peak_pnl, self._session_model_pnl)
            log.info(
                "[MODEL] session PnL  %s  cumulative=$%.4f  peak=$%.4f",
                self.asset, self._session_model_pnl, self._session_model_peak_pnl,
            )
            if (self._session_model_pnl <= self._session_model_peak_pnl - SESSION_LOSS_LIMIT
                    and self._model_enabled):
                self._model_enabled       = False
                self._model_in_cooldown   = True
                self._model_cooldown_count = 0
                self._model_cooldown_pnl   = 0.0
                log.warning(
                    "[MODEL] LOSS LIMIT HIT  %s  session_pnl=$%.4f  peak=$%.4f  drawdown=$%.4f"
                    " — entering %d-window cooldown",
                    self.asset, self._session_model_pnl, self._session_model_peak_pnl,
                    self._session_model_pnl - self._session_model_peak_pnl,
                    LOSS_LIMIT_COOLDOWN_WINDOWS,
                )
        elif self._model_in_cooldown and trade.order_id and trade.order_id.startswith("DRY_RUN"):
            self._model_cooldown_pnl += pnl
            log.info(
                "[MODEL] cooldown paper PnL  %s  +$%.4f  cycle_total=$%.4f",
                self.asset, pnl, self._model_cooldown_pnl,
            )

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

        # Fast lookup: history is in chrono order, search from the end (most recent prices)
        def _px_at(offset: float) -> float | None:
            """Last price at or before now-offset seconds (matches 1s candle close lookback)."""
            target = now - offset
            # Iterate backwards from end — prices are monotonically increasing
            for ts, px in reversed(self._price_history):
                if ts <= target:
                    return px
            return None

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

        # dist_low_30 / dist_high_30: distance from 30s rolling low/high (σ-norm)
        dist_low_30:  float | None = None
        dist_high_30: float | None = None
        cutoff_30 = now - 30.0
        prices_30 = [px for ts, px in self._price_history if ts >= cutoff_30]
        if len(prices_30) >= 10:
            _lo = min(prices_30)
            _hi = max(prices_30)
            dist_low_30  = (p0 - _lo) / sigma   # ≥0: distance above rolling low
            dist_high_30 = (p0 - _hi) / sigma   # ≤0: distance below rolling high

        # Interaction features
        move_x_elapsed = move_sigmas * elapsed_second
        move_x_vol     = (move_sigmas * vol_10s_log) if vol_10s_log is not None else None
        move_x_elapsed_x_vel10s: float | None = (
            move_x_elapsed * vel_10s if vel_10s is not None else None
        )
        move_sigmas_x_acc10s: float | None = (
            move_sigmas * acc_10s if acc_10s is not None else None
        )

        # PM orderbook imbalance features for the market component of the ensemble
        # Use cached books from status loop if available
        up_imbalance: float | None = None
        dn_imbalance: float | None = None
        if self._market is not None:
            up_book = self._cached_up_book
            dn_book = self._cached_dn_book
            if up_book is not None:
                _up_total = up_book.bid_volume + up_book.ask_volume
                if _up_total > 0:
                    up_imbalance = up_book.bid_volume / _up_total
            if dn_book is not None:
                _dn_total = dn_book.bid_volume + dn_book.ask_volume
                if _dn_total > 0:
                    dn_imbalance = dn_book.bid_volume / _dn_total

        feats: dict[str, float | None] = {
            "move_sigmas":    _r(move_sigmas, 4),
            "elapsed_second": elapsed_second,
            "hour_sin":       self._cached_hour_sin,
            "hour_cos":       self._cached_hour_cos,
            "vel_5s":         _r(vel_5s),
            "dist_low_30":    _r(dist_low_30),
            "dist_high_30":   _r(dist_high_30),
            "move_x_elapsed": _r(move_x_elapsed),
            "move_x_vol":     _r(move_x_vol),
            "acc_4s":         _r(acc_4s),
            "move_x_elapsed_x_vel10s": _r(move_x_elapsed_x_vel10s),
            "move_x_acc10s":           _r(move_sigmas_x_acc10s),
            # Extended features
            "vel_2s":         _r(vel_2s),
            "vel_10s":        _r(vel_10s),
            "acc_10s":        _r(acc_10s),
            "vel_ratio":      _r(vel_ratio),
            "vel_decay":      _r(vel_decay),
            "vol_10s_log":    _r(vol_10s_log),
            # Market features (PM orderbook) — used by the market component of the ensemble
            "up_imbalance":   _r(up_imbalance, 4),
            "dn_imbalance":   _r(dn_imbalance, 4),
        }

        return feats

    def _run_inference(self, features: dict) -> float | None:
        """Synchronous sklearn inference — safe to call directly in the event loop (~0.3ms)."""
        if self._model is None or not features:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
                
                t0 = time.perf_counter()
                model_type = self._model.get("type", "logreg")
                if model_type == "ensemble_lgb_mkt":
                    # Run LGB on coin features
                    lgb_feats = self._model["lgb_features"]
                    X_lgb = pd.DataFrame([[features.get(f) for f in lgb_feats]], columns=lgb_feats)
                    p_lgb = float(self._model["lgb_pipe"].predict_proba(X_lgb)[0, 1])
                    # Run market model on coin + PM orderbook features
                    mkt_feats = self._model["mkt_features"]
                    X_mkt = pd.DataFrame([[features.get(f) for f in mkt_feats]], columns=mkt_feats)
                    p_mkt = float(self._model["mkt_pipe"].predict_proba(X_mkt)[0, 1])
                    result = 0.5 * p_lgb + 0.5 * p_mkt
                    log.debug(
                        "inference ensemble  lgb=%.4f  mkt=%.4f  blend=%.4f  %.2fms",
                        p_lgb, p_mkt, result, (time.perf_counter() - t0) * 1000,
                    )
                    return result
                else:
                    # Legacy single-pipeline model (logreg or lgb-only)
                    feat_order = self._model["features"]
                    X = np.array([[features.get(f) for f in feat_order]], dtype=float)
                    result = float(self._model["pipe"].predict_proba(X)[0, 1])
                    log.debug(
                        "inference %s  prob=%.4f  %.2fms",
                        model_type, result, (time.perf_counter() - t0) * 1000,
                    )
                    return result
        except Exception as exc:
            log.warning("Inference failed: %s", exc)
            return None

    async def _status_loop(self) -> None:
        _last_inference_ts: float = 0.0
        _last_write_ts:     float = 0.0
        _last_keepalive_ts: float = 0.0
        _cached_predicted_win: float | None = None
        while True:
            # Keep the HTTP/2 connection to the CLOB warm so order POSTs don't
            # pay a TLS re-handshake. Fire a lightweight GET every 60 seconds.
            now = time.time()
            if self._http is not None and now - _last_keepalive_ts >= 60.0:
                try:
                    await self._http.get(f"{self._clob.host}/time", timeout=3.0)
                    _last_keepalive_ts = now
                except Exception:
                    pass  # stale connection — next order will reconnect

            market = self._market
            up_price = dn_price = up_ask = dn_ask = None
            if market:
                up_price = self._ws.get_price(market.up_token.token_id)
                dn_price = self._ws.get_price(market.down_token.token_id)
                up_ask   = self._ws.get_ask(market.up_token.token_id)
                dn_ask   = self._ws.get_ask(market.down_token.token_id)

            # Cache time.time() once per iteration
            t_now = time.time()
            
            # Update cached books (will be used by feature computation and order execution)
            if market is not None:
                self._cached_up_book = self._ws.get_book(market.up_token.token_id)
                self._cached_dn_book = self._ws.get_book(market.down_token.token_id)
            
            features = self._compute_features()

            # Run inference at most every 0.5s — spawning a thread + building a
            # DataFrame on every tick is the main CPU cost on slower machines.
            predicted_win: float | None = _cached_predicted_win
            if t_now - _last_inference_ts >= 0.5:
                _cached_predicted_win = self._run_inference(features)
                predicted_win      = _cached_predicted_win
                _last_inference_ts = t_now

            # Model trade: fire when edge >= threshold
            # once-per-window (cooldown=0) or every N seconds (cooldown=N)
            if (
                predicted_win is not None
                and market is not None
                and self._coin_open is not None
                and self._coin_current is not None
            ):
                try:
                    # Check gate conditions
                    _cd = self._model_multi_trade_cooldown
                    if _cd == 0:
                        _model_gate = self._model_last_fire_ts is None
                    else:
                        _model_gate = (self._model_last_fire_ts is None or
                                      t_now - self._model_last_fire_ts >= _cd)
                    
                    _max_trades = self._model_max_trades_per_window
                    _trades_limit_ok = (_max_trades == 0 or len(self._model_trades) < _max_trades)

                    if _model_gate and _trades_limit_ok:
                        # Get asks once
                        up_ask_now = self._ws.get_ask(market.up_token.token_id)
                        dn_ask_now = self._ws.get_ask(market.down_token.token_id)
                        
                        # Compute edges
                        up_edge = (predicted_win - up_ask_now) if up_ask_now is not None else None
                        dn_edge = ((1.0 - predicted_win) - dn_ask_now) if dn_ask_now is not None else None
                        
                        # Check window timing
                        elapsed_now = int(t_now - self._window_start)
                        in_model_window = (self._model_window_start <= elapsed_now <= self._model_window_end)
                        
                        if in_model_window:
                            # Pick the best side (highest edge)
                            best_side = None
                            best_ask = None
                            best_edge = None
                            best_token = None
                            best_edge_val = -1.0
                            
                            if up_edge is not None and up_edge >= self._MODEL_EDGE_THRESHOLD and up_edge > best_edge_val:
                                best_side, best_ask, best_edge, best_token = ("UP", up_ask_now, up_edge, market.up_token.token_id)
                                best_edge_val = up_edge
                            if dn_edge is not None and dn_edge >= self._MODEL_EDGE_THRESHOLD and dn_edge > best_edge_val:
                                best_side, best_ask, best_edge, best_token = ("DOWN", dn_ask_now, dn_edge, market.down_token.token_id)
                            
                            if best_side is not None:
                                self._model_last_fire_ts = t_now
                                asyncio.create_task(self._execute_model_buy(
                                    side=best_side,
                                    token_id=best_token,
                                    trigger_ask=best_ask,
                                    edge=best_edge,
                                    predicted_win=predicted_win,
                                    elapsed=elapsed_now,
                                ))
                except Exception as exc:
                    log.warning("Model fire logic failed: %s", exc)

            try:
                _coin_ob_imb = self._coin_stream.get_coin_ob_imbalance()
                payload = {
                    "updated_at":     t_now,
                    "window_start":   self._window_start,
                    "window_end":     self._window_end,
                    "elapsed_secs":   round(t_now - self._window_start),
                    "remaining_secs": round(self._window_end - t_now),
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
                    "coin_bid_vol":   self._coin_stream._coin_bid_vol or None,
                    "coin_ask_vol":   self._coin_stream._coin_ask_vol or None,
                    "coin_ob_imbalance": _coin_ob_imb,
                    "filled":                  self._filled,
                    "momentum_trade_count":    self._momentum_trade_count,
                    "trade":                   asdict(self._trade) if self._trade else None,
                    "momentum_trades":         [asdict(t) for t in self._momentum_trades],

                    "features":       {k: (float(v) if v is not None else None) for k, v in features.items()},
                    "predicted_win":        predicted_win,
                    "MODEL_EDGE_THRESHOLD": self._MODEL_EDGE_THRESHOLD,
                    "model_trade":          asdict(self._model_trade) if self._model_trade else None,
                    "model_trades":         [asdict(t) for t in self._model_trades],
                    "momentum_enabled":     self._momentum_enabled,
                    "model_enabled":        self._model_enabled,
                    "price_history":        [px for _, px in self._price_history],
                }
                if t_now - _last_write_ts >= 2.0:
                    await asyncio.to_thread(_write_status, payload, self._status_json)
                    _last_write_ts = t_now
            except Exception as exc:
                log.warning("Status write failed: %s", exc, exc_info=True)
            await asyncio.sleep(0.25)


