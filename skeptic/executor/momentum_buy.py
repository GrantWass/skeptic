"""
Live executor for the coin-momentum buy strategy.

Watches the real coin price (via Binance REST) within each 5-minute window.
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
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import httpx

from skeptic import config
from skeptic.clients import clob as clob_client
from skeptic.clients import ctf as ctf_client
from skeptic.clients import gamma
from skeptic.clients.ws import MarketChannel
from skeptic.models.market import Market
from skeptic.utils.time import current_window_start, next_window_start, sleep_until

log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

LIVE_DIR = os.path.join("data", "live")

BINANCE_PRICE_URL   = "https://api.binance.com/api/v3/ticker/price"
BINANCE_KLINES_URL  = "https://api.binance.com/api/v3/klines"

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
    "fill_price", "fill_size", "fill_usdc",
    "sigma_value", "sigma_entry", "max_pm_price",
    "coin_open", "coin_trigger", "coin_move",
    "slippage", "window_start_ts", "window_end_ts",
    "resolution", "pnl_usdc", "status", "order_id",
]

SLIPPAGE = 0.05


@dataclass
class MomentumTrade:
    ts: float
    asset: str
    side: str
    token_id: str
    fill_price: float
    fill_size: float
    fill_usdc: float
    sigma_value: float
    sigma_entry: float
    max_pm_price: float
    coin_open: float
    coin_trigger: float
    coin_move: float
    slippage: float
    window_start_ts: int
    window_end_ts: int
    resolution: float | None = None
    pnl_usdc: float | None = None
    status: str = "open"
    order_id: str = ""


def _ensure_live_dir(trades_csv: str) -> None:
    os.makedirs(LIVE_DIR, exist_ok=True)
    if not os.path.exists(trades_csv):
        with open(trades_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=TRADE_FIELDS).writeheader()


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
        dry_run: bool = False,
        name: str = "momentum",
    ) -> None:
        assert direction in ("up", "down", "both"), "direction must be 'up', 'down', or 'both'"
        self.asset        = asset
        self.sigma_value  = sigma_value
        self.sigma_entry  = sigma_entry
        self.max_pm_price = max_pm_price
        self.direction    = direction
        self.wallet_pct   = wallet_pct
        self.dry_run      = dry_run

        self._symbol = ASSET_TO_SYMBOL.get(asset.upper())
        if self._symbol is None:
            raise ValueError(f"No Binance symbol for asset '{asset}'")

        self._trades_csv  = os.path.join(LIVE_DIR, f"trades_{name}.csv")
        self._status_json = os.path.join(LIVE_DIR, f"status_{name}.json")

        self._clob = clob_client.build_client()
        self._ws   = MarketChannel()
        self._http: httpx.AsyncClient | None = None

        # Per-window state
        self._market:   Market | None = None
        self._filled:   bool = False
        self._trade:    MomentumTrade | None = None
        self._presigned: dict[str, object] = {}
        self._window_start: int = 0
        self._window_end:   int = 0
        self._coin_open:    float | None = None
        self._coin_current: float | None = None
        self._position_usdc: float = 0.0

    # ── Public ────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        _ensure_live_dir(self._trades_csv)
        threshold_usdc = self.sigma_entry * self.sigma_value
        log.info(
            "MomentumBuyExecutor  asset=%s  sigma_value=%g  sigma_entry=%.1f  "
            "threshold_move=%g  max_pm_price=%.2f  direction=%s  dry_run=%s",
            self.asset, self.sigma_value, self.sigma_entry,
            threshold_usdc, self.max_pm_price, self.direction, self.dry_run,
        )

        async with httpx.AsyncClient() as http:
            self._http = http
            ws_task     = asyncio.create_task(self._ws.run())
            status_task = asyncio.create_task(self._status_loop())
            ticker_task = asyncio.create_task(self._ticker_loop())

            try:
                while True:
                    ws  = current_window_start()
                    we  = ws + config.WINDOW_SECS
                    now = time.time()

                    if now > we - 5:
                        nw = next_window_start()
                        log.info("Window ending soon — waiting for %s", _fmt_ts(nw))
                        await sleep_until(nw)
                        ws, we = nw, nw + config.WINDOW_SECS

                    log.info("=== Window %s – %s ===", _fmt_ts(ws), _fmt_ts(we))
                    await self._run_window(ws, we)
                    await sleep_until(we)
            finally:
                ws_task.cancel()
                status_task.cancel()
                ticker_task.cancel()

    # ── Window ────────────────────────────────────────────────────────────────

    async def _run_window(self, window_start: int, window_end: int) -> None:
        self._filled        = False
        self._trade         = None
        self._market        = None
        self._presigned     = {}
        self._coin_open     = None
        self._coin_current  = None
        self._window_start  = window_start
        self._window_end    = window_end

        balance = clob_client.get_usdc_balance(self._clob)
        self._position_usdc = round(balance * self.wallet_pct, 4)
        log.info("Balance $%.2f  position $%.2f (%.1f%%)",
                 balance, self._position_usdc, self.wallet_pct * 100)

        # Fetch the coin price at the exact window start timestamp
        self._coin_open = await self._fetch_coin_price_at(window_start)
        if self._coin_open is None:
            log.error("Could not fetch coin open price for %s — skipping window", self._symbol)
            return
        secs_in = int(time.time() - window_start)
        log.info("%s window open price: %g  (fetched from t=0, currently %ds into window)",
                 self.asset, self._coin_open, secs_in)

        # Discover Polymarket market
        market = await gamma.get_current_window_market(
            self.asset, window_start, self._http, retries=24, retry_delay=5.0)
        if market is None:
            log.warning("No Polymarket market for %s this window — skipping", self.asset)
            return
        self._market = market
        log.info("  UP=%s…  DOWN=%s…",
                 market.up_token.token_id[:10], market.down_token.token_id[:10])

        # Subscribe to WebSocket and pre-sign orders
        all_tokens = [market.up_token.token_id, market.down_token.token_id]
        await self._ws.subscribe(*all_tokens)
        await self._ws.reconnect()
        await asyncio.sleep(1.5)

        if not self.dry_run:
            for token_id in all_tokens:
                try:
                    signed = await asyncio.to_thread(
                        clob_client.presign_market_order,
                        self._clob, token_id, self._position_usdc,
                    )
                    self._presigned[token_id] = signed
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

        if self._trade:
            await asyncio.sleep(8.0)
            await self._resolve()

        await self._ws.unsubscribe(*all_tokens)

    async def _momentum_watch_loop(self) -> None:
        """Poll coin price every 500ms; trigger when momentum threshold is crossed."""
        threshold = self.sigma_entry * self.sigma_value
        _skipped: set[str] = set()  # track directions already logged as skipped

        while True:
            coin_price = await self._fetch_coin_price()
            if coin_price is None:
                await asyncio.sleep(0.5)
                continue
            self._coin_current = coin_price

            if self._filled:
                await asyncio.sleep(0.5)
                continue

            move = coin_price - self._coin_open

            if self.direction in ("up", "both") and move >= threshold:
                pm_price = self._ws.get_ask(self._market.up_token.token_id)
                if pm_price is not None and pm_price <= self.max_pm_price:
                    log.info("TRIGGER UP  coin_open=%g  coin_now=%g  move=%+g  pm_ask=%.4f  threshold=%g",
                             self._coin_open, coin_price, move, pm_price, threshold)
                    self._filled = True
                    _skipped.discard("up")
                    await self._execute_buy("UP", self._market.up_token.token_id,
                                            pm_price, coin_price, move)
                elif pm_price is not None and "up" not in _skipped:
                    log.info("SKIP UP  coin_move=%+g  pm_ask=%.4f > max=%.4f",
                             move, pm_price, self.max_pm_price)
                    _skipped.add("up")
            else:
                _skipped.discard("up")  # reset if coin drops back below threshold

            if self.direction in ("down", "both") and move <= -threshold:
                pm_price = self._ws.get_ask(self._market.down_token.token_id)
                if pm_price is not None and pm_price <= self.max_pm_price:
                    log.info("TRIGGER DOWN  coin_open=%g  coin_now=%g  move=%+g  pm_ask=%.4f  threshold=%g",
                             self._coin_open, coin_price, move, pm_price, threshold)
                    self._filled = True
                    _skipped.discard("down")
                    await self._execute_buy("DOWN", self._market.down_token.token_id,
                                            pm_price, coin_price, move)
                elif pm_price is not None and "down" not in _skipped:
                    log.info("SKIP DOWN  coin_move=%+g  pm_ask=%.4f > max=%.4f",
                             move, pm_price, self.max_pm_price)
                    _skipped.add("down")
            else:
                _skipped.discard("down")

            await asyncio.sleep(0.5)

    # ── Execution ─────────────────────────────────────────────────────────────

    async def _execute_buy(
        self,
        side: str,
        token_id: str,
        trigger_pm_price: float,
        coin_trigger: float,
        coin_move: float,
    ) -> None:
        est_size = round(self._position_usdc / trigger_pm_price, 2)

        trade = MomentumTrade(
            ts=time.time(),
            asset=self.asset,
            side=side,
            token_id=token_id,
            fill_price=trigger_pm_price,
            fill_size=est_size,
            fill_usdc=self._position_usdc,
            sigma_value=self.sigma_value,
            sigma_entry=self.sigma_entry,
            max_pm_price=self.max_pm_price,
            coin_open=self._coin_open,
            coin_trigger=coin_trigger,
            coin_move=coin_move,
            slippage=SLIPPAGE,
            window_start_ts=self._window_start,
            window_end_ts=self._window_end,
        )

        if self.dry_run:
            log.info(
                "[DRY RUN] BUY %s %s  $%.2f USDC  pm=%.4f  coin_open=%g  coin_now=%g  move=%+g",
                self.asset, side, self._position_usdc, trigger_pm_price,
                self._coin_open, coin_trigger, coin_move,
            )
            trade.order_id = "DRY_RUN"
        else:
            try:
                signed = self._presigned.get(token_id)
                if signed is not None:
                    order = await asyncio.to_thread(
                        clob_client.post_presigned_order,
                        self._clob, signed, token_id, side, self._position_usdc,
                    )
                else:
                    order = await asyncio.to_thread(
                        clob_client.place_market_order,
                        self._clob, token_id, side, self._position_usdc,
                    )
                trade.order_id   = order.order_id
                trade.fill_price = order.price if order.price > 0 else trigger_pm_price
                trade.fill_size  = order.size  if order.size  > 0 else est_size
                trade.fill_usdc  = round(trade.fill_price * trade.fill_size, 4)
                trade.slippage   = round(trade.fill_price - trigger_pm_price, 4)
                log.info(
                    "FILLED  %s %s  %.4f shares @ %.4f  ($%.2f)  slippage=%+.4f  order=%s",
                    self.asset, side, trade.fill_size, trade.fill_price,
                    trade.fill_usdc, trade.slippage, order.order_id[:16],
                )
            except Exception as exc:
                log.error("Order failed %s %s: %s", self.asset, side, exc)
                trade.status = "order_failed"

        self._trade = trade
        _write_trade(trade, self._trades_csv)

    # ── Resolution ────────────────────────────────────────────────────────────

    async def _resolve(self) -> None:
        trade = self._trade
        if not trade or trade.status != "open":
            return
        for _ in range(18):
            price = self._ws.get_price(trade.token_id)
            if price is not None:
                if price >= 0.95:
                    resolution = 1.0
                    break
                if price <= 0.05:
                    resolution = 0.0
                    break
            await asyncio.sleep(5.0)
        else:
            log.warning("Could not resolve %s %s — unresolved", self.asset, trade.side)
            trade.status = "unresolved"
            _write_trade(trade, self._trades_csv)
            return

        win = resolution >= 0.9
        pnl = ((1.0 - trade.fill_price) if win else -trade.fill_price) * trade.fill_size
        trade.resolution = resolution
        trade.pnl_usdc   = round(pnl, 4)
        trade.status     = "won" if win else "lost"
        log.info("RESOLVED  %s %s  %s  PnL=$%.4f",
                 self.asset, trade.side, trade.status.upper(), pnl)
        _write_trade(trade, self._trades_csv)

    # ── Binance price fetch ───────────────────────────────────────────────────

    async def _fetch_coin_price(self) -> float | None:
        """Fetch current coin price from Binance."""
        try:
            resp = await self._http.get(
                BINANCE_PRICE_URL,
                params={"symbol": self._symbol},
                timeout=5.0,
            )
            resp.raise_for_status()
            return float(resp.json()["price"])
        except Exception as exc:
            log.warning("Binance price fetch failed: %s", exc)
            return None

    async def _fetch_coin_price_at(self, ts: int) -> float | None:
        """
        Fetch the coin open price at a specific Unix timestamp using Binance 1s klines.
        Falls back to current price if the kline is unavailable.
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

        log.warning("Falling back to current price for window open")
        return await self._fetch_coin_price()

    # ── Ticker / status ───────────────────────────────────────────────────────

    async def _ticker_loop(self) -> None:
        # Determine coin decimal places from the window open price once it's available
        # so every tick prints the same width (e.g. BTC=2dp, DOGE=5dp)
        _coin_dp: int | None = None

        while True:
            if not self._market or self._coin_open is None:
                await asyncio.sleep(0.5)
                continue

            # Compute decimal places once from window open magnitude
            if _coin_dp is None:
                mag = abs(self._coin_open)
                if mag >= 1000:   _coin_dp = 2
                elif mag >= 1:    _coin_dp = 4
                elif mag >= 0.01: _coin_dp = 6
                else:             _coin_dp = 8

            now       = time.time()
            elapsed   = int(now - self._window_start)
            remaining = max(0, self._window_end - int(now))
            coin      = self._coin_current or self._coin_open
            move      = coin - self._coin_open
            sigmas    = move / self.sigma_value if self.sigma_value else 0.0

            coin_str  = f"{coin:.{_coin_dp}f}"

            up_mid  = self._ws.get_price(self._market.up_token.token_id)
            dn_mid  = self._ws.get_price(self._market.down_token.token_id)
            up_ask  = self._ws.get_ask(self._market.up_token.token_id)
            dn_ask  = self._ws.get_ask(self._market.down_token.token_id)

            def _side_str(mid, ask):
                if mid is None or ask is None:
                    return "——/——/——"
                return f"{mid:.3f}/{ask:.3f}(+{ask-mid:.3f})"

            up_str = _side_str(up_mid, up_ask)
            dn_str = _side_str(dn_mid, dn_ask)

            trade = self._trade
            if trade and trade.status in ("won", "lost"):
                status_tag = f"✓{trade.status.upper()} ${trade.pnl_usdc:+.2f}"
            elif trade and trade.status == "open":
                status_tag = f"FILLED {trade.side}@{trade.fill_price:.3f}"
            elif self._filled:
                status_tag = "filled"
            else:
                status_tag = "watching"

            print(
                f"[{elapsed:>3}s/{remaining:>3}s]  {coin_str} ({sigmas:+.2f}σ)"
                f"  UP {up_str}  DN {dn_str}  [{status_tag}]",
                flush=True,
            )
            await asyncio.sleep(5.0)

    async def _status_loop(self) -> None:
        while True:
            try:
                market = self._market
                up_price = dn_price = None
                if market:
                    up_price = self._ws.get_price(market.up_token.token_id)
                    dn_price = self._ws.get_price(market.down_token.token_id)

                _write_status({
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
                    "coin_move":      round((self._coin_current or self._coin_open or 0) - (self._coin_open or 0), 4),
                    "threshold_move": round(self.sigma_entry * self.sigma_value, 4),
                    "up_price":       up_price,
                    "down_price":     dn_price,
                    "filled":         self._filled,
                    "trade":          asdict(self._trade) if self._trade else None,
                    "dry_run":        self.dry_run,
                }, self._status_json)
            except Exception as exc:
                log.debug("Status write failed: %s", exc)
            await asyncio.sleep(0.5)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S UTC")
