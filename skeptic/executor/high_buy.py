"""
Live executor for the high-probability buy strategy.

For each 5-minute window:
1. Discover the active UP/DOWN market for each asset via Gamma API
2. Subscribe to live token price streams via WebSocket
3. When UP or DOWN price >= threshold (first trigger wins), place a buy immediately
4. After window closes, infer resolution from final price and log PnL

Writes two files for dashboard consumption:
  data/live/status.json  — current prices + open positions (updated ~every 500ms)
  data/live/trades.csv   — append-only trade log
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

LIVE_DIR = os.path.join("data", "live")

TRADE_FIELDS = [
    "ts", "asset", "side", "token_id",
    "fill_price", "fill_size", "fill_usdc",
    "threshold", "slippage",
    "window_start_ts", "window_end_ts",
    "resolution", "pnl_usdc", "status", "order_id",
]

SLIPPAGE = 0.05  # matches research model


@dataclass
class LiveTrade:
    ts: float
    asset: str
    side: str           # "UP" or "DOWN"
    token_id: str
    fill_price: float
    fill_size: float    # shares purchased
    fill_usdc: float    # USDC spent
    threshold: float
    slippage: float
    window_start_ts: int
    window_end_ts: int
    resolution: float | None = None  # 1.0 = win, 0.0 = loss
    pnl_usdc: float | None = None
    status: str = "open"             # open | won | lost | unresolved
    order_id: str = ""


def _ensure_live_dir(trades_csv: str) -> None:
    os.makedirs(LIVE_DIR, exist_ok=True)
    if not os.path.exists(trades_csv):
        with open(trades_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=TRADE_FIELDS).writeheader()


def _write_trade(trade: LiveTrade, trades_csv: str) -> None:
    """Append a trade row to the instance's trades CSV."""
    _ensure_live_dir(trades_csv)
    with open(trades_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
        writer.writerow(asdict(trade))


def _write_status(status: dict, status_json: str) -> None:
    os.makedirs(LIVE_DIR, exist_ok=True)
    tmp = status_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(status, f)
    os.replace(tmp, status_json)


class HighBuyExecutor:
    """
    Monitors live Polymarket 5-min markets and executes the high-probability
    buy strategy: buy UP or DOWN the first time price >= threshold.
    """

    def __init__(
        self,
        assets: list[str],
        threshold: float,
        wallet_pct: float,
        dry_run: bool = False,
        name: str = "default",
        cutoff_secs: int = 0,
    ) -> None:
        self.assets = assets
        self.threshold = threshold
        self.wallet_pct = wallet_pct
        self.dry_run = dry_run
        self.cutoff_secs = cutoff_secs  # seconds into window before triggering (0 = any time)
        self._trades_csv  = os.path.join(LIVE_DIR, f"trades_{name}.csv")
        self._status_json = os.path.join(LIVE_DIR, f"status_{name}.json")
        self._position_usdc: float = 0.0  # set at window start from live balance
        self._window_start: int = 0
        self._window_end:   int = 0

        self._clob  = clob_client.build_client()
        self._ws    = MarketChannel()
        self._http: httpx.AsyncClient | None = None

        # Per-window state (reset each window)
        self._markets:   dict[str, Market]    = {}  # asset → Market
        self._filled:    dict[str, bool]      = {}  # asset → already triggered this window
        self._trades:    dict[str, LiveTrade] = {}  # asset → open trade
        self._presigned: dict[str, object]    = {}  # token_id → pre-signed order

    # ── Public API ───────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main loop — runs forever, one iteration per 5-minute window."""
        _ensure_live_dir(self._trades_csv)
        log.info("HighBuyExecutor starting  threshold=%.2f  wallet_pct=%.1f%%  assets=%s  dry_run=%s",
                 self.threshold, self.wallet_pct * 100, self.assets, self.dry_run)

        async with httpx.AsyncClient() as http:
            self._http = http
            ws_task     = asyncio.create_task(self._ws.run())
            status_task = asyncio.create_task(self._status_loop())
            ticker_task = asyncio.create_task(self._ticker_loop())

            try:
                while True:
                    ws = current_window_start()
                    we = ws + config.WINDOW_SECS
                    now = time.time()

                    # Align to window boundary
                    if now > we - 5:
                        # Current window nearly over — wait for next
                        nw = next_window_start()
                        log.info("Window ending soon, waiting for next window at %s",
                                 _fmt_ts(nw))
                        await sleep_until(nw)
                        ws = nw
                        we = ws + config.WINDOW_SECS

                    log.info("=== Window %s – %s ===", _fmt_ts(ws), _fmt_ts(we))
                    await self._run_window(ws, we)

                    # Sleep to next window boundary
                    await sleep_until(we)
            finally:
                ws_task.cancel()
                status_task.cancel()
                ticker_task.cancel()

    # ── Window execution ─────────────────────────────────────────────────────

    async def _run_window(self, window_start: int, window_end: int) -> None:
        self._filled    = {a: False for a in self.assets}
        self._trades    = {}
        self._markets   = {}
        self._presigned = {}
        self._window_start = window_start
        self._window_end   = window_end

        # 1. Fetch live wallet balance and compute position size
        balance = clob_client.get_usdc_balance(self._clob)
        self._position_usdc = round(balance * self.wallet_pct, 4)
        log.info("Wallet balance: $%.2f  position size: $%.2f (%.1f%%)",
                 balance, self._position_usdc, self.wallet_pct * 100)

        # 3. Discover markets in parallel
        results = await asyncio.gather(
            *[gamma.get_current_window_market(a, window_start, self._http,
                                               retries=24, retry_delay=5.0)
              for a in self.assets],
            return_exceptions=True,
        )
        for asset, result in zip(self.assets, results):
            if isinstance(result, Exception) or result is None:
                log.warning("No market for %s this window — skipping", asset)
                continue
            self._markets[asset] = result
            log.info("  %s  up=%s…  down=%s…",
                     asset,
                     result.up_token.token_id[:10],
                     result.down_token.token_id[:10])

        if not self._markets:
            log.warning("No markets found — skipping window")
            return

        # 2. Subscribe and force reconnect to get fresh book snapshots
        all_tokens = []
        for m in self._markets.values():
            all_tokens += [m.up_token.token_id, m.down_token.token_id]
        await self._ws.subscribe(*all_tokens)
        await self._ws.reconnect()
        await asyncio.sleep(1.5)  # Wait for initial book snapshots

        # Pre-sign FOK orders for every token so the hot path only needs to POST.
        # price_cap=0.99 lets the exchange fill at best available price up to 0.99.
        # This also warms the tick-size / neg-risk / fee-rate caches for new token IDs.
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

        # 3. Watch prices until window ends (leave 8s for resolution)
        watch_until = window_end - 8
        now = time.time()
        if now < watch_until:
            try:
                await asyncio.wait_for(
                    self._price_watch_loop(window_start, window_end),
                    timeout=watch_until - now,
                )
            except asyncio.TimeoutError:
                pass  # Normal — window watching period elapsed

        # 4. Resolve open trades
        if self._trades:
            await asyncio.sleep(8.0)  # Let resolution propagate
            await self._resolve_all(window_end)

        # 5. Redeem any won trades on-chain (disabled until wallet is funded with MATIC)
        # if not self.dry_run and self._trades:
        #     await self._redeem_won_trades()

        # 6. Unsubscribe
        await self._ws.unsubscribe(*all_tokens)

    async def _price_watch_loop(self, window_start: int, window_end: int) -> None:
        """Poll price cache every 10ms and trigger buys when threshold is hit."""
        trigger_after = window_start + self.cutoff_secs
        # Snapshot which sides were already above threshold the moment the cutoff is reached.
        # These get execution-quality slippage (fill vs trigger_price), not vs threshold.
        above_at_cutoff: dict[str, set[str]] = {}
        cutoff_recorded = self.cutoff_secs == 0  # no cutoff → never need to record

        while True:
            now = time.time()
            in_window = now >= trigger_after

            if in_window and not cutoff_recorded:
                for asset, market in self._markets.items():
                    above_at_cutoff[asset] = set()
                    for side, token in [("UP", market.up_token), ("DOWN", market.down_token)]:
                        p = self._ws.get_ask(token.token_id)
                        if p is not None and p >= self.threshold:
                            above_at_cutoff[asset].add(side)
                cutoff_recorded = True

            for asset, market in self._markets.items():
                if self._filled[asset] or not in_window:
                    continue

                for side, token in [("UP", market.up_token), ("DOWN", market.down_token)]:
                    price = self._ws.get_ask(token.token_id)
                    if price is None or price < self.threshold:
                        continue
                    already_above = side in above_at_cutoff.get(asset, set())
                    log.info("TRIGGER  %s %s  price=%.4f  threshold=%.2f  elapsed=%.0fs",
                             asset, side, price, self.threshold, now - window_start)
                    self._filled[asset] = True
                    await self._execute_buy(
                        asset, side, token.token_id, price,
                        window_start, window_end, already_above,
                    )
                    break  # One fill per asset per window

            await asyncio.sleep(0.01)

    # ── Order execution ──────────────────────────────────────────────────────

    async def _execute_buy(
        self,
        asset: str,
        side: str,
        token_id: str,
        trigger_price: float,
        window_start: int,
        window_end: int,
        already_above: bool = False,
    ) -> None:
        """Submit a FOK market buy for position_usdc at the live book price."""
        est_size = round(self._position_usdc / trigger_price, 2)  # estimate for logging

        trade = LiveTrade(
            ts=time.time(),
            asset=asset,
            side=side,
            token_id=token_id,
            fill_price=trigger_price,    # updated with actual fill after order response
            fill_size=est_size,
            fill_usdc=self._position_usdc,
            threshold=self.threshold,
            slippage=SLIPPAGE,
            window_start_ts=window_start,
            window_end_ts=window_end,
            status="open",
        )

        if self.dry_run:
            log.info("[DRY RUN] MARKET BUY %s %s  $%.2f USDC  (trigger price=%.4f)",
                     asset, side, self._position_usdc, trigger_price)
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
                    # Fallback: full sign+post (slower, used if pre-sign failed)
                    order = await asyncio.to_thread(
                        clob_client.place_market_order,
                        self._clob, token_id, side, self._position_usdc,
                    )
                trade.order_id   = order.order_id
                trade.fill_price = order.price if order.price > 0 else trigger_price
                trade.fill_size  = order.size  if order.size  > 0 else est_size
                trade.fill_usdc  = round(trade.fill_price * trade.fill_size, 4)
                # already_above: price was above threshold when the late window opened —
                # slippage = fill vs what we saw (execution quality).
                # Otherwise: fresh crossing, slippage = fill vs threshold (standard).
                slip_base        = trigger_price if already_above else self.threshold
                trade.slippage   = round(trade.fill_price - slip_base, 4)
                log.info("FILLED  %s %s  %.4f shares @ %.4f  ($%.2f)  order=%s",
                         asset, side, trade.fill_size, trade.fill_price,
                         trade.fill_usdc, order.order_id[:16])
            except Exception as exc:
                log.error("Order failed for %s %s: %s", asset, side, exc)
                trade.status = "order_failed"

        self._trades[asset] = trade
        _write_trade(trade, self._trades_csv)

    # ── Resolution ───────────────────────────────────────────────────────────

    async def _resolve_all(self, window_end: int) -> None:
        for asset, trade in self._trades.items():
            if trade.status != "open":
                continue
            resolution = await self._infer_resolution(trade)
            if resolution is None:
                log.warning("Could not resolve %s %s — marking unresolved", asset, trade.side)
                trade.status = "unresolved"
            else:
                win = resolution >= 0.9
                pnl = ((1.0 - trade.fill_price) if win else -trade.fill_price) * trade.fill_size
                trade.resolution = resolution
                trade.pnl_usdc   = round(pnl, 4)
                trade.status     = "won" if win else "lost"
                log.info("RESOLVED  %s %s  %s  PnL=$%.4f",
                         asset, trade.side, trade.status.upper(), pnl)
            _write_trade(trade, self._trades_csv)

    async def _redeem_won_trades(self) -> None:
        """Redeem all won trades from this window on-chain via the CTF contract."""
        for asset, trade in self._trades.items():
            if trade.status != "won":
                continue
            market = self._markets.get(asset)
            if market is None:
                continue
            log.info("Redeeming  %s %s  cond=…%s", asset, trade.side, market.condition_id[-8:])
            await asyncio.to_thread(ctf_client.redeem_positions, market.condition_id, trade.side)

    async def _infer_resolution(self, trade: LiveTrade) -> float | None:
        """
        Poll the WebSocket price cache for the filled token.
        After resolution, the winning token trades near 1.0, the loser near 0.0.
        Try for up to 90 seconds.
        """
        for _ in range(18):  # 18 × 5s = 90s
            price = self._ws.get_price(trade.token_id)
            if price is not None:
                if price >= 0.95:
                    return 1.0
                if price <= 0.05:
                    return 0.0
            await asyncio.sleep(5.0)
        return None

    # ── Terminal ticker ───────────────────────────────────────────────────────

    async def _ticker_loop(self) -> None:
        """Print a price line to stdout every 5 seconds."""
        while True:
            await asyncio.sleep(5.0)
            if not self._markets:
                continue
            ws  = self._window_start
            we  = self._window_end
            elapsed   = int(time.time() - ws)
            remaining = max(0, we - int(time.time()))
            parts = [f"[{elapsed:>3}s / {remaining:>3}s left]"]
            for asset, market in self._markets.items():
                up_p = self._ws.get_price(market.up_token.token_id)
                dn_p = self._ws.get_price(market.down_token.token_id)
                up_str = f"{up_p:.3f}" if up_p is not None else "  —  "
                dn_str = f"{dn_p:.3f}" if dn_p is not None else "  —  "
                trade  = self._trades.get(asset)
                if trade and trade.status in ("won", "lost"):
                    tag = f"✓{trade.status.upper()} ${trade.pnl_usdc:+.2f}"
                elif trade and trade.status == "open":
                    tag = f"FILLED {trade.side}@{trade.fill_price:.3f}"
                elif self._filled.get(asset):
                    tag = "filled"
                else:
                    # Highlight if either side is near threshold
                    up_near = up_p is not None and up_p >= self.threshold * 0.97
                    dn_near = dn_p is not None and dn_p >= self.threshold * 0.97
                    in_window = time.time() >= ws + self.cutoff_secs
                    if not in_window:
                        tag = "waiting"
                    elif up_near or dn_near:
                        tag = "⚡ NEAR"
                    else:
                        tag = "watching"
                parts.append(f"{asset} UP={up_str} DN={dn_str} [{tag}]")
            print("  ".join(parts), flush=True)

    # ── Status broadcast ─────────────────────────────────────────────────────

    async def _status_loop(self) -> None:
        """Write status.json every 500ms for the dashboard."""
        while True:
            try:
                ws  = self._window_start
                we  = self._window_end
                now = time.time()

                asset_status = {}
                for asset in self.assets:
                    market = self._markets.get(asset)
                    up_price = dn_price = None
                    if market:
                        up_price = self._ws.get_price(market.up_token.token_id)
                        dn_price = self._ws.get_price(market.down_token.token_id)

                    trade = self._trades.get(asset)
                    asset_status[asset] = {
                        "up_price":  up_price,
                        "down_price": dn_price,
                        "filled":    self._filled.get(asset, False),
                        "trade":     asdict(trade) if trade else None,
                    }

                _write_status({
                    "updated_at":   now,
                    "window_start": ws,
                    "window_end":   we,
                    "elapsed_secs": round(now - ws),
                    "remaining_secs": round(we - now),
                    "threshold":    self.threshold,
                    "wallet_pct":    self.wallet_pct,
                    "position_usdc": self._position_usdc,
                    "dry_run":      self.dry_run,
                    "assets":       asset_status,
                }, self._status_json)
            except Exception as exc:
                log.debug("Status write failed: %s", exc)

            await asyncio.sleep(0.5)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S UTC")
