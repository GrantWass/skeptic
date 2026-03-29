"""
Fetches historical 5-minute market data for the research phase.

For each closed market, we pull all trades from the CLOB and reconstruct:
  - Prices in the first 60 seconds (minute-1 proxy)
  - Final resolution price (1.0 or 0.0 per outcome)

Note: Intra-session price reconstruction from trades is an approximation.
The CLOB `get_trades` endpoint returns all trades chronologically, so we use
the first-60-second window to proxy what our buy order would have seen.
"""
import asyncio
import logging
import time as _time
from dataclasses import dataclass, field

import httpx
from py_clob_client.client import ClobClient

from skeptic import config
from skeptic.clients import gamma as gamma_client
from skeptic.clients import clob as clob_client
from skeptic.models.market import Market

log = logging.getLogger(__name__)


@dataclass
class HistoricalSession:
    asset: str
    condition_id: str
    window_start_ts: int
    up_token_id: str
    down_token_id: str

    # Trade (timestamp, price) pairs during the first 60 seconds for each outcome
    up_trades_m1: list[tuple[int, float]] = field(default_factory=list)   # (ts, price) ≤60s
    down_trades_m1: list[tuple[int, float]] = field(default_factory=list)

    # All trade (timestamp, price) pairs during the full window, chronological
    up_trades_all: list[tuple[int, float]] = field(default_factory=list)
    down_trades_all: list[tuple[int, float]] = field(default_factory=list)

    # Resolution: 1.0 if UP won, 0.0 if DOWN won; None if unresolvable
    up_resolution: float | None = None
    down_resolution: float | None = None

    @property
    def up_min_m1(self) -> float | None:
        return min(p for _, p in self.up_trades_m1) if self.up_trades_m1 else None

    @property
    def down_min_m1(self) -> float | None:
        return min(p for _, p in self.down_trades_m1) if self.down_trades_m1 else None

    def up_first_fill_ts(self, buy_threshold: float) -> int | None:
        """Timestamp of the first UP trade at or below buy_threshold in minute 1."""
        return next((ts for ts, p in self.up_trades_m1 if p <= buy_threshold), None)

    def down_first_fill_ts(self, buy_threshold: float) -> int | None:
        """Timestamp of the first DOWN trade at or below buy_threshold in minute 1."""
        return next((ts for ts, p in self.down_trades_m1 if p <= buy_threshold), None)

    def up_max_after_fill(self, buy_threshold: float) -> float | None:
        """Max UP price after the first buy fill in minute 1 at or below buy_threshold."""
        fill_ts = next(
            (ts for ts, p in self.up_trades_m1 if p <= buy_threshold), None
        )
        if fill_ts is None:
            return None
        subsequent = [p for ts, p in self.up_trades_all if ts > fill_ts]
        return max(subsequent) if subsequent else None

    def down_max_after_fill(self, buy_threshold: float) -> float | None:
        """Max DOWN price after the first buy fill in minute 1 at or below buy_threshold."""
        fill_ts = next(
            (ts for ts, p in self.down_trades_m1 if p <= buy_threshold), None
        )
        if fill_ts is None:
            return None
        subsequent = [p for ts, p in self.down_trades_all if ts > fill_ts]
        return max(subsequent) if subsequent else None


async def fetch_sessions_for_asset(
    asset: str,
    clob: ClobClient,
    http: httpx.AsyncClient,
    limit: int = 100,
    offset: int = 0,
) -> list[HistoricalSession]:
    """Fetch and reconstruct historical sessions for a given asset."""
    log.info("Fetching historical markets for %s (limit=%d, offset=%d)…", asset, limit, offset)
    markets = await gamma_client.get_historical_markets(asset, http, limit=limit, offset=offset)
    log.info("Found %d closed markets for %s", len(markets), asset)

    sessions: list[HistoricalSession] = []
    for market in markets:
        session = await _build_session(market, clob)
        if session is not None:
            sessions.append(session)
        await asyncio.sleep(0.1)  # gentle rate limiting

    return sessions


async def _build_session(market: Market, clob: ClobClient) -> HistoricalSession | None:
    """Pull trades for both tokens and build a HistoricalSession."""
    try:
        window_end = market.start_ts + 300  # 5 minutes

        up_trades = await asyncio.to_thread(clob_client.get_trades, clob, market.up_token.token_id)
        down_trades = await asyncio.to_thread(clob_client.get_trades, clob, market.down_token.token_id)

        session = HistoricalSession(
            asset=market.asset,
            condition_id=market.condition_id,
            window_start_ts=market.start_ts,
            up_token_id=market.up_token.token_id,
            down_token_id=market.down_token.token_id,
        )

        for trade in up_trades:
            ts = int(trade.get("created_at", 0) or trade.get("timestamp", 0) or 0)
            price = float(trade.get("price", 0))
            if price <= 0:
                continue
            if market.start_ts <= ts <= window_end:
                session.up_trades_all.append((ts, price))
                if ts <= market.start_ts + config.MONITOR_SECS:
                    session.up_trades_m1.append((ts, price))

        for trade in down_trades:
            ts = int(trade.get("created_at", 0) or trade.get("timestamp", 0) or 0)
            price = float(trade.get("price", 0))
            if price <= 0:
                continue
            if market.start_ts <= ts <= window_end:
                session.down_trades_all.append((ts, price))
                if ts <= market.start_ts + config.MONITOR_SECS:
                    session.down_trades_m1.append((ts, price))

        # Resolution: derive from market outcomePrices or final trade prices
        # At resolution UP = 1.0 or 0.0; DOWN = complementary
        if session.up_trades_all:
            final_up = session.up_trades_all[-1][1]
            session.up_resolution = 1.0 if final_up >= 0.9 else 0.0
            session.down_resolution = 1.0 - session.up_resolution

        return session

    except Exception as e:
        log.warning("Failed to build session for %s/%s: %s", market.asset, market.condition_id, e)
        return None


def load_from_price_files(
    assets: list[str],
    prices_dir: str = "data/prices",
    min_points: int = 280,
    last_days: int | None = None,
) -> dict[str, list[HistoricalSession]]:
    """
    Build HistoricalSession objects from the per-second price CSV files
    written by scripts/collect_prices.py.

    CSV columns: ts, window_ts, asset, up_price, down_price
    """
    import csv
    import os
    from pathlib import Path

    result: dict[str, list[HistoricalSession]] = {a: [] for a in assets}
    prices_path = Path(prices_dir)

    if not prices_path.exists():
        log.warning("prices dir %s not found", prices_dir)
        return result

    csv_files = sorted(prices_path.glob("prices_*.csv"))
    if last_days is not None:
        csv_files = csv_files[-last_days:]
    if not csv_files:
        log.warning("No price CSV files found in %s", prices_dir)
        return result

    # Group rows by (asset, window_ts)
    # key → list of (ts, up_price, down_price)
    buckets: dict[tuple[str, int], list[tuple[int, float | None, float | None]]] = {}

    for csv_file in csv_files:
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                asset = row.get("asset", "").upper()
                if assets and asset not in assets:
                    continue
                try:
                    ts = int(row["ts"])
                    window_ts = int(row["window_ts"])
                    up = float(row["up_price"]) if row.get("up_price") else None
                    dn = float(row["down_price"]) if row.get("down_price") else None
                    buckets.setdefault((asset, window_ts), []).append((ts, up, dn))
                except (ValueError, KeyError):
                    continue

    for (asset, window_ts), rows in buckets.items():
        if asset not in result:
            continue
        rows.sort(key=lambda r: r[0])  # sort by timestamp

        m1_cutoff = window_ts + config.MONITOR_SECS
        window_end = window_ts + 300

        session = HistoricalSession(
            asset=asset,
            condition_id="",
            window_start_ts=window_ts,
            up_token_id="",
            down_token_id="",
        )

        for ts, up, dn in rows:
            if up is not None:
                session.up_trades_all.append((ts, up))
                if ts <= m1_cutoff:
                    session.up_trades_m1.append((ts, up))
            if dn is not None:
                session.down_trades_all.append((ts, dn))
                if ts <= m1_cutoff:
                    session.down_trades_m1.append((ts, dn))

        # Derive resolution from last price in window
        # Final price should be near 1.0 (win) or 0.0 (loss) after resolution
        last_up = session.up_trades_all[-1][1] if session.up_trades_all else None
        last_dn = session.down_trades_all[-1][1] if session.down_trades_all else None
        if last_up is not None:
            session.up_resolution = 1.0 if last_up >= 0.9 else (0.0 if last_up <= 0.1 else None)
        if last_dn is not None:
            session.down_resolution = 1.0 if last_dn >= 0.9 else (0.0 if last_dn <= 0.1 else None)

        if min_points > 0 and len(session.up_trades_all) < min_points:
            log.debug("Skipping incomplete window %s/%s (%d points)", asset, window_ts, len(session.up_trades_all))
            continue
        result[asset].append(session)

    for asset in assets:
        log.info("load_from_price_files(%s): %d sessions loaded", asset, len(result.get(asset, [])))

    return result


async def fetch_all_assets(
    assets: list[str],
    clob: ClobClient,
    limit: int = 100,
) -> dict[str, list[HistoricalSession]]:
    """Fetch historical sessions for all assets. Returns dict[asset → sessions]."""
    result: dict[str, list[HistoricalSession]] = {}
    async with httpx.AsyncClient(timeout=30.0) as http:
        for asset in assets:
            sessions = await fetch_sessions_for_asset(asset, clob, http, limit=limit)
            result[asset] = sessions
            log.info("%s: %d sessions with trade data", asset, len(sessions))
    return result
