"""
Gamma REST API client for market discovery and historical data.
No authentication required.

Key findings:
- 5-minute markets live under the /events endpoint (not /markets)
- Each event has a "markets" array with sub-markets: outcomes[0]="Up", outcomes[1]="Down"
- Historical (closed) events use archived=true (not closed=true)
- Active events use active=true&closed=false
- Slug pattern: {asset}-updown-5m-{unix_timestamp}
"""
import asyncio
import json
import logging

import httpx

from skeptic import config
from skeptic.models.market import Market, Token
from skeptic.utils.time import market_slug

logger = logging.getLogger("gamma")

_HEADERS = {"Accept": "application/json"}


def _parse_event(event: dict) -> Market | None:
    """Parse a raw Gamma event dict (containing sub-markets) into a Market dataclass."""
    try:
        slug = event.get("slug", "") or event.get("ticker", "")
        sub_markets = event.get("markets", [])

        if not sub_markets:
            logger.warning("Event %s has no sub-markets", slug)
            return None

        # Use the first sub-market for conditionId and token IDs
        sub = sub_markets[0]
        clob_ids_raw = sub.get("clobTokenIds") or []
        # clobTokenIds may be a JSON-encoded string or already a list
        if isinstance(clob_ids_raw, str):
            try:
                clob_ids = json.loads(clob_ids_raw)
            except json.JSONDecodeError:
                clob_ids = []
        else:
            clob_ids = clob_ids_raw

        outcomes = sub.get("outcomes") or ["Up", "Down"]

        if len(clob_ids) < 2:
            logger.warning("Event %s missing clobTokenIds", slug)
            return None

        # outcomes[0] = "Up", outcomes[1] = "Down"
        up_token = Token(token_id=clob_ids[0], outcome="UP")
        down_token = Token(token_id=clob_ids[1], outcome="DOWN")

        # Extract asset from slug: "{asset}-updown-5m-{ts}"
        parts = slug.split("-")
        asset = parts[0].upper() if parts else "UNKNOWN"

        # Parse timestamps
        start_ts = 0
        end_ts = 0
        try:
            slug_ts = int(parts[-1])
            start_ts = slug_ts
            end_ts = slug_ts + 300
        except (ValueError, IndexError):
            pass

        condition_id = sub.get("conditionId", "")

        return Market(
            condition_id=condition_id,
            slug=slug,
            asset=asset,
            start_ts=start_ts,
            end_ts=end_ts,
            up_token=up_token,
            down_token=down_token,
            active=event.get("active", False),
        )
    except Exception as e:
        logger.warning("Failed to parse event: %s — %s", event.get("slug"), e)
        return None


async def get_event_by_slug(slug: str, client: httpx.AsyncClient) -> Market | None:
    """Fetch a single 5-min market event by its exact slug."""
    try:
        resp = await client.get(
            f"{config.GAMMA_HOST}/events",
            params={"slug": slug, "limit": 5},
            headers=_HEADERS,
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        events = data if isinstance(data, list) else data.get("events", [])
        for evt in events:
            if evt.get("slug") == slug or evt.get("ticker") == slug:
                return _parse_event(evt)
        return None
    except Exception as e:
        logger.error("get_event_by_slug(%s) failed: %s", slug, e)
        return None


async def get_current_window_market(
    asset: str,
    window_ts: int,
    client: httpx.AsyncClient,
    retries: int = 60,
    retry_delay: float = 5.0,
) -> Market | None:
    """
    Fetch the market for `asset` starting at `window_ts`.
    Retries up to `retries` times to handle late market creation.
    """
    slug = market_slug(asset, window_ts)
    for attempt in range(retries):
        market = await get_event_by_slug(slug, client)
        if market is not None:
            return market
        if attempt < retries - 1:
            logger.debug("Market %s not found, retrying in %.1fs…", slug, retry_delay)
            await asyncio.sleep(retry_delay)
    logger.warning("Market %s not found after %d attempts", slug, retries)
    return None


async def get_historical_markets(
    asset: str,
    client: httpx.AsyncClient,
    limit: int = 100,
    offset: int = 0,
) -> list[Market]:
    """
    Fetch historical (archived) 5-minute market events for an asset.
    Uses archived=true — closed 5-min events appear under archived, not closed.
    """
    slug_prefix = f"{asset.lower()}-updown-5m"
    markets: list[Market] = []
    fetched = 0
    page_size = 200  # fetch large pages since 5m events are ~57% of archived events

    while fetched < limit:
        try:
            resp = await client.get(
                f"{config.GAMMA_HOST}/events",
                params={
                    "archived": "true",
                    "order": "createdAt",
                    "ascending": "false",
                    "limit": page_size,
                    "offset": offset + fetched,
                },
                headers=_HEADERS,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            raw_list = data if isinstance(data, list) else data.get("events", [])

            if not raw_list:
                break

            for raw in raw_list:
                evt_slug = raw.get("slug", "") or raw.get("ticker", "")
                if evt_slug.startswith(slug_prefix):
                    m = _parse_event(raw)
                    if m:
                        markets.append(m)

            fetched += len(raw_list)

            # If we got fewer than requested, no more pages
            if len(raw_list) < page_size:
                break

        except Exception as e:
            logger.error("get_historical_markets(%s) page failed: %s", asset, e)
            break

    logger.info("get_historical_markets(%s): found %d 5m events from %d archived fetched",
                asset, len(markets), fetched)
    return markets


async def get_all_5m_assets(client: httpx.AsyncClient) -> list[str]:
    """
    Discover all assets with active 5-minute markets by querying the events endpoint.
    """
    found: set[str] = set()
    try:
        resp = await client.get(
            f"{config.GAMMA_HOST}/events",
            params={"active": "true", "closed": "false", "limit": 100,
                    "order": "createdAt", "ascending": "false"},
            headers=_HEADERS,
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_list = data if isinstance(data, list) else data.get("events", [])
        for raw in raw_list:
            slug = raw.get("slug", "") or raw.get("ticker", "")
            if "updown-5m" in slug:
                parts = slug.split("-")
                found.add(parts[0].upper())
    except Exception as e:
        logger.error("get_all_5m_assets() failed: %s", e)
    return sorted(found)
