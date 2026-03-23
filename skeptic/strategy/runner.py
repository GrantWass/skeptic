"""
Runner: outer loop that manages window alignment and fires concurrent
SessionExecutors for each configured asset.
"""
import asyncio
import logging

import httpx
from py_clob_client.client import ClobClient
from rich.console import Console

from skeptic import config
from skeptic.clients import clob as clob_client
from skeptic.clients import gamma as gamma_client
from skeptic.clients.ws import UserChannel, MarketChannel
from skeptic.models.market import Market
from skeptic.storage import db
from skeptic.strategy.executor import SessionExecutor
from skeptic.utils.time import next_window_start, seconds_until_next_window, sleep_until

log = logging.getLogger(__name__)
console = Console()


async def run(
    assets: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    """
    Main trading loop. Runs indefinitely, one iteration per 5-minute window.

    Args:
        assets: Asset list to trade. Defaults to config.ASSETS.
        dry_run: If True, log orders without submitting to the CLOB.
    """
    if assets is None:
        assets = config.ASSETS

    db.init_db()

    # Build CLOB client
    client: ClobClient = clob_client.build_client()
    balance = clob_client.get_usdc_balance(client)
    console.print(f"[green]CLOB client ready. Balance: ${balance:.2f} USDC[/green]")

    # Start WebSocket channels
    user_ws = UserChannel(
        api_key=config.CLOB_API_KEY,
        secret=config.CLOB_SECRET,
        passphrase=config.CLOB_PASSPHRASE,
    )
    market_ws = MarketChannel()

    user_task = asyncio.create_task(user_ws.run(), name="user-ws")
    market_task = asyncio.create_task(market_ws.run(), name="market-ws")

    try:
        async with httpx.AsyncClient() as http:
            while True:
                window_ts = next_window_start()
                secs = seconds_until_next_window()
                console.print(
                    f"[cyan]Next window @ {window_ts} ({secs:.1f}s away) — "
                    f"assets: {', '.join(assets)}[/cyan]"
                )

                # --- Pre-fetch markets and subscribe to tokens while waiting ---
                fetch_task = asyncio.create_task(
                    _prefetch_markets(assets, window_ts, http, market_ws)
                )

                # Sleep until window boundary
                await sleep_until(float(window_ts))

                markets = await fetch_task

                if not markets:
                    log.warning("No markets found for window %d — skipping", window_ts)
                    await asyncio.sleep(5)
                    continue

                # Refresh balance
                balance = clob_client.get_usdc_balance(client)
                capital_per_asset = balance / len(markets) if markets else 0.0

                console.print(
                    f"[yellow]Window {window_ts} open — "
                    f"balance=${balance:.2f}, {len(markets)} markets, "
                    f"${capital_per_asset:.2f}/asset[/yellow]"
                )

                # Run all asset sessions concurrently
                tasks = [
                    asyncio.create_task(
                        SessionExecutor(client, user_ws, market_ws, dry_run=dry_run).run(
                            market, capital_per_asset
                        ),
                        name=f"session-{market.asset}",
                    )
                    for market in markets
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for market, result in zip(markets, results):
                    if isinstance(result, Exception):
                        log.error("[%s] Session failed: %s", market.asset, result)
                    else:
                        fill_str = f"fill={result.filled_outcome}" if result.fill_occurred else "no fill"
                        console.print(f"  [dim]{market.asset}: {fill_str}[/dim]")

    finally:
        user_task.cancel()
        market_task.cancel()
        await asyncio.gather(user_task, market_task, return_exceptions=True)


async def _prefetch_markets(
    assets: list[str],
    window_ts: int,
    http: httpx.AsyncClient,
    market_ws: MarketChannel,
) -> list[Market]:
    """Fetch markets for the upcoming window and subscribe to their tokens."""
    tasks = [
        gamma_client.get_current_window_market(asset, window_ts, http)
        for asset in assets
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    markets = []
    tokens_to_subscribe = []
    for asset, result in zip(assets, results):
        if isinstance(result, Exception):
            log.error("Failed to fetch market for %s: %s", asset, result)
        elif result is None:
            log.warning("No market found for %s at window %d", asset, window_ts)
        else:
            markets.append(result)
            tokens_to_subscribe.extend([
                result.up_token.token_id,
                result.down_token.token_id,
            ])

    if tokens_to_subscribe:
        await market_ws.subscribe(*tokens_to_subscribe)

    return markets
