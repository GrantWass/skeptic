import time as _time
import asyncio

WINDOW_SECS = 300


def current_window_start() -> int:
    """Unix timestamp of the currently active 5-minute window."""
    return (int(_time.time()) // WINDOW_SECS) * WINDOW_SECS


def next_window_start() -> int:
    """Unix timestamp of the next 5-minute window boundary."""
    return current_window_start() + WINDOW_SECS


def seconds_until_next_window() -> float:
    return next_window_start() - _time.time()


def market_slug(asset: str, window_ts: int) -> str:
    """Build the Polymarket market slug for a given asset and window timestamp."""
    return f"{asset.lower()}-updown-5m-{window_ts}"


async def sleep_until(target_ts: float, precision_secs: float = 0.05) -> None:
    """
    Sleep until target_ts (unix float). Uses coarse sleep then a tight poll
    loop in the final `precision_secs` seconds for accuracy.
    """
    coarse_wait = target_ts - _time.time() - precision_secs
    if coarse_wait > 0:
        await asyncio.sleep(coarse_wait)
    while _time.time() < target_ts:
        await asyncio.sleep(0.01)
