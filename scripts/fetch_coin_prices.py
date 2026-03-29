#!/usr/bin/env python3
"""
Fetch 1-second OHLCV data from Binance for assets matching existing price CSVs.

Reads the time range from data/prices/*.csv, then pulls Binance klines at 1s
granularity for each asset and saves to data/coin_prices/.

Output columns: ts (unix seconds), open, high, low, close, volume

Usage:
    python scripts/fetch_coin_prices.py
    python scripts/fetch_coin_prices.py --assets BTC DOGE --prices-dir data/prices
    python scripts/fetch_coin_prices.py --start 1774656000 --end 1774900000
"""
import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from skeptic import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
ASSET_TO_SYMBOL = {
    "BTC":  "BTCUSDT",
    "ETH":  "ETHUSDT",
    "SOL":  "SOLUSDT",
    "DOGE": "DOGEUSDT",
    "XRP":  "XRPUSDT",
    "BNB":  "BNBUSDT",
    "HYPE": "HYPEUSDT",
}
CHUNK_SECS  = 1000   # 1000 x 1s candles per request
RATE_LIMIT_DELAY = 0.12  # ~8 req/s, well within Binance's 1200 req/min limit
OUTPUT_DIR  = os.path.join("data", "coin_prices")


def _get_price_time_range(prices_dir: str) -> tuple[int, int]:
    """Scan price CSV files to find the overall min/max timestamp."""
    csv_files = sorted(Path(prices_dir).glob("prices_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No prices_*.csv files found in {prices_dir}")

    ts_min = float("inf")
    ts_max = float("-inf")
    for f in csv_files:
        with open(f, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    ts = int(row["ts"])
                    if ts < ts_min:
                        ts_min = ts
                    if ts > ts_max:
                        ts_max = ts
                except (KeyError, ValueError):
                    continue
    if ts_min == float("inf"):
        raise ValueError("No valid timestamps found in price CSV files")
    return int(ts_min), int(ts_max)


def _fetch_klines_chunk(
    client: httpx.Client,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> list[list]:
    """Fetch up to 1000 1s klines from Binance. Returns raw list of lists."""
    resp = client.get(
        BINANCE_KLINES_URL,
        params={
            "symbol":    symbol,
            "interval":  "1s",
            "startTime": start_ms,
            "endTime":   end_ms,
            "limit":     1000,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_and_save(
    symbol: str,
    start_ts: int,
    end_ts: int,
    output_path: str,
) -> int:
    """
    Fetch 1s Binance klines for [start_ts, end_ts] and write to output_path.
    Returns total candles written.
    """
    total = 0
    write_header = not os.path.exists(output_path)

    with httpx.Client() as client, open(output_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["ts", "open", "high", "low", "close", "volume"])

        cursor = start_ts * 1000  # ms
        end_ms  = end_ts   * 1000

        while cursor < end_ms:
            chunk_end = min(cursor + CHUNK_SECS * 1000 - 1, end_ms)
            try:
                rows = _fetch_klines_chunk(client, symbol, cursor, chunk_end)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400:
                    # Symbol not listed on Binance
                    log.warning("%s: HTTP 400 — symbol may not exist on Binance, skipping", symbol)
                    return 0
                raise

            if not rows:
                break

            for row in rows:
                # row: [open_time_ms, open, high, low, close, volume, close_time_ms, ...]
                ts_sec = int(row[0]) // 1000
                writer.writerow([ts_sec, row[1], row[2], row[3], row[4], row[5]])
                total += 1

            # Advance cursor past the last candle returned
            last_open_ms = int(rows[-1][0])
            cursor = last_open_ms + 1000  # next second

            time.sleep(RATE_LIMIT_DELAY)

        log.info("%s: wrote %d candles → %s", symbol, total, output_path)
    return total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch 1s Binance klines for Polymarket assets")
    p.add_argument("--assets", nargs="+", default=list(ASSET_TO_SYMBOL.keys()),
                   help="Assets to fetch (default: all configured)")
    p.add_argument("--prices-dir", default="data/prices",
                   help="Directory of prices_*.csv files used to infer time range")
    p.add_argument("--start", type=int, default=None,
                   help="Override start Unix timestamp (seconds)")
    p.add_argument("--end",   type=int, default=None,
                   help="Override end Unix timestamp (seconds)")
    p.add_argument("--out-dir", default=OUTPUT_DIR,
                   help="Output directory for coin price CSVs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.start and args.end:
        start_ts, end_ts = args.start, args.end
    else:
        log.info("Scanning %s for time range…", args.prices_dir)
        start_ts, end_ts = _get_price_time_range(args.prices_dir)
        if args.start:
            start_ts = args.start
        if args.end:
            end_ts = args.end

    duration_h = (end_ts - start_ts) / 3600
    log.info("Time range: %d → %d  (%.1f hours)", start_ts, end_ts, duration_h)

    os.makedirs(args.out_dir, exist_ok=True)

    for asset in args.assets:
        symbol = ASSET_TO_SYMBOL.get(asset.upper())
        if symbol is None:
            log.warning("No Binance symbol mapping for %s — skipping", asset)
            continue

        out_path = os.path.join(args.out_dir, f"{symbol}_1s.csv")
        log.info("Fetching %s (%s) → %s", asset, symbol, out_path)
        n = fetch_and_save(symbol, start_ts, end_ts, out_path)
        if n == 0:
            log.warning("No data written for %s", symbol)

    log.info("Done.")


if __name__ == "__main__":
    main()
