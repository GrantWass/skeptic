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
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import pandas as pd

from skeptic import config, storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%M:%S")
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
RATE_LIMIT_DELAY = 0.05  # ~20 req/s average; back off on 429s if needed
OUTPUT_DIR  = os.path.join("data", "coin_prices")


def _get_price_time_range(prices_dir: str) -> tuple[int, int]:
    """Scan price CSV files to find the overall min/max timestamp."""
    csv_files = storage.list_csv_paths(prices_dir, "prices_*.csv")
    if not csv_files:
        raise FileNotFoundError(f"No prices_*.csv files found in {prices_dir}")

    ts_min = float("inf")
    ts_max = float("-inf")
    for f in csv_files:
        df = storage.read_csv(f, usecols=["ts"])
        if df.empty:
            continue
        ts_col = pd.to_numeric(df["ts"], errors="coerce").dropna()
        if ts_col.empty:
            continue
        file_min = int(ts_col.min())
        file_max = int(ts_col.max())
        if file_min < ts_min:
            ts_min = file_min
        if file_max > ts_max:
            ts_max = file_max
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


def _last_ts_in_file(path: str) -> int | None:
    """Return the last 'ts' value in an existing CSV, or None if file is empty/missing."""
    if not os.path.exists(path):
        return None
    last_ts = None
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                last_ts = int(row["ts"])
            except (KeyError, ValueError):
                continue
    return last_ts


def _first_ts_in_file(path: str) -> int | None:
    """Return the first valid 'ts' value in an existing CSV, or None if missing."""
    if not os.path.exists(path):
        return None
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                return int(row["ts"])
            except (KeyError, ValueError):
                continue
    return None


def _write_klines_range(
    client: httpx.Client,
    symbol: str,
    start_ts: int,
    end_ts: int,
    writer: Any,
    rate_limit_delay: float,
) -> int:
    """Write klines for [start_ts, end_ts] into writer. Returns rows written."""
    if start_ts > end_ts:
        return 0

    written = 0
    cursor = start_ts * 1000
    end_ms = end_ts * 1000

    while cursor <= end_ms:
        chunk_end = min(cursor + CHUNK_SECS * 1000 - 1, end_ms)
        retries = 0
        while True:
            try:
                rows = _fetch_klines_chunk(client, symbol, cursor, chunk_end)
                break
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status in (429, 418, 500, 502, 503, 504) and retries < 5:
                    sleep_s = max(rate_limit_delay, 0.5) * (2**retries)
                    log.warning("%s: HTTP %d, backing off %.1fs", symbol, status, sleep_s)
                    time.sleep(sleep_s)
                    retries += 1
                    continue
                raise
        if not rows:
            break

        for row in rows:
            ts_sec = int(row[0]) // 1000
            writer.writerow([ts_sec, row[1], row[2], row[3], row[4], row[5]])
            written += 1

        last_open_ms = int(rows[-1][0])
        cursor = last_open_ms + 1000
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

    return written


def _prepend_csv_rows(prepend_path: str, existing_path: str) -> None:
    """Prepend CSV data rows from prepend_path to existing_path, keeping one header."""
    merged_path = existing_path + ".merge.tmp"

    with open(merged_path, "w", newline="") as out_fh:
        writer = csv.writer(out_fh)
        writer.writerow(["ts", "open", "high", "low", "close", "volume"])

        for src in (prepend_path, existing_path):
            if not os.path.exists(src):
                continue
            with open(src, newline="") as in_fh:
                reader = csv.reader(in_fh)
                next(reader, None)  # skip header
                for row in reader:
                    if row:
                        writer.writerow(row)

    os.replace(merged_path, existing_path)


def fetch_and_save(
    symbol: str,
    start_ts: int,
    end_ts: int,
    output_path: str,
    rate_limit_delay: float,
) -> int:
    """
    Fetch 1s Binance klines for [start_ts, end_ts] and append to output_path.
    Skips data already present by resuming from the last recorded timestamp.
    Returns total candles written.
    """
    total = 0
    first_ts = _first_ts_in_file(output_path)

    try:
        with httpx.Client() as client:
            # Backfill missing early range by prepending, preserving existing rows.
            if first_ts is not None and start_ts < first_ts:
                backfill_end = min(end_ts, first_ts - 1)
                if start_ts <= backfill_end:
                    log.info(
                        "%s: backfilling missing early range %d..%d (preserving existing rows)",
                        symbol,
                        start_ts,
                        backfill_end,
                    )
                    prepend_path = output_path + ".prepend.tmp"
                    with open(prepend_path, "w", newline="") as pre_fh:
                        pre_writer = csv.writer(pre_fh)
                        pre_writer.writerow(["ts", "open", "high", "low", "close", "volume"])
                        wrote = _write_klines_range(client, symbol, start_ts, backfill_end, pre_writer, rate_limit_delay)

                    if wrote > 0:
                        _prepend_csv_rows(prepend_path, output_path)
                    os.remove(prepend_path)
                    total += wrote

            # Continue normal append-resume behavior for missing tail range.
            last_ts = _last_ts_in_file(output_path)
            if last_ts is not None:
                append_start = max(start_ts, last_ts + 1)
                if append_start > end_ts:
                    if total == 0:
                        log.info("%s: already up-to-date (last ts %d), skipping", symbol, last_ts)
                    log.info("%s: wrote %d candles → %s", symbol, total, output_path)
                    return total
                if append_start > start_ts:
                    log.info(
                        "%s: resuming from ts %d (skipping %d already-fetched seconds)",
                        symbol,
                        append_start,
                        append_start - start_ts,
                    )
            else:
                append_start = start_ts

            write_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
            with open(output_path, "a", newline="") as fh:
                writer = csv.writer(fh)
                if write_header:
                    writer.writerow(["ts", "open", "high", "low", "close", "volume"])
                total += _write_klines_range(client, symbol, append_start, end_ts, writer, rate_limit_delay)

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:
            # Symbol not listed on Binance
            log.warning("%s: HTTP 400 — symbol may not exist on Binance, skipping", symbol)
            return 0
        raise

    log.info("%s: wrote %d candles → %s", symbol, total, output_path)
    return total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch 1s Binance klines for Polymarket assets")
    p.add_argument("--assets", nargs="+", default=list(ASSET_TO_SYMBOL.keys()),
                   help="Assets to fetch (default: all configured)")
    p.add_argument("--prices-dir", default=storage.default_data_location("prices", "data/prices"),
                   help="Directory of prices_*.csv files used to infer time range")
    p.add_argument("--start", type=int, default=None,
                   help="Override start Unix timestamp (seconds)")
    p.add_argument("--end",   type=int, default=None,
                   help="Override end Unix timestamp (seconds)")
    p.add_argument("--out-dir", default=OUTPUT_DIR,
                   help="Output directory for coin price CSVs")
    p.add_argument("--rate-limit-delay", type=float, default=RATE_LIMIT_DELAY,
                   help="Seconds to sleep between requests (default: %(default)s)")
    p.add_argument("--start-january", action="store_true",
                   help="Set start timestamp to Jan 1 and end to Feb 1 of the current year (UTC), fetches January only.")
    return p.parse_args()


def main() -> None:
    args = parse_args()


    # Determine start and end timestamps
    if args.start_january:
        # Jan 1 00:00:00 UTC → Feb 1 00:00:00 UTC (January only)
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        start_ts = int(datetime(now.year, 1, 1, tzinfo=timezone.utc).timestamp())
        end_ts   = int(datetime(now.year, 2, 1, tzinfo=timezone.utc).timestamp())
        log.info(
            "--start-january: fetching Jan 1 → Feb 1, %d UTC (%d → %d)",
            now.year, start_ts, end_ts,
        )
    elif args.start and args.end:
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
        n = fetch_and_save(symbol, start_ts, end_ts, out_path, args.rate_limit_delay)
        if n == 0:
            log.warning("No data written for %s", symbol)

    log.info("Done.")


if __name__ == "__main__":
    main()


