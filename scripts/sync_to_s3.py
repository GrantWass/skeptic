#!/usr/bin/env python3
"""
Sync local data directories to S3.

Uploads prices/, orderbook/, and/or coin_prices/ to:
  s3://{S3_BUCKET}/{S3_PREFIX}/{dir}/

Requires S3_BUCKET (and optionally S3_PREFIX) to be set in .env or environment.

Usage:
    python scripts/sync_to_s3.py                           # sync all three dirs
    python scripts/sync_to_s3.py --dirs prices orderbook   # sync specific dirs
    python scripts/sync_to_s3.py --today-only              # only today's files (used by collect_and_sync)
    python scripts/sync_to_s3.py --data-root /custom/path
"""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from skeptic import config, storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ALL_DIRS = ["prices", "orderbook", "coin_prices"]

SYNC_FN = {
    "prices":      storage.sync_prices,
    "orderbook":   storage.sync_orderbook,
    "coin_prices": storage.sync_coin_prices,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sync local data to S3")
    p.add_argument(
        "--dirs", nargs="+", default=ALL_DIRS,
        choices=ALL_DIRS, metavar="DIR",
        help="Which directories to sync (default: all). Choices: prices, orderbook, coin_prices",
    )
    p.add_argument(
        "--data-root", default="data",
        help="Root directory containing prices/, orderbook/, coin_prices/ (default: data)",
    )
    p.add_argument(
        "--today-only", action="store_true",
        help="Only upload today's UTC-dated file (prices + orderbook). "
             "Faster for periodic syncs during collection.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not config.S3_BUCKET:
        log.error("S3_BUCKET is not set. Add it to your .env file and try again.")
        sys.exit(1)

    log.info("Bucket     : s3://%s", config.S3_BUCKET)
    log.info("Prefix     : %s", config.S3_PREFIX)
    log.info("Today only : %s", args.today_only)

    data_root = Path(args.data_root)
    total = 0

    for dir_name in args.dirs:
        local_dir = data_root / dir_name
        if not local_dir.exists():
            log.warning("%s does not exist — skipping", local_dir)
            continue
        log.info("Syncing %s/ → s3://%s/%s/%s/", local_dir, config.S3_BUCKET, config.S3_PREFIX, dir_name)
        n = SYNC_FN[dir_name](local_dir, today_only=args.today_only)
        log.info("  %d file(s) uploaded", n)
        total += n

    log.info("Done — %d file(s) uploaded total.", total)


if __name__ == "__main__":
    main()
