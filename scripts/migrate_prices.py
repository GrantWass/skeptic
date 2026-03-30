#!/usr/bin/env python3
"""
Migrate prices_*.csv files:
  Add missing columns (up_bid, up_ask, up_spread, up_imbalance,
  dn_bid, dn_ask, dn_spread, dn_imbalance) — left empty where absent.

Usage:
    python scripts/migrate_prices.py
    python scripts/migrate_prices.py --prices-dir data/prices
"""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FULL_COLUMNS = [
    "ts", "window_ts", "asset",
    "up_price", "down_price",
    "up_bid", "up_ask", "up_spread", "up_imbalance",
    "dn_bid", "dn_ask", "dn_spread", "dn_imbalance",
]


def migrate_file(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)

    extra_cols = ["up_bid", "up_ask", "up_spread", "up_imbalance",
                  "dn_bid", "dn_ask", "dn_spread", "dn_imbalance"]
    for col in extra_cols:
        if col not in df.columns:
            df[col] = ""

    # Reorder to canonical column order
    ordered = [c for c in FULL_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in FULL_COLUMNS]
    df = df[ordered + extra]

    df.to_csv(csv_path, index=False)
    log.info("Updated %s (%d rows)", csv_path.name, len(df))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prices-dir", default="data/prices")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    csv_files = sorted(Path(args.prices_dir).glob("prices_*.csv"))
    if not csv_files:
        log.error("No prices_*.csv files found in %s", args.prices_dir)
        sys.exit(1)

    for f in csv_files:
        migrate_file(f)

    log.info("Done. %d files updated.", len(csv_files))


if __name__ == "__main__":
    main()
