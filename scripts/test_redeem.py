#!/usr/bin/env python3
"""
Quick test for on-chain CTF redemption.

Reads won trades from data/live/trades.csv, looks up condition_ids
from Gamma, then calls redeem_positions for each.

Usage:
    python scripts/test_redeem.py              # redeem all won trades
    python scripts/test_redeem.py --dry-run    # print what would be redeemed
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import pandas as pd

from skeptic.clients import ctf as ctf_client
from skeptic import config
GAMMA_HOST = config.GAMMA_HOST

TRADES_CSV = os.path.join("data", "live", "trades.csv")


def get_condition_id(window_start_ts: int, http: httpx.Client) -> str | None:
    """Look up the condition_id for a window via the Gamma API."""
    for asset in ["DOGE", "BTC", "ETH", "SOL", "HYPE", "XRP", "BNB"]:
        slug = f"{asset.lower()}-updown-5m-{window_start_ts}"
        try:
            resp = http.get(f"{GAMMA_HOST}/events", params={"slug": slug, "limit": 1})
            events = resp.json()
            if events:
                markets = events[0].get("markets") or []
                if markets:
                    return markets[0].get("conditionId")
        except Exception:
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(TRADES_CSV)

    # Include won trades + order_failed rows that have a real order ID
    # (order_failed can mean the order went through but the NameError prevented logging)
    redeemable = df[
        (df["status"].isin(["won", "order_failed"])) &
        df["order_id"].notna() &
        (df["order_id"] != "DRY_RUN") &
        (df["order_id"] != "")
    ].drop_duplicates(subset=["order_id"])

    if redeemable.empty:
        print("No redeemable trades found in trades.csv.")
        return

    print(f"Found {len(redeemable)} redeemable trade(s):\n")

    with httpx.Client(timeout=15) as http:
        for _, row in redeemable.iterrows():
            condition_id = get_condition_id(int(row["window_start_ts"]), http)
            if condition_id is None:
                print(f"  SKIP  {row['asset']} {row['side']}  window={int(row['window_start_ts'])}  — could not find condition_id")
                continue

            print(f"  {'DRY RUN' if args.dry_run else 'REDEEM'}  "
                  f"{row['asset']} {row['side']}  "
                  f"fill={row['fill_size']:.4f} @ {row['fill_price']:.4f}  "
                  f"cond=…{condition_id[-8:]}")

            if not args.dry_run:
                tx = ctf_client.redeem_positions(condition_id, row["side"])
                print(f"    tx={tx}")


if __name__ == "__main__":
    main()
