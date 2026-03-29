#!/usr/bin/env python3
"""
Audit actual slippage across live trade files.

Usage:
    python scripts/audit_slippage.py
    python scripts/audit_slippage.py --dir data/live
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


def load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # One row per order: keep latest update per order_id
    df = df[df["order_id"].notna() & (df["order_id"] != "") & (df["order_id"] != "DRY_RUN")]
    df = df.sort_values("ts").groupby("order_id", as_index=False).last()
    return df


def print_stats(label: str, df: pd.DataFrame) -> None:
    if df.empty:
        print(f"\n{label}: no real fills\n")
        return

    slip = df["slippage"]
    print(f"\n{'─' * 50}")
    print(f"  {label}  ({len(df)} fills)")
    print(f"{'─' * 50}")
    print(f"  Mean slippage : {slip.mean():+.4f}  ({slip.mean() * 100:+.2f}¢)")
    print(f"  Median        : {slip.median():+.4f}  ({slip.median() * 100:+.2f}¢)")
    print(f"  Min           : {slip.min():+.4f}  ({slip.min() * 100:+.2f}¢)")
    print(f"  Max           : {slip.max():+.4f}  ({slip.max() * 100:+.2f}¢)")
    print(f"  Std dev       : {slip.std():+.4f}")
    print()
    print(f"  {'Order':<18}  {'Asset':<6}  {'Side':<5}  {'Threshold':>9}  {'Fill':>6}  {'Slippage':>9}  Status")
    print(f"  {'─'*18}  {'─'*6}  {'─'*5}  {'─'*9}  {'─'*6}  {'─'*9}  {'─'*10}")
    for _, r in df.sort_values("ts").iterrows():
        order_short = str(r["order_id"])[:16] + "…"
        print(
            f"  {order_short:<18}  {r['asset']:<6}  {r['side']:<5}  "
            f"{r['threshold']:>9.2f}  {r['fill_price']:>6.4f}  "
            f"{r['slippage']:>+8.4f}  {r.get('status', '—')}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Audit slippage across live trade files")
    p.add_argument("--dir", default="data/live", help="Directory containing trades_*.csv files")
    args = p.parse_args()

    trade_files = sorted(Path(args.dir).glob("trades_*.csv"))
    if not trade_files:
        print(f"No trades_*.csv files found in {args.dir}")
        return

    all_frames = []
    for f in trade_files:
        df = load_trades(f)
        if not df.empty:
            df["_source"] = f.stem
            print_stats(f.stem, df)
            all_frames.append(df)

    if len(all_frames) > 1:
        combined = pd.concat(all_frames, ignore_index=True)
        print_stats("ALL INSTANCES COMBINED", combined)


if __name__ == "__main__":
    main()
