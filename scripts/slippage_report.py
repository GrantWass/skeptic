#!/usr/bin/env python3
"""
Slippage report for live momentum trades.

Slippage = fill_price - trigger_pm_price (positive = paid more than mid at trigger).

Usage:
    python scripts/slippage_report.py
    python scripts/slippage_report.py --csv data/live/trades_momentum.csv
"""
import argparse
import sys
import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/live/trades_momentum.csv")
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    # drop dry runs and keep one row per trade (the resolved/final status row)
    df = df[df["order_id"] != "DRY_RUN"]
    df = df[df["status"].isin(["won", "lost", "unresolved"])]

    if df.empty:
        print("No live trades found.")
        sys.exit(0)

    print(f"Live trades: {len(df)}\n")

    # ── per asset ─────────────────────────────────────────────────────────────
    print("=== Slippage by Asset ===\n")
    rows = []
    for asset, grp in df.groupby("asset"):
        s = grp["slippage"].dropna()
        if s.empty:
            continue
        rows.append({
            "asset":       asset,
            "n":           len(s),
            "mean":        round(float(s.mean()), 4),
            "median":      round(float(s.median()), 4),
            "p25":         round(float(s.quantile(0.25)), 4),
            "p75":         round(float(s.quantile(0.75)), 4),
            "worst":       round(float(s.max()), 4),
            "best":        round(float(s.min()), 4),
        })
    if rows:
        print(pd.DataFrame(rows).sort_values("mean").to_string(index=False))

    # ── per asset + side ──────────────────────────────────────────────────────
    print("\n\n=== Slippage by Asset + Side ===\n")
    rows = []
    for (asset, side), grp in df.groupby(["asset", "side"]):
        s = grp["slippage"].dropna()
        if s.empty:
            continue
        rows.append({
            "asset":  asset,
            "side":   side,
            "n":      len(s),
            "mean":   round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "worst":  round(float(s.max()), 4),
        })
    if rows:
        print(pd.DataFrame(rows).sort_values(["asset", "side"]).to_string(index=False))

    # ── per asset + sigma_entry ───────────────────────────────────────────────
    print("\n\n=== Slippage by Asset + Sigma Entry ===\n")
    rows = []
    for (asset, sig), grp in df.groupby(["asset", "sigma_entry"]):
        s = grp["slippage"].dropna()
        if s.empty:
            continue
        rows.append({
            "asset":       asset,
            "sigma_entry": sig,
            "n":           len(s),
            "mean":        round(float(s.mean()), 4),
            "median":      round(float(s.median()), 4),
            "worst":       round(float(s.max()), 4),
        })
    if rows:
        print(pd.DataFrame(rows).sort_values(["asset", "sigma_entry"]).to_string(index=False))


if __name__ == "__main__":
    main()
