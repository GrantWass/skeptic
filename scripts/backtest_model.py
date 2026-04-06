#!/usr/bin/env python3
"""
Backtest the P(resolved_up) model strategy.

Strategy: at the FIRST second in each window where model edge >= threshold,
buy that side. One trade per window per asset.

Edge = predicted_prob - pm_price_up_equiv (market's implied P(UP)).
Positive edge → buy UP. Negative edge (if --allow-down) → buy DOWN.

Uses model_predictions.csv (test set only — no training data leakage).

Usage:
    python scripts/backtest_model.py
    python scripts/backtest_model.py --threshold 0.10
    python scripts/backtest_model.py --sweep
    python scripts/backtest_model.py --assets BTC ETH --threshold 0.15
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV = "data/reports/model_predictions.csv"


# ── Core simulation ────────────────────────────────────────────────────────────

def simulate(df: pd.DataFrame, threshold: float, allow_down: bool = False) -> pd.DataFrame:
    """
    For each (asset, window_ts): find the first second where |edge| >= threshold
    and edge is in the allowed direction. Execute one trade at that second.

    Returns a DataFrame of trades with columns:
      asset, window_ts, entry_second, predicted_prob, pm_price_up_equiv,
      edge, side, resolved_up, fill_price, pnl, won
    """
    trades = []

    for (asset, window_ts), wdf in df.groupby(["asset", "window_ts"]):
        wdf = wdf.sort_values("elapsed_second")

        # candidates: UP edge
        up_cands = wdf[wdf["edge"] >= threshold]
        # candidates: DOWN edge (buy DOWN = bet against UP)
        dn_cands = wdf[wdf["edge"] <= -threshold] if allow_down else pd.DataFrame()

        # pick whichever side fires first
        first_up = up_cands.iloc[0] if not up_cands.empty else None
        first_dn = dn_cands.iloc[0] if not dn_cands.empty else None

        if first_up is None and first_dn is None:
            continue

        if first_up is not None and first_dn is not None:
            row  = first_up if first_up["elapsed_second"] <= first_dn["elapsed_second"] else first_dn
        elif first_up is not None:
            row  = first_up
        else:
            row  = first_dn

        side = "UP" if row["edge"] > 0 else "DOWN"

        # fill price = pm_price_up_equiv for UP, (1 - pm_price_up_equiv) for DOWN
        if side == "UP":
            fill_price = float(row["pm_price_up_equiv"])
            won        = bool(row["resolved_up"])
        else:
            fill_price = 1.0 - float(row["pm_price_up_equiv"])
            won        = not bool(row["resolved_up"])

        pnl = (1.0 - fill_price) if won else -fill_price

        trades.append({
            "asset":             asset,
            "window_ts":         window_ts,
            "entry_second":      int(row["elapsed_second"]),
            "predicted_prob":    round(float(row["predicted_prob"]), 4),
            "pm_price_up_equiv": round(float(row["pm_price_up_equiv"]), 4),
            "edge":              round(float(row["edge"]), 4),
            "side":              side,
            "resolved_up":       int(row["resolved_up"]),
            "fill_price":        round(fill_price, 4),
            "pnl":               round(pnl, 4),
            "won":               won,
        })

    return pd.DataFrame(trades)


# ── Summary stats ─────────────────────────────────────────────────────────────

def summarise(trades: pd.DataFrame, threshold: float) -> dict:
    if trades.empty:
        return {"threshold": threshold, "n_trades": 0}

    return {
        "threshold":       threshold,
        "n_trades":        len(trades),
        "win_rate":        round(trades["won"].mean(), 4),
        "avg_edge":        round(trades["edge"].mean(), 4),
        "avg_pnl":         round(trades["pnl"].mean(), 4),
        "total_pnl":       round(trades["pnl"].sum(), 4),
        "avg_fill_price":  round(trades["fill_price"].mean(), 4),
        "avg_entry_sec":   round(trades["entry_second"].mean(), 1),
    }


def print_summary(trades: pd.DataFrame, threshold: float) -> None:
    s = summarise(trades, threshold)
    print(f"\n{'='*55}")
    print(f"  Threshold: {threshold:.2f}   Trades: {s['n_trades']}")
    print(f"{'='*55}")
    if s["n_trades"] == 0:
        print("  No trades at this threshold.")
        return
    print(f"  Win rate:        {s['win_rate']*100:.1f}%")
    print(f"  Avg edge:        {s['avg_edge']:+.4f}")
    print(f"  Avg PnL/trade:   {s['avg_pnl']:+.4f}")
    print(f"  Total PnL:       {s['total_pnl']:+.4f}")
    print(f"  Avg fill price:  {s['avg_fill_price']:.4f}")
    print(f"  Avg entry sec:   {s['avg_entry_sec']:.0f}s")

    # per-asset breakdown
    print(f"\n  Per-asset breakdown:")
    print(f"  {'asset':<6} {'n':>5} {'win%':>6} {'avg_pnl':>9} {'total_pnl':>10}")
    print(f"  {'-'*40}")
    for asset, adf in trades.groupby("asset"):
        print(
            f"  {asset:<6} {len(adf):>5} {adf['won'].mean()*100:>5.1f}%"
            f" {adf['pnl'].mean():>+9.4f} {adf['pnl'].sum():>+10.4f}"
        )


def print_sweep(df: pd.DataFrame, allow_down: bool) -> None:
    thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
    rows = []
    for t in thresholds:
        trades = simulate(df, t, allow_down)
        rows.append(summarise(trades, t))

    tbl = pd.DataFrame(rows)
    print("\nThreshold sweep:")
    print(tbl.to_string(index=False))

    # equity curve at best threshold (highest total_pnl with n_trades >= 50)
    valid = tbl[(tbl["n_trades"] >= 50) & (tbl["total_pnl"].notna())]
    if not valid.empty:
        best_t = float(valid.loc[valid["total_pnl"].idxmax(), "threshold"])
        print(f"\nBest threshold (n>=50): {best_t:.2f}")
        best_trades = simulate(df, best_t, allow_down).sort_values("window_ts")
        cum = best_trades["pnl"].cumsum()
        print(f"Equity curve — final PnL: {cum.iloc[-1]:+.4f}  "
              f"max drawdown: {(cum - cum.cummax()).min():+.4f}")


# ── Calibration check ─────────────────────────────────────────────────────────

def print_calibration(trades: pd.DataFrame) -> None:
    """
    Among executed trades, check whether predicted_prob matches actual win rate.
    Buckets of 0.1 width.
    """
    if trades.empty:
        return
    print("\nCalibration check (executed trades only):")
    print(f"  {'bucket':<14} {'avg_pred':>9} {'actual_win%':>12} {'gap':>8} {'n':>5}")
    print(f"  {'-'*52}")
    bins = [i / 10 for i in range(11)]
    trades = trades.copy()
    trades["bucket"] = pd.cut(trades["predicted_prob"], bins=bins, include_lowest=True)
    for bucket, grp in trades.groupby("bucket", observed=True):
        if grp.empty:
            continue
        avg_pred   = grp["predicted_prob"].mean()
        actual_win = grp["won"].mean()
        gap        = avg_pred - actual_win
        flag       = " ⚠" if abs(gap) > 0.05 else ""
        print(f"  {str(bucket):<14} {avg_pred:>9.3f} {actual_win*100:>11.1f}% {gap:>+7.3f}{flag}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest the P(resolved_up) model strategy")
    p.add_argument("--csv",        default=DEFAULT_CSV,
                   help="model_predictions.csv (default: %(default)s)")
    p.add_argument("--threshold",  type=float, default=0.10,
                   help="Minimum edge to trigger a trade (default: 0.10)")
    p.add_argument("--assets",     nargs="+", default=None,
                   help="Filter to specific assets (default: all)")
    p.add_argument("--allow-down", action="store_true", default=False,
                   help="Also trade when model edge favours DOWN (default: UP only)")
    p.add_argument("--sweep",      action="store_true", default=False,
                   help="Sweep thresholds 0.05–0.30 and print comparison table")
    p.add_argument("--out-csv",    default=None,
                   help="Save trade log to this CSV path (optional)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found. Run scripts/train_model.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["predicted_prob", "pm_price_up_equiv", "resolved_up"])
    df["resolved_up"] = df["resolved_up"].astype(int)

    if args.assets:
        assets = [a.upper() for a in args.assets]
        df = df[df["asset"].isin(assets)]
        if df.empty:
            print(f"ERROR: no data for assets {assets}", file=sys.stderr)
            sys.exit(1)

    print(f"Loaded {len(df):,} rows  |  "
          f"{df['asset'].nunique()} assets  |  "
          f"{df.groupby(['asset','window_ts']).ngroups:,} windows  "
          f"(test set only)")

    if args.sweep:
        print_sweep(df, args.allow_down)
    else:
        trades = simulate(df, args.threshold, args.allow_down)
        print_summary(trades, args.threshold)
        print_calibration(trades)

        if args.out_csv and not trades.empty:
            trades.to_csv(args.out_csv, index=False)
            print(f"\nTrade log → {args.out_csv}")


if __name__ == "__main__":
    main()
