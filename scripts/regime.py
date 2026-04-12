#!/usr/bin/env python3
"""
Current volatility regime detector.

Reads the last 6 hours of coin price data and computes:
  vol_ratio = std(prior 6h window moves) / sigma_value (from config/assets.yaml)

Buckets:
  low    vol_ratio < 0.75  — quiet, threshold crossings are rare but strong signals
  normal 0.75 – 1.25       — typical conditions
  high   vol_ratio > 1.25  — noisy, coin is crossing sigma on smaller-than-intended moves

Usage:
    python scripts/regime.py
    python scripts/regime.py --assets BTC ETH
    python scripts/regime.py --lookback 3   # hours (default: 6)
"""
import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml

WINDOW_SECS = 300
MIN_WINDOWS = 6

ASSET_TO_SYMBOL = {
    "BTC":  "BTCUSDT",
    "ETH":  "ETHUSDT",
    "SOL":  "SOLUSDT",
    "DOGE": "DOGEUSDT",
    "XRP":  "XRPUSDT",
    "BNB":  "BNBUSDT",
    "HYPE": "HYPEUSDT",
}

LOW_THRESHOLD  = 0.75
HIGH_THRESHOLD = 1.25


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def load_coin_close(coin_dir: str, asset: str) -> pd.Series | None:
    symbol = ASSET_TO_SYMBOL.get(asset.upper())
    if symbol is None:
        return None
    path = os.path.join(coin_dir, f"{symbol}_1s.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, usecols=["ts", "close"])
    df = df.drop_duplicates("ts").sort_values("ts")
    return pd.Series(df["close"].values.astype(float), index=df["ts"].values)


def compute_regime(
    coin_series: pd.Series,
    sigma: float,
    lookback_secs: int,
    now_ts: int,
) -> tuple[float | None, str, int]:
    """
    Returns (vol_ratio, regime_label, n_windows_used).
    Uses 5-min window moves from [now - lookback, now).
    """
    ts_idx = coin_series.index.values
    vals   = coin_series.values

    cutoff = now_ts - lookback_secs

    # Align to 5-min boundaries
    wstart = (cutoff // WINDOW_SECS) * WINDOW_SECS
    if cutoff % WINDOW_SECS != 0:
        wstart += WINDOW_SECS
    # Last complete window strictly before now
    wend = (now_ts // WINDOW_SECS) * WINDOW_SECS - WINDOW_SECS

    moves = []
    for wts in range(wstart, wend + 1, WINDOW_SECS):
        lo = int(np.searchsorted(ts_idx, wts,               side="left"))
        hi = int(np.searchsorted(ts_idx, wts + WINDOW_SECS, side="left"))
        if (hi - lo) < 280:
            continue
        moves.append(float(vals[hi - 1]) - float(vals[lo]))

    if len(moves) < MIN_WINDOWS:
        return None, "unknown", len(moves)

    rolling_sigma = float(np.std(moves))
    vol_ratio = rolling_sigma / sigma if sigma > 0 else None

    if vol_ratio is None:
        return None, "unknown", len(moves)
    elif vol_ratio < LOW_THRESHOLD:
        label = "low"
    elif vol_ratio > HIGH_THRESHOLD:
        label = "high"
    else:
        label = "normal"

    return vol_ratio, label, len(moves)


REGIME_COLOR = {
    "low":     "\033[36m",   # cyan
    "normal":  "\033[32m",   # green
    "high":    "\033[33m",   # yellow
    "unknown": "\033[90m",   # grey
}
RESET = "\033[0m"


def main() -> None:
    p = argparse.ArgumentParser(description="Current vol regime per asset")
    p.add_argument("--assets",    nargs="+", default=None,
                   help="Assets to check (default: all in config)")
    p.add_argument("--config",    default="config/assets.yaml")
    p.add_argument("--coin-dir",  default="data/coin_prices")
    p.add_argument("--lookback",  type=float, default=6.0,
                   help="Lookback window in hours (default: 6)")
    p.add_argument("--no-color",  action="store_true")
    args = p.parse_args()

    cfg          = load_config(args.config)
    lookback_sec = int(args.lookback * 3600)
    now_ts       = int(time.time())
    now_str      = datetime.fromtimestamp(now_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    assets = [a.upper() for a in args.assets] if args.assets else [
        a for a in ASSET_TO_SYMBOL if a in cfg
    ]

    print(f"\nVol Regime  —  {now_str}  (lookback {args.lookback:.0f}h)\n")
    print(f"  {'asset':<6}  {'regime':<8}  {'vol_ratio':>9}  {'vs sigma':>10}  {'n_windows':>9}  {'sigma_value':>12}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*9}  {'─'*10}  {'─'*9}  {'─'*12}")

    for asset in assets:
        asset_cfg = cfg.get(asset, {})
        sigma = asset_cfg.get("sigma_value")
        if sigma is None:
            print(f"  {asset:<6}  {'─':>8}  no sigma_value in config")
            continue

        coin = load_coin_close(args.coin_dir, asset)
        if coin is None or coin.empty:
            print(f"  {asset:<6}  {'─':>8}  no coin data")
            continue

        vol_ratio, label, n_windows = compute_regime(coin, sigma, lookback_sec, now_ts)

        if args.no_color:
            color, reset = "", ""
        else:
            color = REGIME_COLOR.get(label, "")
            reset = RESET

        ratio_str = f"{vol_ratio:.3f}" if vol_ratio is not None else "  n/a  "
        # Show realized sigma in price units
        realized_sigma_str = f"${vol_ratio * sigma:.4f}" if vol_ratio is not None else "  n/a  "

        print(
            f"  {asset:<6}  {color}{label:<8}{reset}  {ratio_str:>9}  "
            f"{realized_sigma_str:>10}  {n_windows:>9}  ${sigma:.6g}"
        )

    print()
    print(f"  Buckets:  low < {LOW_THRESHOLD}  |  normal {LOW_THRESHOLD}–{HIGH_THRESHOLD}  |  high > {HIGH_THRESHOLD}")
    print(f"  vol_ratio = std(prior {args.lookback:.0f}h 5-min moves) / sigma_value")
    print(f"  vs sigma  = realized sigma in price units (what a 1σ move actually is right now)\n")


if __name__ == "__main__":
    main()
