#!/usr/bin/env python3
"""
Train a classifier to predict high-buy win/loss from market price data.

Features are extracted from the price time series up to the trigger point —
NO live trade data is used. Labels come from session resolution.

Features per session (at a given threshold):
  elapsed_secs          when the trigger fired in the window
  trigger_price         exact price at trigger moment
  price_velocity_30s    linear slope of the triggered side in the first 30s
  max_price_pre_trigger max price of triggered side before trigger
  price_spread_at_60s   |up_price - down_price| at t=60s (price divergence)
  hour_of_day           UTC hour of window start
  day_of_week           0=Mon … 6=Sun
  triggered_up          1 if UP side triggered, 0 if DOWN

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --threshold 0.74 --assets BTC DOGE
    python scripts/train_model.py --all-thresholds
"""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timezone

from skeptic import config
from skeptic.research import fetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = os.path.join("data", "models")


# ── Feature extraction ────────────────────────────────────────────────────────

def _price_at(trades: list[tuple[int, float]], target_ts: int) -> float | None:
    """Last known price at or before target_ts."""
    candidates = [p for ts, p in trades if ts <= target_ts]
    return candidates[-1] if candidates else None

def _window_trades(trades: list[tuple[int, float]], start_ts: int, end_ts: int):
    return [(ts, p) for ts, p in trades if start_ts <= ts <= end_ts]


def _volatility(trades: list[tuple[int, float]]) -> float | None:
    """Std dev of returns (price differences)."""
    if len(trades) < 3:
        return None
    prices = np.array([p for _, p in trades], dtype=float)
    rets = np.diff(prices)
    if len(rets) == 0:
        return None
    return float(np.std(rets))


def _max_drawdown(prices: list[float]) -> float | None:
    """Max drawdown in a price series."""
    if len(prices) < 2:
        return None
    peak = prices[0]
    max_dd = 0.0
    for p in prices:
        if p > peak:
            peak = p
        dd = (peak - p)
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def _linear_slope(trades: list[tuple[int, float]], start_ts: int, window_secs: int) -> float | None:
    """Slope (price/sec) of a linear fit over the first `window_secs` seconds."""
    pts = [(ts - start_ts, p) for ts, p in trades
           if start_ts <= ts <= start_ts + window_secs]
    if len(pts) < 3:
        return None
    xs = np.array([x for x, _ in pts], dtype=float)
    ys = np.array([y for _, y in pts], dtype=float)
    # simple linear regression slope
    x_mean = xs.mean()
    denom = ((xs - x_mean) ** 2).sum()
    if denom == 0:
        return None
    return float(((xs - x_mean) * (ys - ys.mean())).sum() / denom)


def extract_features(
    session: "fetcher.HistoricalSession",
    threshold: float,
) -> dict | None:
    """
    Extract features for one session at `threshold`.
    Returns None if there is no trigger or resolution is unknown.
    """
    s = session

    # Find first trigger
    up_ts = next((ts for ts, p in s.up_trades_all   if p >= threshold), None)
    dn_ts = next((ts for ts, p in s.down_trades_all if p >= threshold), None)

    if up_ts is None and dn_ts is None:
        return None  # no fill this session

    if up_ts is not None and (dn_ts is None or up_ts <= dn_ts):
        triggered_up  = 1
        trigger_ts    = up_ts
        trigger_price = next(p for ts, p in s.up_trades_all if ts == up_ts)
        resolution    = s.up_resolution
        self_trades   = s.up_trades_all
        opp_trades    = s.down_trades_all
    else:
        assert dn_ts is not None
        triggered_up  = 0
        trigger_ts    = dn_ts
        trigger_price = next(p for ts, p in s.down_trades_all if ts == dn_ts)
        resolution    = s.down_resolution
        self_trades   = s.down_trades_all
        opp_trades    = s.up_trades_all

    if resolution is None:
        return None  # can't label

    label = 1 if resolution >= 0.9 else 0
    elapsed = trigger_ts - s.window_start_ts

    # Pre-trigger max price
    pre_prices = [p for ts, p in self_trades if ts < trigger_ts]
    max_pre    = max(pre_prices) if pre_prices else trigger_price

    # Price spread at t=60s
    up_at_60 = _price_at(s.up_trades_all,   s.window_start_ts + 60)
    dn_at_60 = _price_at(s.down_trades_all, s.window_start_ts + 60)
    up_at_trigger = _price_at(s.up_trades_all, trigger_ts)
    dn_at_trigger = _price_at(s.down_trades_all, trigger_ts)

    spread_at_trigger = (
        abs(up_at_trigger - dn_at_trigger)
        if (up_at_trigger is not None and dn_at_trigger is not None)
        else None
    )

    spread_60 = abs(up_at_60 - dn_at_60) if (up_at_60 and dn_at_60) else None

    # Velocity (slope) of triggered side in first 30s
    velocity_30s = _linear_slope(self_trades, s.window_start_ts, 30)

    # Pre-trigger velocities
    pre_trades_15 = _window_trades(self_trades, trigger_ts - 15, trigger_ts)
    pre_trades_30 = _window_trades(self_trades, trigger_ts - 30, trigger_ts)

    pre_vel_15 = _linear_slope(pre_trades_15, trigger_ts - 15, 15)
    pre_vel_30 = _linear_slope(pre_trades_30, trigger_ts - 30, 30)

    vol_30 = _volatility(pre_trades_30)

    pre_prices = [p for ts, p in self_trades if ts < trigger_ts]
    max_pre = max(pre_prices) if pre_prices else trigger_price

    max_dd_pre = _max_drawdown(pre_prices) if pre_prices else 0.0

    dt = datetime.fromtimestamp(s.window_start_ts, tz=timezone.utc)

    return {
        "elapsed_secs":          elapsed,
        "trigger_price":         trigger_price,

        # Momentum
        "price_velocity_30s":    velocity_30s,
        "pre_velocity_15s":      pre_vel_15,
        "pre_velocity_30s":      pre_vel_30,

        # Structure
        "max_price_pre_trigger": max_pre,
        "max_drawdown_pre":      max_dd_pre,

        # Market quality
        "volatility_30s":        vol_30,
        "spread_at_trigger":     spread_at_trigger,

        # Context
        "hour_of_day":           dt.hour,
        "day_of_week":           dt.weekday(),
        "triggered_up":          triggered_up,
        "threshold":             threshold,

        # Label
        "label":                 label,
        "asset":                 s.asset,
    }


def build_dataset(
    all_sessions: dict,
    thresholds: list[float],
) -> pd.DataFrame:
    rows = []
    for asset, sessions in all_sessions.items():
        for t in thresholds:
            for s in sessions:
                feat = extract_features(s, t)
                if feat is not None:
                    rows.append(feat)
    df = pd.DataFrame(rows)
    log.info("Dataset: %d samples across %d assets / %d thresholds",
             len(df), len(all_sessions), len(thresholds))
    return df


# ── Training ──────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "elapsed_secs",
    "trigger_price",

    # Momentum
    "price_velocity_30s",
    "pre_velocity_15s",
    "pre_velocity_30s",

    # Structure
    "max_price_pre_trigger",
    "max_drawdown_pre",

    # Market quality
    "volatility_30s",
    "spread_at_trigger",

    # Context
    "hour_of_day",
    "day_of_week",
    "triggered_up",
    "threshold",
]


def train(df: pd.DataFrame) -> None:
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import StratifiedKFold, cross_validate
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        import pickle
    except ImportError:
        log.error("scikit-learn not installed. Run: pip install scikit-learn")
        sys.exit(1)

    X = df[FEATURE_COLS].copy()
    y = df["label"].values

    if y.sum() == 0 or (1 - y).sum() == 0:
        log.error("Only one class in dataset — cannot train")
        return

    log.info("Class distribution: %d wins / %d losses (%.1f%% win rate)",
             int(y.sum()), int((1 - y).sum()), y.mean() * 100)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf",     GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(pipe, X, y, cv=cv,
                            scoring=["accuracy", "roc_auc"],
                            return_train_score=True)

    print("\n── Cross-Validation Results (5-fold) ──────────────────────────────")
    print(f"  Accuracy  : {scores['test_accuracy'].mean():.3f}  ±{scores['test_accuracy'].std():.3f}")
    print(f"  ROC-AUC   : {scores['test_roc_auc'].mean():.3f}  ±{scores['test_roc_auc'].std():.3f}")
    print(f"  Train Acc : {scores['train_accuracy'].mean():.3f}  (overfit gap: "
          f"{scores['train_accuracy'].mean() - scores['test_accuracy'].mean():.3f})")

    # Fit on full data for feature importances + saving
    pipe.fit(X, y)
    clf = pipe.named_steps["clf"]

    print("\n── Feature Importances ────────────────────────────────────────────")
    importances = sorted(zip(FEATURE_COLS, clf.feature_importances_),
                         key=lambda x: x[1], reverse=True)
    for feat, imp in importances:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<28}  {imp:.4f}  {bar}")

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "high_buy_model.pkl")
    with open(model_path, "wb") as fh:
        pickle.dump(pipe, fh)
    log.info("Model saved → %s", model_path)

    # Save feature importance CSV
    imp_path = os.path.join(MODELS_DIR, "feature_importances.csv")
    pd.DataFrame(importances, columns=["feature", "importance"]).to_csv(imp_path, index=False)
    log.info("Feature importances saved → %s", imp_path)

    print(f"\n  Model saved: {model_path}")
    print(f"  Importances: {imp_path}")

    # Timing insight
    timing_feat = next(
        (imp for feat, imp in importances if feat == "elapsed_secs"), None
    )
    if timing_feat is not None:
        rank = [feat for feat, _ in importances].index("elapsed_secs") + 1
        print(f"\n  elapsed_secs importance: {timing_feat:.4f} (rank #{rank}/{len(importances)})")
        if rank <= 3:
            print("  → Trigger timing is a TOP predictor of win/loss.")
        elif rank <= 6:
            print("  → Trigger timing has moderate predictive power.")
        else:
            print("  → Trigger timing has low predictive power; other features dominate.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train high-buy win/loss classifier from market data")
    p.add_argument("--assets", nargs="+", default=config.ASSETS,
                   help="Assets to include")
    p.add_argument("--threshold", type=float, default=None,
                   help="Single threshold to use (e.g. 0.74)")
    p.add_argument("--all-thresholds", action="store_true",
                   help="Sweep thresholds 0.65–0.90 in 0.05 steps")
    p.add_argument("--prices-dir", default="data/prices",
                   help="Directory of prices_*.csv files")
    p.add_argument("--min-points", type=int, default=280,
                   help="Minimum data points per window (default: 280)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.all_thresholds:
        thresholds = [round(t / 100, 2) for t in range(65, 93, 5)]
    elif args.threshold is not None:
        thresholds = [args.threshold]
    else:
        thresholds = [0.70, 0.74, 0.78, 0.80, 0.85]

    log.info("Loading sessions from %s for assets %s…", args.prices_dir, args.assets)
    all_sessions = fetcher.load_from_price_files(
        args.assets,
        prices_dir=args.prices_dir,
        min_points=args.min_points,
    )
    total = sum(len(v) for v in all_sessions.values())
    if total == 0:
        log.error("No sessions loaded. Run collect_prices.py first.")
        sys.exit(1)
    log.info("Loaded %d sessions total", total)

    df = build_dataset(all_sessions, thresholds)
    if df.empty:
        log.error("No triggered samples found at thresholds %s", thresholds)
        sys.exit(1)

    train(df)


if __name__ == "__main__":
    main()
