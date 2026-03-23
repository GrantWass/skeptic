"""
Threshold optimizer for the 5-minute up/down strategy.

Given historical sessions, simulates the strategy across a grid of
(buy_threshold, sell_threshold) pairs and computes edge per session.

Strategy simulation:
  - "Fill" occurs if min(UP or DOWN price in minute 1) <= buy_threshold
  - "Sell triggered" if max(subsequent prices for that outcome) >= sell_threshold
  - If neither: position goes to resolution (1.0 win or 0.0 loss)
"""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from skeptic.research.fetcher import HistoricalSession

log = logging.getLogger(__name__)


@dataclass
class ThresholdResult:
    buy: float
    sell: float
    n_sessions: int
    n_fills: int
    n_sell_hits: int        # sell threshold reached after fill
    n_resolution_wins: int  # filled but sell not hit → resolved as winner
    n_resolution_losses: int
    fill_rate: float        # n_fills / n_sessions
    sell_hit_rate: float    # n_sell_hits / n_fills (conditional)
    edge_per_session: float # expected PnL per session


def simulate(
    sessions: list[HistoricalSession],
    buy: float,
    sell: float,
) -> ThresholdResult:
    n_fills = 0
    n_sell_hits = 0
    n_res_wins = 0
    n_res_losses = 0

    for s in sessions:
        # Check if either UP or DOWN touched the buy threshold in minute 1
        up_fill = s.up_min_m1 is not None and s.up_min_m1 <= buy
        down_fill = s.down_min_m1 is not None and s.down_min_m1 <= buy

        if not up_fill and not down_fill:
            continue

        n_fills += 1

        # Prefer whichever side hit the buy threshold first / lower
        if up_fill and down_fill:
            # Both hit — in practice strategy cancels one; model as "pick the one that pays"
            # Conservative: use the one with better outcome
            up_pays = (s.up_max_after_fill(buy) or 0) >= sell or (s.up_resolution or 0) >= 0.9
            down_pays = (s.down_max_after_fill(buy) or 0) >= sell or (s.down_resolution or 0) >= 0.9
            filled_outcome_wins = up_pays or down_pays
            # Sell hit if either outcome would have sold
            sell_hit = (
                (s.up_max_after_fill(buy) or 0) >= sell or
                (s.down_max_after_fill(buy) or 0) >= sell
            )
        elif up_fill:
            sell_hit = (s.up_max_after_fill(buy) or 0) >= sell
            filled_outcome_wins = sell_hit or (s.up_resolution or 0) >= 0.9
        else:
            sell_hit = (s.down_max_after_fill(buy) or 0) >= sell
            filled_outcome_wins = sell_hit or (s.down_resolution or 0) >= 0.9

        if sell_hit:
            n_sell_hits += 1
        elif filled_outcome_wins:
            n_res_wins += 1
        else:
            n_res_losses += 1

    n = len(sessions)
    if n == 0:
        return ThresholdResult(buy, sell, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0)

    fill_rate = n_fills / n
    sell_hit_rate = n_sell_hits / n_fills if n_fills > 0 else 0.0

    # Expected PnL per fill:
    #   sell hit → profit = sell - buy per share (normalized; 1 share deployed)
    #   resolution win → profit = 1.0 - buy
    #   resolution loss → loss = -buy
    pnl_if_sell = sell - buy
    pnl_if_res_win = 1.0 - buy
    pnl_if_res_loss = -buy

    expected_pnl_per_fill = (
        (n_sell_hits * pnl_if_sell +
         n_res_wins * pnl_if_res_win +
         n_res_losses * pnl_if_res_loss) / n_fills
        if n_fills > 0 else 0.0
    )
    edge_per_session = fill_rate * expected_pnl_per_fill

    return ThresholdResult(
        buy=buy,
        sell=sell,
        n_sessions=n,
        n_fills=n_fills,
        n_sell_hits=n_sell_hits,
        n_resolution_wins=n_res_wins,
        n_resolution_losses=n_res_losses,
        fill_rate=fill_rate,
        sell_hit_rate=sell_hit_rate,
        edge_per_session=edge_per_session,
    )


def optimize_thresholds(
    sessions: list[HistoricalSession],
    buy_range: tuple[float, float] = (0.20, 0.49),
    sell_range: tuple[float, float] = (0.51, 0.90),
    step: float = 0.01,
) -> pd.DataFrame:
    """
    Grid search over (buy, sell) threshold pairs.
    Returns a DataFrame sorted by edge_per_session descending.
    """
    buys = np.arange(buy_range[0], buy_range[1] + step, step).round(2)
    sells = np.arange(sell_range[0], sell_range[1] + step, step).round(2)

    rows = []
    for buy in buys:
        for sell in sells:
            r = simulate(sessions, float(buy), float(sell))
            rows.append({
                "buy": r.buy,
                "sell": r.sell,
                "n_sessions": r.n_sessions,
                "n_fills": r.n_fills,
                "fill_rate": round(r.fill_rate, 4),
                "n_sell_hits": r.n_sell_hits,
                "sell_hit_rate": round(r.sell_hit_rate, 4),
                "n_res_wins": r.n_resolution_wins,
                "n_res_losses": r.n_resolution_losses,
                "edge_per_session": round(r.edge_per_session, 6),
            })

    df = pd.DataFrame(rows)
    return df.sort_values("edge_per_session", ascending=False).reset_index(drop=True)


def best_params(df: pd.DataFrame) -> dict:
    """Return the best (buy, sell) pair from an optimize_thresholds result."""
    if df.empty:
        return {}
    row = df.iloc[0]
    return row.to_dict()


def rank_assets(
    asset_sessions: dict[str, list[HistoricalSession]],
    buy: float = 0.36,
    sell: float = 0.56,
) -> pd.DataFrame:
    """
    Compare assets at fixed (buy, sell) thresholds. Returns a DataFrame
    with one row per asset sorted by edge_per_session.
    """
    rows = []
    for asset, sessions in asset_sessions.items():
        if not sessions:
            continue
        r = simulate(sessions, buy, sell)
        rows.append({
            "asset": asset,
            "n_sessions": r.n_sessions,
            "fill_rate": round(r.fill_rate, 4),
            "sell_hit_rate": round(r.sell_hit_rate, 4),
            "edge_per_session": round(r.edge_per_session, 6),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("edge_per_session", ascending=False).reset_index(drop=True)
    return df
