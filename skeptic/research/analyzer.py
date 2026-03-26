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


def _fill_trades(s: HistoricalSession, fill_window: int) -> tuple[list, list]:
    """Return (up_m1, dn_m1) trades filtered to the fill window."""
    cutoff = s.window_start_ts + fill_window
    return (
        [(ts, p) for ts, p in s.up_trades_all if ts <= cutoff],
        [(ts, p) for ts, p in s.down_trades_all if ts <= cutoff],
    )


def _max_after(trades_all: list, fill_ts: int | None) -> float | None:
    if fill_ts is None:
        return None
    vals = [p for ts, p in trades_all if ts > fill_ts]
    return max(vals) if vals else None


def simulate(
    sessions: list[HistoricalSession],
    buy: float,
    sell: float,
    fill_window: int = 60,
) -> ThresholdResult:
    n_fills = 0
    n_sell_hits = 0
    n_res_wins = 0
    n_res_losses = 0

    for s in sessions:
        up_m1, dn_m1 = _fill_trades(s, fill_window)

        up_min  = min((p for _, p in up_m1), default=None)
        dn_min  = min((p for _, p in dn_m1), default=None)
        up_fill = up_min is not None and up_min <= buy
        down_fill = dn_min is not None and dn_min <= buy

        if not up_fill and not down_fill:
            continue

        n_fills += 1

        if up_fill and down_fill:
            # Both touched the threshold — only the first fill is kept; the other
            # is immediately cancelled. Determine which side triggered first.
            # UP+DOWN ≈ 1.0 in a binary market, so true ties are impossible, but we
            # default to UP as a tie-breaker for the rare edge case.
            up_ts   = next((ts for ts, p in up_m1 if p <= buy), None)
            down_ts = next((ts for ts, p in dn_m1 if p <= buy), None)
            if up_ts is not None and (down_ts is None or up_ts <= down_ts):
                sell_hit = (_max_after(s.up_trades_all, up_ts) or 0) >= sell
                filled_outcome_wins = sell_hit or (s.up_resolution or 0) >= 0.9
            else:
                sell_hit = (_max_after(s.down_trades_all, down_ts) or 0) >= sell
                filled_outcome_wins = sell_hit or (s.down_resolution or 0) >= 0.9
        elif up_fill:
            up_ts   = next((ts for ts, p in up_m1 if p <= buy), None)
            sell_hit = (_max_after(s.up_trades_all, up_ts) or 0) >= sell
            filled_outcome_wins = sell_hit or (s.up_resolution or 0) >= 0.9
        else:
            dn_ts   = next((ts for ts, p in dn_m1 if p <= buy), None)
            sell_hit = (_max_after(s.down_trades_all, dn_ts) or 0) >= sell
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


def simulate_side(
    sessions: list[HistoricalSession],
    buy: float,
    sell: float,
    fill_window: int = 60,
    side: str = "both",  # "up" | "down" | "both"
) -> ThresholdResult:
    """Like simulate() but restricts which side can fill."""
    n_fills = 0
    n_sell_hits = 0
    n_res_wins = 0
    n_res_losses = 0

    for s in sessions:
        up_m1, dn_m1 = _fill_trades(s, fill_window)

        up_min  = min((p for _, p in up_m1), default=None) if side in ("up", "both")   else None
        dn_min  = min((p for _, p in dn_m1), default=None) if side in ("down", "both") else None
        up_fill   = up_min is not None and up_min <= buy
        down_fill = dn_min is not None and dn_min <= buy

        if not up_fill and not down_fill:
            continue

        n_fills += 1

        if up_fill and down_fill:
            up_ts   = next((ts for ts, p in up_m1 if p <= buy), None)
            down_ts = next((ts for ts, p in dn_m1 if p <= buy), None)
            if up_ts is not None and (down_ts is None or up_ts <= down_ts):
                sell_hit = (_max_after(s.up_trades_all, up_ts) or 0) >= sell
                filled_outcome_wins = sell_hit or (s.up_resolution or 0) >= 0.9
            else:
                sell_hit = (_max_after(s.down_trades_all, down_ts) or 0) >= sell
                filled_outcome_wins = sell_hit or (s.down_resolution or 0) >= 0.9
        elif up_fill:
            up_ts = next((ts for ts, p in up_m1 if p <= buy), None)
            sell_hit = (_max_after(s.up_trades_all, up_ts) or 0) >= sell
            filled_outcome_wins = sell_hit or (s.up_resolution or 0) >= 0.9
        else:
            dn_ts = next((ts for ts, p in dn_m1 if p <= buy), None)
            sell_hit = (_max_after(s.down_trades_all, dn_ts) or 0) >= sell
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
    expected_pnl_per_fill = (
        (n_sell_hits * (sell - buy) + n_res_wins * (1.0 - buy) + n_res_losses * (-buy)) / n_fills
        if n_fills > 0 else 0.0
    )
    return ThresholdResult(
        buy=buy, sell=sell, n_sessions=n, n_fills=n_fills,
        n_sell_hits=n_sell_hits, n_resolution_wins=n_res_wins,
        n_resolution_losses=n_res_losses, fill_rate=fill_rate,
        sell_hit_rate=sell_hit_rate,
        edge_per_session=fill_rate * expected_pnl_per_fill,
    )


def optimize_thresholds_sided(
    sessions: list[HistoricalSession],
    buy_range: tuple[float, float] = (0.10, 0.49),
    sell_range: tuple[float, float] = (0.45, 0.96),
    step: float = 0.03,
    fill_window: int = 60,
    side: str = "both",
) -> pd.DataFrame:
    """Grid search using simulate_side for a specific betting side."""
    buys  = np.arange(buy_range[0],  buy_range[1]  + step, step).round(2)
    sells = np.arange(sell_range[0], sell_range[1] + step, step).round(2)
    rows = []
    for buy in buys:
        for sell in sells:
            r = simulate_side(sessions, float(buy), float(sell), fill_window=fill_window, side=side)
            rows.append({
                "buy": r.buy, "sell": r.sell, "side": side,
                "n_sessions": r.n_sessions, "n_fills": r.n_fills,
                "fill_rate": round(r.fill_rate, 4),
                "sell_hit_rate": round(r.sell_hit_rate, 4),
                "edge_per_session": round(r.edge_per_session, 6),
            })
    df = pd.DataFrame(rows)
    return df.sort_values("edge_per_session", ascending=False).reset_index(drop=True)


def optimize_thresholds_sided_3d(
    sessions: list[HistoricalSession],
    buy_range: tuple[float, float] = (0.10, 0.49),
    sell_range: tuple[float, float] = (0.45, 0.96),
    step: float = 0.03,
    fill_window_range: tuple[int, int] = (10, 60),
    fill_window_step: int = 10,
    side: str = "both",
) -> pd.DataFrame:
    """3D grid search (buy × sell × fill_window) for a specific betting side."""
    fill_windows = list(range(fill_window_range[0], fill_window_range[1] + fill_window_step, fill_window_step))
    slices = []
    for fw in fill_windows:
        df = optimize_thresholds_sided(sessions, buy_range, sell_range, step, fw, side=side)
        df.insert(2, "fill_window", fw)
        slices.append(df)
    combined = pd.concat(slices, ignore_index=True)
    return combined.sort_values("edge_per_session", ascending=False).reset_index(drop=True)


def group_by_prev_resolution(
    sessions: list[HistoricalSession],
    window_secs: int = 300,
) -> tuple[list[HistoricalSession], list[HistoricalSession], int]:
    """
    Split sessions by the resolution of the immediately preceding window.

    Only includes sessions where:
      - the previous window (window_start_ts - 300s) exists in the dataset
      - that window's resolution is unambiguous (≥ 0.9 on one side)

    Returns (prev_up, prev_down, n_discarded).
    """
    by_ts = {s.window_start_ts: s for s in sessions}
    prev_up: list[HistoricalSession] = []
    prev_down: list[HistoricalSession] = []
    n_discarded = 0

    for s in sessions:
        prev = by_ts.get(s.window_start_ts - window_secs)
        if prev is None:
            n_discarded += 1
            continue
        if (prev.up_resolution or 0) >= 0.9:
            prev_up.append(s)
        elif (prev.down_resolution or 0) >= 0.9:
            prev_down.append(s)
        else:
            n_discarded += 1  # ambiguous resolution

    return prev_up, prev_down, n_discarded


def optimize_thresholds(
    sessions: list[HistoricalSession],
    buy_range: tuple[float, float] = (0.20, 0.49),
    sell_range: tuple[float, float] = (0.51, 0.90),
    step: float = 0.03,
    fill_window: int = 60,
) -> pd.DataFrame:
    """
    Grid search over (buy, sell) threshold pairs at a fixed fill window.
    Returns a DataFrame sorted by edge_per_session descending.
    """
    buys = np.arange(buy_range[0], buy_range[1] + step, step).round(2)
    sells = np.arange(sell_range[0], sell_range[1] + step, step).round(2)

    rows = []
    for buy in buys:
        for sell in sells:
            r = simulate(sessions, float(buy), float(sell), fill_window=fill_window)
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


def optimize_thresholds_3d(
    sessions: list[HistoricalSession],
    buy_range: tuple[float, float] = (0.10, 0.50),
    sell_range: tuple[float, float] = (0.45, 0.96),
    step: float = 0.03,
    fill_window_range: tuple[int, int] = (10, 90),
    fill_window_step: int = 10,
) -> pd.DataFrame:
    """
    3D grid search over buy × sell × fill_window.
    Returns a DataFrame with a fill_window column, sorted by edge_per_session descending.
    """
    fill_windows = list(range(fill_window_range[0], fill_window_range[1] + fill_window_step, fill_window_step))
    slices = []
    for fw in fill_windows:
        df = optimize_thresholds(sessions, buy_range, sell_range, step, fill_window=fw)
        df.insert(2, "fill_window", fw)
        slices.append(df)
    combined = pd.concat(slices, ignore_index=True)
    return combined.sort_values("edge_per_session", ascending=False).reset_index(drop=True)


def best_params(df: pd.DataFrame) -> dict:
    """Return the best row from an optimize_thresholds or optimize_thresholds_3d result."""
    if df.empty:
        return {}
    row = df.iloc[0]
    return row.to_dict()


def best_params_min_fill_rate(df: pd.DataFrame, min_fill_rate: float) -> dict:
    """Return the best params row where fill_rate >= min_fill_rate."""
    if df.empty:
        return {}
    filtered = df[df["fill_rate"] >= min_fill_rate - 1e-9]
    if filtered.empty:
        return {}
    return filtered.iloc[0].to_dict()


def best_neighborhood_params_min_fill_rate(
    df: pd.DataFrame,
    min_fill_rate: float,
    buy_radius: float = 0.03,
    sell_radius: float = 0.03,
    fw_radius: int = 10,
) -> dict:
    """
    Find the best neighborhood center where the center itself has fill_rate >= min_fill_rate,
    but neighborhood means are computed using the full grid (neighbors need not meet the constraint).
    """
    if df.empty:
        return {}
    has_fw = "fill_window" in df.columns
    candidates = df[df["fill_rate"] >= min_fill_rate - 1e-9]
    if candidates.empty:
        return {}

    def _nb_mean(row) -> float:
        mask = (
            ((df["buy"]  - row["buy"] ).abs() <= buy_radius  + 1e-9) &
            ((df["sell"] - row["sell"]).abs() <= sell_radius + 1e-9)
        )
        if has_fw:
            mask &= (df["fill_window"] - row["fill_window"]).abs() <= fw_radius + 1e-9
        return float(df["edge_per_session"][mask].mean())

    nb_means  = candidates.apply(_nb_mean, axis=1)
    best_idx  = int(nb_means.idxmax())  # type: ignore[arg-type]
    best_mean = float(nb_means[best_idx])

    best_row  = df.iloc[best_idx]
    peak_row  = df.iloc[0]
    nb_buy    = float(best_row["buy"])   # type: ignore[arg-type]
    nb_sell   = float(best_row["sell"])  # type: ignore[arg-type]
    nb_fw     = int(best_row["fill_window"]) if has_fw else None  # type: ignore[arg-type]
    peak_buy  = float(peak_row["buy"])   # type: ignore[arg-type]
    peak_sell = float(peak_row["sell"])  # type: ignore[arg-type]
    peak_fw   = int(peak_row["fill_window"]) if has_fw else None  # type: ignore[arg-type]

    buy_diff  = abs(nb_buy  - peak_buy)
    sell_diff = abs(nb_sell - peak_sell)
    fw_diff   = abs(nb_fw   - peak_fw) if (peak_fw is not None and nb_fw is not None) else 0
    if buy_diff <= buy_radius and sell_diff <= sell_radius and fw_diff <= fw_radius:
        agreement = "agree"
    elif buy_diff <= buy_radius * 2 and sell_diff <= sell_radius * 2:
        agreement = "nearby"
    else:
        agreement = "diverge"

    nb_mask = (
        ((df["buy"]  - nb_buy ).abs() <= buy_radius  + 1e-9) &
        ((df["sell"] - nb_sell).abs() <= sell_radius + 1e-9)
    )
    if has_fw and nb_fw is not None:
        nb_mask &= (df["fill_window"] - nb_fw).abs() <= fw_radius + 1e-9
    exact = (df["buy"] == nb_buy) & (df["sell"] == nb_sell)
    if has_fw and nb_fw is not None:
        exact &= df["fill_window"] == nb_fw
    neighbors = df[nb_mask & ~exact]
    pct_pos = float((neighbors["edge_per_session"] > 0).mean()) if not neighbors.empty else 0.0

    return {
        "buy":                    nb_buy,
        "sell":                   nb_sell,
        "fill_window":            nb_fw,
        "neighborhood_mean_edge": round(best_mean, 6),
        "pct_positive":           round(pct_pos, 3),
        "peak_buy":               peak_buy,
        "peak_sell":              peak_sell,
        "peak_fill_window":       peak_fw,
        "peak_edge":              round(float(peak_row["edge_per_session"]), 6),  # type: ignore[arg-type]
        "peak_vs_neighborhood":   agreement,
    }


def neighborhood_robustness(
    df: pd.DataFrame,
    best: dict,
    buy_radius: float = 0.03,
    sell_radius: float = 0.03,
    fw_radius: int = 10,
) -> dict:
    """
    Measures how robust the best parameter set is by evaluating its immediate
    neighborhood in (buy, sell[, fill_window]) space.

    A flat plateau around the best point → low overfitting risk.
    A lone spike → likely overfit to the data.

    Returns:
        n_neighbors       — number of grid points within the radius
        neighbor_mean_edge — mean edge_per_session of those neighbors
        robustness_ratio  — neighbor_mean / best_edge  (1.0 = perfect plateau)
        pct_positive      — fraction of neighbors with edge > 0
        shape             — "plateau" / "moderate" / "spike" label
    """
    if df.empty or not best:
        return {}

    best_buy  = best["buy"]
    best_sell = best["sell"]
    best_edge = best.get("edge_per_session", 0)
    best_fw   = best.get("fill_window")

    mask = (
        ((df["buy"]  - best_buy ).abs() <= buy_radius  + 1e-9) &
        ((df["sell"] - best_sell).abs() <= sell_radius + 1e-9)
    )
    if "fill_window" in df.columns and best_fw is not None:
        mask &= (df["fill_window"] - best_fw).abs() <= fw_radius + 1e-9

    # exclude the best point itself
    exact = (df["buy"] == best_buy) & (df["sell"] == best_sell)
    if "fill_window" in df.columns and best_fw is not None:
        exact &= df["fill_window"] == best_fw

    neighbors = df[mask & ~exact]

    if neighbors.empty:
        return {"n_neighbors": 0, "neighbor_mean_edge": None,
                "robustness_ratio": None, "pct_positive": None, "shape": "unknown"}

    mean_edge = neighbors["edge_per_session"].mean()
    pct_pos   = float((neighbors["edge_per_session"] > 0).mean())
    ratio     = (mean_edge / best_edge) if best_edge and best_edge != 0 else None

    if ratio is None:
        shape = "unknown"
    elif ratio >= 0.70:
        shape = "plateau"
    elif ratio >= 0.30:
        shape = "moderate"
    else:
        shape = "spike"

    return {
        "n_neighbors":        len(neighbors),
        "neighbor_mean_edge": round(mean_edge, 6),
        "robustness_ratio":   round(ratio, 3) if ratio is not None else None,
        "pct_positive":       round(pct_pos, 3),
        "shape":              shape,
    }


def best_neighborhood_params(
    df: pd.DataFrame,
    buy_radius: float = 0.03,
    sell_radius: float = 0.03,
    fw_radius: int = 10,
) -> dict:
    """
    Scan every point in the grid and find the one whose *neighborhood average*
    edge is highest — not just the single peak point.

    This is the most robust parameter region: even if the exact center isn't the
    global peak, the surrounding area consistently performs well.

    When this differs significantly from best_params(), the peak is likely a
    spike (overfit). When they agree, the peak sits on a genuine plateau.

    Returns the same fields as neighborhood_robustness(), plus:
        buy, sell, fill_window  — center of the best neighborhood
        peak_buy, peak_sell     — best individual point for comparison
        peak_vs_neighborhood    — "agree" / "nearby" / "diverge"
    """
    if df.empty:
        return {}

    has_fw = "fill_window" in df.columns

    def _neighborhood_mean(row) -> float:
        mask = (
            ((df["buy"]  - row["buy"] ).abs() <= buy_radius  + 1e-9) &
            ((df["sell"] - row["sell"]).abs() <= sell_radius + 1e-9)
        )
        if has_fw:
            mask &= (df["fill_window"] - row["fill_window"]).abs() <= fw_radius + 1e-9
        return df.loc[mask, "edge_per_session"].mean()

    nb_means = df.apply(_neighborhood_mean, axis=1)
    best_nb_idx  = nb_means.idxmax()
    best_nb_row  = df.loc[best_nb_idx]
    best_nb_mean = nb_means[best_nb_idx]

    # Peak (best individual point) for comparison
    peak_row  = df.iloc[0]  # df is pre-sorted descending by edge_per_session
    peak_buy  = float(peak_row["buy"])
    peak_sell = float(peak_row["sell"])
    peak_fw   = int(peak_row["fill_window"]) if has_fw else None

    nb_buy  = float(best_nb_row["buy"])
    nb_sell = float(best_nb_row["sell"])
    nb_fw   = int(best_nb_row["fill_window"]) if has_fw else None

    buy_diff  = abs(nb_buy  - peak_buy)
    sell_diff = abs(nb_sell - peak_sell)
    fw_diff   = abs(nb_fw   - peak_fw) if (peak_fw is not None and nb_fw is not None) else 0

    if buy_diff <= buy_radius and sell_diff <= sell_radius and fw_diff <= fw_radius:
        agreement = "agree"
    elif buy_diff <= buy_radius * 2 and sell_diff <= sell_radius * 2:
        agreement = "nearby"
    else:
        agreement = "diverge"

    # Neighborhood stats at the best-neighborhood center
    mask = (
        ((df["buy"]  - nb_buy ).abs() <= buy_radius  + 1e-9) &
        ((df["sell"] - nb_sell).abs() <= sell_radius + 1e-9)
    )
    if has_fw and nb_fw is not None:
        mask &= (df["fill_window"] - nb_fw).abs() <= fw_radius + 1e-9
    exact = (df["buy"] == nb_buy) & (df["sell"] == nb_sell)
    if has_fw and nb_fw is not None:
        exact &= df["fill_window"] == nb_fw
    neighbors = df[mask & ~exact]
    pct_pos = float((neighbors["edge_per_session"] > 0).mean()) if not neighbors.empty else 0.0

    return {
        "buy":                  nb_buy,
        "sell":                 nb_sell,
        "fill_window":          nb_fw,
        "neighborhood_mean_edge": round(float(best_nb_mean), 6),
        "n_neighbors":          int(mask.sum()) - 1,
        "pct_positive":         round(pct_pos, 3),
        "peak_buy":             peak_buy,
        "peak_sell":            peak_sell,
        "peak_fill_window":     peak_fw,
        "peak_edge":            round(float(peak_row["edge_per_session"]), 6),
        "peak_vs_neighborhood": agreement,
    }


def analyze_high_buy(
    sessions: list[HistoricalSession],
    threshold: float = 0.90,
) -> dict:
    """
    For sessions where UP or DOWN price ever traded at or above `threshold`,
    compute fill rate, win rate, and edge.

    Payoff model: buy at `threshold`, win (1 - threshold) if that side resolves,
    lose `threshold` otherwise.
    """
    win_payout = round(1.0 - threshold, 6)
    lose_payout = -threshold

    up_fills = up_wins = up_losses = 0
    dn_fills = dn_wins = dn_losses = 0

    for s in sessions:
        if not s.up_trades_all:
            continue

        up_touched = any(p >= threshold for _, p in s.up_trades_all)
        dn_touched = any(p >= threshold for _, p in s.down_trades_all)

        if up_touched and s.up_resolution is not None:
            up_fills += 1
            if s.up_resolution >= 0.9:
                up_wins += 1
            else:
                up_losses += 1

        if dn_touched and s.down_resolution is not None:
            dn_fills += 1
            if s.down_resolution >= 0.9:
                dn_wins += 1
            else:
                dn_losses += 1

    n = len(sessions)

    def _stats(fills: int, wins: int, losses: int) -> dict:
        if fills == 0:
            return {"fill_rate": 0.0, "win_rate": None, "edge_per_fill": None, "edge_per_session": 0.0}
        win_rate = wins / fills
        edge_per_fill = win_rate * win_payout + (1 - win_rate) * lose_payout
        return {
            "fill_rate": round(fills / n, 4),
            "win_rate": round(win_rate, 4),
            "edge_per_fill": round(edge_per_fill, 6),
            "edge_per_session": round(edge_per_fill * fills / n, 6),
            "n_fills": fills,
            "n_wins": wins,
            "n_losses": losses,
        }

    return {
        "n_sessions": n,
        "threshold": threshold,
        "up": _stats(up_fills, up_wins, up_losses),
        "down": _stats(dn_fills, dn_wins, dn_losses),
    }


def rank_assets(
    asset_sessions: dict[str, list[HistoricalSession]],
    buy: float,
    sell: float,
    fill_window: int = 60,
) -> pd.DataFrame:
    """
    Compare assets at fixed (buy, sell) thresholds. Returns a DataFrame
    with one row per asset sorted by edge_per_session.
    """
    rows = []
    for asset, sessions in asset_sessions.items():
        if not sessions:
            continue
        r = simulate(sessions, buy, sell, fill_window=fill_window)
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
