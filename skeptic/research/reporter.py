"""
Generates research reports from threshold optimization results.

Outputs:
  - data/reports/optimal_params.csv   — best (buy, sell) per asset
  - data/reports/asset_ranking.csv    — assets ranked at fixed thresholds
  - data/reports/research_report.md   — human-readable summary
"""
import glob
import os
import logging
import random
from datetime import datetime

import pandas as pd

from skeptic import config
from skeptic.research.analyzer import ThresholdResult

log = logging.getLogger(__name__)


def _ensure_reports_dir() -> None:
    os.makedirs(config.REPORTS_DIR, exist_ok=True)


def save_optimal_params(
    per_asset_best: dict[str, dict],
    all_best: dict,
) -> str:
    """Save optimal params table to CSV and return path."""
    _ensure_reports_dir()
    rows = []
    for asset, params in per_asset_best.items():
        rows.append({"asset": asset, **params})
    df = pd.DataFrame(rows)
    path = os.path.join(config.REPORTS_DIR, "optimal_params.csv")
    df.to_csv(path, index=False)
    log.info("Saved optimal params to %s", path)
    return path


def save_asset_ranking(df: pd.DataFrame) -> str:
    _ensure_reports_dir()
    path = os.path.join(config.REPORTS_DIR, "asset_ranking.csv")
    df.to_csv(path, index=False)
    log.info("Saved asset ranking to %s", path)
    return path


def save_full_grid(asset: str, df: pd.DataFrame) -> str:
    _ensure_reports_dir()
    path = os.path.join(config.REPORTS_DIR, f"grid_{asset.lower()}.csv")
    df.head(200).to_csv(path, index=False)
    log.info("Saved grid for %s to %s", asset, path)
    return path


def write_report(
    per_asset_best: dict[str, dict],
    asset_ranking: pd.DataFrame,
    per_asset_robustness: dict[str, dict] | None = None,
    current_buy: float | None = config.BUY_PRICE,
    current_sell: float | None = config.SELL_PRICE,
    data_source: str = "api",
    capital: float = 500.0,
    position_size_pct: float = config.POSITION_SIZE_PCT,
    spread_cost: float = 0.002,
) -> str:
    """Write the markdown research report. Returns path."""
    _ensure_reports_dir()
    # Delete any previous research reports so the preview doesn't serve a cached copy
    for _old in glob.glob(os.path.join(config.REPORTS_DIR, "research_report_*.md")):
        os.remove(_old)
    suffix = random.randint(10000, 99999)
    path = os.path.join(config.REPORTS_DIR, f"research_report_{suffix}.md")
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    if data_source == "prices":
        methodology_lines = [
            "Per-second mid-prices were collected via `collect_prices.py` and stored as CSV files in `data/prices/`.",
            "Each 5-minute window's price series is used to simulate fills and exits.",
            "",
            "### Strategy simulation",
            "- **Fill** occurs if the mid-price for either UP or DOWN ≤ `buy_threshold` within the first 60 seconds",
            "- **Sell hit** occurs if the mid-price subsequently reaches ≥ `sell_threshold` before window end",
            "- Otherwise position goes to resolution (1.0 or 0.0)",
            "- **Edge per session** = fill_rate × expected_PnL_per_fill",
        ]
    elif data_source == "db":
        methodology_lines = [
            "Session data was loaded from `data/sessions.db` recorded during paper/live trading.",
            "Minute-1 price snapshots and fill data are used to simulate threshold outcomes.",
            "",
            "### Strategy simulation",
            "- **Fill** occurs if the minute-1 snapshot price ≤ `buy_threshold`",
            "- **Sell hit** occurs if the recorded sell price ≥ `sell_threshold`",
            "- Otherwise position goes to resolution (1.0 or 0.0)",
            "- **Edge per session** = fill_rate × expected_PnL_per_fill",
        ]
    else:
        methodology_lines = [
            "Historical 5-minute UP/DOWN markets were fetched from the Polymarket Gamma API.",
            "Trade data from the CLOB was used to reconstruct intra-session prices.",
            "",
            "> **Note:** Price reconstruction from trades is an approximation.",
            "> The bot only sees limit order fills, not the full order book mid-price.",
            "> Results should be treated as directional signals, not precise predictions.",
            "",
            "### Strategy simulation",
            "- **Fill** occurs if the minimum trade price for either UP or DOWN ≤ `buy_threshold` within minute 1",
            "- **Sell hit** occurs if the max subsequent trade price ≥ `sell_threshold`",
            "- Otherwise position goes to resolution (1.0 or 0.0)",
            "- **Edge per session** = fill_rate × expected_PnL_per_fill",
        ]

    lines = [
        "# Skeptic Research Report",
        f"_Generated: {now}_",
        "",
        "## Methodology",
        "",
        *methodology_lines,
        "",
        "---",
        "",
        f"## Asset Ranking at Current Thresholds "
        f"(buy={'unset' if current_buy is None else f'{current_buy:.2f}'}, "
        f"sell={'unset' if current_sell is None else f'{current_sell:.2f}'})",
        "",
    ]

    if not asset_ranking.empty:
        lines.append(asset_ranking.to_markdown(index=False))
    else:
        lines.append("_No data available._")

    _robustness = per_asset_robustness or {}

    lines += [
        "",
        "---",
        "",
        "## Optimal Thresholds per Asset",
        "",
        "| Asset | Optimal Buy | Optimal Sell | Fill Window | Fill Rate | Sell Hit Rate | Edge/Session | Neighbors | Neighbor Mean Edge | Robustness Ratio | % Positive | Shape |",
        "|-------|-------------|--------------|-------------|-----------|---------------|--------------|-----------|-------------------|-----------------|------------|-------|",
    ]

    SESSIONS_PER_DAY = 288  # 24h × 60min / 5min
    position_usdc = capital * position_size_pct

    for asset, params in per_asset_best.items():
        buy  = params.get("buy", "—")
        sell = params.get("sell", "—")
        fw   = f"{int(params['fill_window'])}s" if "fill_window" in params else "60s"
        fr   = params.get("fill_rate", "—")
        shr  = params.get("sell_hit_rate", "—")
        edge = params.get("edge_per_session", "—")
        rob     = _robustness.get(asset, {})
        n_nb    = rob.get("n_neighbors", "—")
        nb_mean = rob.get("neighbor_mean_edge", "—")
        ratio   = rob.get("robustness_ratio", "—")
        pct_pos = f"{rob['pct_positive']:.0%}" if rob.get("pct_positive") is not None else "—"
        shape   = rob.get("shape", "—")
        lines.append(
            f"| {asset} | {buy} | {sell} | {fw} | {fr} | {shr} | {edge} "
            f"| {n_nb} | {nb_mean} | {ratio} | {pct_pos} | {shape} |"
        )

    # --- Estimated profit section ---
    # Position sizing is fractional (always X% of current capital), so the bot
    # can never go negative — losses shrink the next bet rather than producing
    # a cash deficit.  We therefore express returns as % growth of capital and
    # compute the compounded capital after N sessions using:
    #   capital_after_n = capital_0 × (1 + pct_return_per_session)^n
    # where pct_return_per_session = (net_edge_per_share / buy_price) × position_size_pct
    import math as _math
    lines += [
        "",
        "---",
        "",
        "## Estimated Profit",
        "",
        f"_Assumptions: ${capital:,.0f} starting capital, {position_size_pct:.0%} fractional "
        f"position size, {SESSIONS_PER_DAY} windows/day._",
        "",
        f"_Fees: Polymarket charges 0% maker/taker. Spread cost = {spread_cost:.3f}/share/crossing._",
        "",
        f"> **Fractional sizing**: each bet is always {position_size_pct:.0%} of current capital — "
        "losses shrink subsequent bets so capital can never go negative.",
        "",
        "| Asset | Buy | Sell | $/Session (net) | $/Day | $/Week | $/Month |",
        "|-------|-----|------|-----------------|-------|--------|---------|",
    ]

    for asset, params in per_asset_best.items():
        buy = params.get("buy")
        sell = params.get("sell")
        edge = params.get("edge_per_session")
        fill_rate = params.get("fill_rate", 0)
        sell_hit_rate = params.get("sell_hit_rate", 0)
        if buy and sell and edge is not None and buy > 0:
            shares = position_usdc / buy
            gross_per_session = edge * shares
            spread_per_session = fill_rate * spread_cost * shares + fill_rate * sell_hit_rate * spread_cost * shares
            net_per_session = gross_per_session - spread_per_session
            lines.append(
                f"| {asset} | {buy} | {sell} "
                f"| ${net_per_session:+.4f} "
                f"| ${net_per_session * SESSIONS_PER_DAY:+.2f} "
                f"| ${net_per_session * SESSIONS_PER_DAY * 7:+.2f} "
                f"| ${net_per_session * SESSIONS_PER_DAY * 30:+.2f} |"
            )

    lines += [
        "",
        "---",
        "",
        "## Recommendation",
        "",
    ]

    if per_asset_best:
        # Best overall: highest edge across all assets
        best_asset = max(per_asset_best, key=lambda a: per_asset_best[a].get("edge_per_session", -999))
        bp = per_asset_best[best_asset]
        buy = bp.get("buy", current_buy)
        sell = bp.get("sell", current_sell)
        edge = bp.get("edge_per_session", 0)
        fill_rate = bp.get("fill_rate", 0)
        sell_hit_rate = bp.get("sell_hit_rate", 0)
        if buy:
            shares = position_usdc / buy
            gross = edge * shares
            spread_cost_session = fill_rate * spread_cost * shares + fill_rate * sell_hit_rate * spread_cost * shares
            net_session = gross - spread_cost_session
            net_day = net_session * SESSIONS_PER_DAY
        else:
            net_session = 0.0
            net_day = 0.0
        best_fw = int(bp["fill_window"]) if "fill_window" in bp else 60
        rob = _robustness.get(best_asset, {})
        rob_ratio  = rob.get("robustness_ratio")
        rob_shape  = rob.get("shape", "unknown")
        rob_pct    = rob.get("pct_positive")
        rob_mean   = rob.get("neighbor_mean_edge")
        rob_line = (
            f"`{rob_shape}` — ratio={rob_ratio}, neighbor mean edge={rob_mean}, "
            f"{rob_pct:.0%} of neighbors positive"
            if rob_ratio is not None and rob_pct is not None
            else "_not available_"
        )
        lines += [
            f"**Best asset:** `{best_asset}`",
            f"**Recommended buy threshold:** `{buy}`",
            f"**Recommended sell threshold:** `{sell}`",
            f"**Recommended fill window:** `{best_fw}s`",
            f"**Expected edge per session:** `{edge}`",
            f"**Est. $/session (net, linear):** `${net_session:+.4f}`",
            f"**Est. $/day (net, linear):** `${net_day:+.2f}` at ${capital:,.0f} capital",
            f"**Robustness (±0.04 buy/sell, ±10s window):** {rob_line}",
            "",
            f"To update config, set in `skeptic/config.py`:",
            f"```python",
            f"BUY_PRICE = {buy}",
            f"SELL_PRICE = {bp.get('sell', current_sell)}",
            f"MONITOR_SECS = {best_fw}",
            f"```",
        ]
    else:
        lines.append("_Insufficient data to make a recommendation._")

    # --- Compounding profit section ---
    # Each session one of three things happens (conditional on a fill):
    #   1. Sell hits        → gain (sell - buy) / buy  × position_size_pct  of capital
    #   2. Resolution win   → gain (1.0  - buy) / buy  × position_size_pct  of capital
    #   3. Resolution loss  → lose position_size_pct of capital entirely
    # No fill → capital unchanged.
    #
    # Per-session growth multiplier:
    #   m = (1 - fill_rate)
    #     + fill_rate × [ p_sell × (1 + pct×(sell-buy)/buy)
    #                   + p_res_win  × (1 + pct×(1-buy)/buy)
    #                   + p_res_loss × (1 - pct) ]
    #
    # After N sessions: capital × m^N   (true geometric compounding).
    # lines += [
    #     "",
    #     "---",
    #     "",
    #     "## Compounding Profit Estimate",
    #     "",
    #     f"_Assumes {position_size_pct:.0%} of current capital deployed each filled session. "
    #     f"Capital can never go negative (fractional sizing). "
    #     f"{SESSIONS_PER_DAY} windows/day._",
    #     "",
    #     "| Asset | Buy | Sell | Session mult. | Capital/Day | Capital/Week | Capital/Month | Capital/Year |",
    #     "|-------|-----|------|--------------|-------------|--------------|---------------|--------------|",
    # ]

    # for asset, params in per_asset_best.items():
    #     buy = params.get("buy")
    #     sell = params.get("sell")
    #     fill_rate = params.get("fill_rate", 0)
    #     n_fills = params.get("n_fills") or 0
    #     n_sell_hits = params.get("n_sell_hits") or 0
    #     n_res_wins = params.get("n_res_wins") or 0
    #     n_res_losses = params.get("n_res_losses") or 0

    #     if not (buy and sell and n_fills > 0):
    #         continue

    #     p_sell = n_sell_hits / n_fills
    #     p_res_win = n_res_wins / n_fills
    #     p_res_loss = n_res_losses / n_fills

    #     mult_sell     = 1 + position_size_pct * (sell - buy) / buy
    #     mult_res_win  = 1 + position_size_pct * (1.0 - buy) / buy
    #     mult_res_loss = 1 - position_size_pct

    #     fill_weighted = p_sell * mult_sell + p_res_win * mult_res_win + p_res_loss * mult_res_loss
    #     session_mult  = (1 - fill_rate) * 1.0 + fill_rate * fill_weighted

    #     cap_day   = capital * session_mult ** SESSIONS_PER_DAY
    #     cap_week  = capital * session_mult ** (SESSIONS_PER_DAY * 7)
    #     cap_month = capital * session_mult ** (SESSIONS_PER_DAY * 30)
    #     cap_year  = capital * session_mult ** (SESSIONS_PER_DAY * 365)

    #     lines.append(
    #         f"| {asset} | {buy} | {sell} "
    #         f"| {session_mult:.8f} "
    #         f"| ${cap_day:,.2f} "
    #         f"| ${cap_week:,.2f} "
    #         f"| ${cap_month:,.2f} "
    #         f"| ${cap_year:,.2f} |"
    #     )

    content = "\n".join(lines)
    with open(path, "w") as f:
        f.write(content)
    log.info("Research report written to %s", path)
    return path
