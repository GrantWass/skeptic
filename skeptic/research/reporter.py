"""
Generates research reports from threshold optimization results.

Outputs:
  - data/reports/optimal_params.csv   — best (buy, sell) per asset
  - data/reports/asset_ranking.csv    — assets ranked at fixed thresholds
  - data/reports/research_report.md   — human-readable summary
"""
import os
import logging
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
    current_buy: float = config.BUY_PRICE,
    current_sell: float = config.SELL_PRICE,
    data_source: str = "api",
    capital: float = 500.0,
    position_size_pct: float = config.POSITION_SIZE_PCT,
    spread_cost: float = 0.002,
) -> str:
    """Write the markdown research report. Returns path."""
    _ensure_reports_dir()
    path = os.path.join(config.REPORTS_DIR, "research_report.md")
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
        f"## Asset Ranking at Current Thresholds (buy={current_buy:.2f}, sell={current_sell:.2f})",
        "",
    ]

    if not asset_ranking.empty:
        lines.append(asset_ranking.to_markdown(index=False))
    else:
        lines.append("_No data available._")

    lines += [
        "",
        "---",
        "",
        "## Optimal Thresholds per Asset",
        "",
        "| Asset | Optimal Buy | Optimal Sell | Fill Rate | Sell Hit Rate | Edge/Session |",
        "|-------|-------------|--------------|-----------|---------------|--------------|",
    ]

    SESSIONS_PER_DAY = 288  # 24h × 60min / 5min
    position_usdc = capital * position_size_pct

    for asset, params in per_asset_best.items():
        buy = params.get("buy", "—")
        sell = params.get("sell", "—")
        fr = params.get("fill_rate", "—")
        shr = params.get("sell_hit_rate", "—")
        edge = params.get("edge_per_session", "—")
        lines.append(f"| {asset} | {buy} | {sell} | {fr} | {shr} | {edge} |")

    # --- Estimated profit section ---
    # Spread cost: paid once on entry (buy) and once on exit (sell).
    # Each crossing costs ~half the spread, so two crossings = spread_cost per round-trip.
    # Applied per fill event (buy) + per sell event (conditional).
    # Polymarket CLOB charges 0% maker and 0% taker fees; only spread applies.
    lines += [
        "",
        "---",
        "",
        "## Estimated Profit",
        "",
        f"_Assumptions: ${capital:,.0f} starting capital, {position_size_pct:.0%} position size "
        f"(${position_usdc:.2f}/trade), {SESSIONS_PER_DAY} windows/day._",
        "",
        f"_Fees: Polymarket CLOB charges 0% maker/taker fees. Spread cost = {spread_cost:.3f} per share "
        f"per crossing (entry + exit = {spread_cost * 2:.3f} round-trip). "
        f"Adjust with `--spread-cost` if observed spread differs._",
        "",
        "| Asset | Buy | Sell | $/Session (gross) | Spread Cost/Session | $/Session (net) | $/Day (net) | $/Week (net) | $/Month (net) |",
        "|-------|-----|------|-------------------|---------------------|-----------------|-------------|--------------|---------------|",
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
            # Spread cost: pay on entry when fill occurs, pay again on exit when sell executes
            spread_per_session = fill_rate * spread_cost * shares + fill_rate * sell_hit_rate * spread_cost * shares
            net_per_session = gross_per_session - spread_per_session
            net_day = net_per_session * SESSIONS_PER_DAY
            lines.append(
                f"| {asset} | {buy} | {sell} "
                f"| ${gross_per_session:+.4f} "
                f"| ${spread_per_session:.4f} "
                f"| ${net_per_session:+.4f} "
                f"| ${net_day:+.2f} "
                f"| ${net_day * 7:+.2f} "
                f"| ${net_day * 30:+.2f} |"
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
            net_day = (gross - spread_cost_session) * SESSIONS_PER_DAY
        else:
            net_day = 0
        lines += [
            f"**Best asset:** `{best_asset}`",
            f"**Recommended buy threshold:** `{buy}`",
            f"**Recommended sell threshold:** `{sell}`",
            f"**Expected edge per session:** `{edge}`",
            f"**Estimated daily profit (net of spread):** `${net_day:+.2f}` on ${capital:,.0f} capital",
            "",
            f"To update config, set in `skeptic/config.py`:",
            f"```python",
            f"BUY_PRICE = {buy}",
            f"SELL_PRICE = {bp.get('sell', current_sell)}",
            f"```",
        ]
    else:
        lines.append("_Insufficient data to make a recommendation._")

    content = "\n".join(lines)
    with open(path, "w") as f:
        f.write(content)
    log.info("Research report written to %s", path)
    return path
