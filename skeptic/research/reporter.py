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


# ── Formatting helpers ────────────────────────────────────────────────────────

def _pct(v) -> str:
    return f"{v:.1%}" if isinstance(v, (int, float)) else "—"

def _edge(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

def _price(v) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else "—"

def _ratio(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

def _shape_label(shape: str) -> str:
    return {"plateau": "plateau", "moderate": "moderate", "spike": "spike"}.get(shape, "unknown")

def _agree_label(agree: str) -> str:
    return agree if agree else "—"


def write_report(
    per_asset_best: dict[str, dict],
    asset_ranking: pd.DataFrame,
    per_asset_robustness: dict[str, dict] | None = None,
    per_asset_best_nb: dict[str, dict] | None = None,
    per_asset_best_30pct: dict[str, dict] | None = None,
    current_buy: float | None = config.BUY_PRICE,
    current_sell: float | None = config.SELL_PRICE,
    data_source: str = "api",
    capital: float = 500.0,
    position_size_pct: float = config.POSITION_SIZE_PCT,
    spread_cost: float = 0.002,
) -> str:
    """Write the markdown research report. Returns path."""
    _ensure_reports_dir()
    for _old in glob.glob(os.path.join(config.REPORTS_DIR, "research_report_*.md")):
        os.remove(_old)
    suffix = random.randint(10000, 99999)
    path = os.path.join(config.REPORTS_DIR, f"research_report_{suffix}.md")
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    _robustness = per_asset_robustness or {}
    _best_nb    = per_asset_best_nb    or {}
    _best_30    = per_asset_best_30pct or {}

    SESSIONS_PER_DAY = 288  # 24h × 60min / 5min
    position_usdc = capital * position_size_pct

    # ── Recommendation: prefer moderate/plateau over raw edge ─────────────────
    def _asset_priority(a):
        shape = _robustness.get(a, {}).get("shape", "spike")
        edge  = per_asset_best[a].get("edge_per_session", -999)
        rank  = {"plateau": 2, "moderate": 1, "spike": 0, "unknown": 0}
        return (rank.get(shape, 0), edge)

    best_asset = max(per_asset_best, key=_asset_priority) if per_asset_best else None

    # ── TL;DR ─────────────────────────────────────────────────────────────────
    lines = [
        "# Skeptic Research Report",
        f"_Generated: {now}_",
        "",
        "## TL;DR",
        "",
        "| Asset | Sessions | Fill Rate | Edge/Session | Shape | Note |",
        "|-------|----------|-----------|--------------|-------|------|",
    ]

    for asset, params in per_asset_best.items():
        n_sess  = int(params.get("n_sessions", 0))
        fr      = params.get("fill_rate", 0)
        edge    = params.get("edge_per_session")
        shape   = _robustness.get(asset, {}).get("shape", "unknown")
        agree   = _best_nb.get(asset, {}).get("peak_vs_neighborhood", "")
        lines.append(
            f"| {asset} | {n_sess} | {_pct(fr)} | {_edge(edge)} | {_shape_label(shape)} |"
        )

    # ── Methodology ───────────────────────────────────────────────────────────
    fw_desc = f"first {config.MONITOR_SECS} seconds"
    if data_source == "prices":
        methodology_lines = [
            "Per-second mid-prices were collected via `collect_prices.py` and stored as CSV files in `data/prices/`.",
            "Each 5-minute window's price series is used to simulate fills and exits.",
            "",
            "### Strategy simulation",
            f"- **Fill** occurs if the mid-price for either UP or DOWN ≤ `buy_threshold` within the {fw_desc}",
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
            "> Results should be treated as directional signals, not precise predictions.",
            "",
            "### Strategy simulation",
            f"- **Fill** occurs if the minimum trade price for either UP or DOWN ≤ `buy_threshold` within the {fw_desc}",
            "- **Sell hit** occurs if the max subsequent trade price ≥ `sell_threshold`",
            "- Otherwise position goes to resolution (1.0 or 0.0)",
            "- **Edge per session** = fill_rate × expected_PnL_per_fill",
        ]

    lines += [
        "",
        "---",
        "",
        "## Methodology",
        "",
        *methodology_lines,
    ]

    # ── Asset Ranking (only when thresholds are set) ──────────────────────────
    if current_buy is not None and current_sell is not None and not asset_ranking.empty:
        lines += [
            "",
            "---",
            "",
            f"## Asset Ranking at Current Thresholds (buy={current_buy:.2f}, sell={current_sell:.2f})",
            "",
            asset_ranking.to_markdown(index=False),
        ]

    # ── Optimal Parameters ────────────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## Optimal Parameters per Asset",
        "",
        "> Parameters with the highest edge across the full grid.",
        "> **Peak vs NB** compares the peak to the best surrounding region — diverge means the peak may be a lucky outlier.",
        "> **Shape** measures whether the peak sits on a broad plateau or is a lone spike.",
        "> **NB Ratio** = neighbor mean edge / peak edge. How much of the peak's edge survives in the surrounding grid points (±1 step in each direction).",
        "> - 1.0 = neighbors match the peak exactly — pure plateau",
        "> - 0.6 = neighbors average 60% of peak edge — reasonably flat",
        "> - 0.0 or negative = neighbors have zero or negative edge — lone spike surrounded by losing combinations, strong overfitting signal",
        "",
        "| Asset | Sessions | Buy | Sell | Fill Win | Fill Rate | Sell Hit% | Edge/Session | Shape | NB Ratio | Peak vs NB |",
        "|-------|----------|-----|------|----------|-----------|-----------|--------------|-------|----------|------------|",
    ]

    for asset, params in per_asset_best.items():
        n_sess = int(params.get("n_sessions", 0))
        buy    = params.get("buy")
        sell   = params.get("sell")
        fw     = f"{int(params['fill_window'])}s" if "fill_window" in params else "60s"
        fr     = params.get("fill_rate")
        shr    = params.get("sell_hit_rate")
        edge   = params.get("edge_per_session")
        rob    = _robustness.get(asset, {})
        shape  = _shape_label(rob.get("shape", "unknown"))
        ratio  = _ratio(rob.get("robustness_ratio"))
        nb     = _best_nb.get(asset, {})
        agree  = _agree_label(nb.get("peak_vs_neighborhood", "—"))
        lines.append(
            f"| {asset} | {n_sess} | {_price(buy)} | {_price(sell)} | {fw} "
            f"| {_pct(fr)} | {_pct(shr)} | {_edge(edge)} | {shape} | {ratio} | {agree} |"
        )

    # ── Best Neighborhood ─────────────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## Best Neighborhood per Asset",
        "",
        "> The region whose *average* surrounding edge is highest — more trustworthy than the raw peak when they disagree.",
        "> Use these parameters instead of the peak when **Peak vs NB** shows diverge.",
        "",
        "| Asset | NB Buy | NB Sell | NB Fill Win | NB Mean Edge | % NB Positive | Peak Edge | Peak vs NB |",
        "|-------|--------|---------|-------------|--------------|---------------|-----------|------------|",
    ]
    for asset in per_asset_best:
        nb = _best_nb.get(asset, {})
        if not nb:
            lines.append(f"| {asset} | — | — | — | — | — | — | — |")
            continue
        nb_fw  = f"{nb['fill_window']}s"      if nb.get("fill_window")       is not None else "—"
        pct_p  = _pct(nb.get("pct_positive"))
        agree  = _agree_label(nb.get("peak_vs_neighborhood", "—"))
        lines.append(
            f"| {asset} | {_price(nb.get('buy'))} | {_price(nb.get('sell'))} | {nb_fw} "
            f"| {_edge(nb.get('neighborhood_mean_edge'))} | {pct_p} "
            f"| {_edge(nb.get('peak_edge'))} | {agree} |"
        )

    # ── ≥30% Fill Rate Constrained ────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## Optimal Parameters at ≥30% Fill Rate",
        "",
        "> Only combinations where fill_rate ≥ 30% are eligible.",
        "> Low fill rates can produce high edge from a small lucky sample.",
        "> This table forces the strategy to trigger regularly.",
        "",
        "| Asset | Buy | Sell | Fill Win | Fill Rate | Sell Hit% | Edge/Session |",
        "|-------|-----|------|----------|-----------|-----------|--------------|",
    ]
    for asset in per_asset_best:
        p30 = _best_30.get(asset, {})
        if not p30:
            lines.append(f"| {asset} | — | — | — | — | — | _no combo meets 30% fill rate_ |")
            continue
        fw30 = f"{int(p30['fill_window'])}s" if "fill_window" in p30 else "60s"
        lines.append(
            f"| {asset} | {_price(p30.get('buy'))} | {_price(p30.get('sell'))} | {fw30} "
            f"| {_pct(p30.get('fill_rate'))} | {_pct(p30.get('sell_hit_rate'))} "
            f"| {_edge(p30.get('edge_per_session'))} |"
        )

    # ── Estimated Profit ──────────────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## Estimated Profit",
        "",
        f"_Assumptions: ${capital:,.0f} starting capital, {position_size_pct:.0%} per trade, {SESSIONS_PER_DAY} windows/day._",
        f"_Spread cost: {spread_cost:.3f}/share/crossing. Polymarket charges 0% maker/taker fees._",
        "",
        "> **These numbers are theoretical upper bounds.** They assume the in-sample edge holds",
        "> perfectly out-of-sample, every session runs, and execution matches simulation exactly.",
        "> Treat them as a relative ranking between assets, not a cash forecast.",
        "> Assets with shape=spike are especially likely to underperform these estimates.",
        "",
        "| Asset | Buy | Sell | Shape | $/Session (net) | $/Day | $/Week | $/Month |",
        "|-------|-----|------|-------|-----------------|-------|--------|---------|",
    ]

    for asset, params in per_asset_best.items():
        buy       = params.get("buy")
        sell      = params.get("sell")
        edge      = params.get("edge_per_session")
        fill_rate = params.get("fill_rate", 0)
        shr       = params.get("sell_hit_rate", 0)
        shape     = _shape_label(_robustness.get(asset, {}).get("shape", "unknown"))
        if buy and sell and edge is not None and buy > 0:
            shares  = position_usdc / buy
            gross   = edge * shares
            spread_ = fill_rate * spread_cost * shares + fill_rate * shr * spread_cost * shares
            net     = gross - spread_
            lines.append(
                f"| {asset} | {_price(buy)} | {_price(sell)} | {shape} "
                f"| ${net:+.4f} | ${net * SESSIONS_PER_DAY:+.2f} "
                f"| ${net * SESSIONS_PER_DAY * 7:+.2f} "
                f"| ${net * SESSIONS_PER_DAY * 30:+.2f} |"
            )

    # ── Estimated Profit — Best Neighborhood Params ───────────────────────────
    lines += [
        "",
        "### Best Neighborhood Params",
        "",
        "_Uses the neighborhood mean edge (average of surrounding grid points) as the edge estimate._",
        "_Fill rate is approximated from the peak params for each asset._",
        "",
        "| Asset | NB Buy | NB Sell | NB Fill Win | NB Mean Edge | $/Session (net) | $/Day | $/Week | $/Month |",
        "|-------|--------|---------|-------------|-------------|-----------------|-------|--------|---------|",
    ]

    for asset, nb in _best_nb.items():
        nb_buy  = nb.get("buy")
        nb_sell = nb.get("sell")
        nb_edge = nb.get("neighborhood_mean_edge")
        nb_fw   = nb.get("fill_window")
        # Use peak fill_rate as approximation
        peak    = per_asset_best.get(asset, {})
        fill_rate = peak.get("fill_rate", 0)
        shr       = peak.get("sell_hit_rate", 0)
        fw_str  = f"{int(nb_fw)}s" if nb_fw is not None else "—"
        if nb_buy and nb_sell and nb_edge is not None and nb_buy > 0:
            shares  = position_usdc / nb_buy
            gross   = nb_edge * shares
            spread_ = fill_rate * spread_cost * shares + fill_rate * shr * spread_cost * shares
            net     = gross - spread_
            lines.append(
                f"| {asset} | {_price(nb_buy)} | {_price(nb_sell)} | {fw_str} "
                f"| {_edge(nb_edge)} "
                f"| ${net:+.4f} | ${net * SESSIONS_PER_DAY:+.2f} "
                f"| ${net * SESSIONS_PER_DAY * 7:+.2f} "
                f"| ${net * SESSIONS_PER_DAY * 30:+.2f} |"
            )

    # ── Recommendation ────────────────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## Recommendation",
        "",
    ]

    if best_asset:
        bp        = per_asset_best[best_asset]
        buy       = bp.get("buy", current_buy)
        sell      = bp.get("sell", current_sell)
        edge      = bp.get("edge_per_session", 0)
        fill_rate = bp.get("fill_rate", 0)
        shr       = bp.get("sell_hit_rate", 0)
        best_fw   = int(bp["fill_window"]) if "fill_window" in bp else 60
        rob       = _robustness.get(best_asset, {})
        shape     = rob.get("shape", "unknown")
        rob_ratio = rob.get("robustness_ratio")
        rob_pct   = rob.get("pct_positive")
        rob_mean  = rob.get("neighbor_mean_edge")
        nb        = _best_nb.get(best_asset, {})
        agree     = nb.get("peak_vs_neighborhood", "")

        def _net(b, e, fr, s):
            if not b or e is None:
                return 0.0, 0.0
            sh = position_usdc / b
            sp = fr * spread_cost * sh + fr * s * spread_cost * sh
            ns = e * sh - sp
            return ns, ns * SESSIONS_PER_DAY

        net_session, net_day = _net(buy, edge, fill_rate, shr)

        rob_line = (
            f"{_shape_label(shape)} — ratio={_ratio(rob_ratio)}, "
            f"neighbor mean edge={_edge(rob_mean)}, "
            f"{_pct(rob_pct)} of neighbors positive"
            if rob_ratio is not None and rob_pct is not None
            else "_not available_"
        )

        # Flags (no emojis)
        flags = []
        if shape == "spike":
            flags.append("> Note: shape is spike — the peak has weak neighborhood support. See Best Neighborhood recommendation below.")
        if best_fw < 30:
            flags.append(f"> Note: fill window is {best_fw}s — very short; verify this isn't a noise artifact.")

        lines += [
            f"**Best asset:** {best_asset}",
            "",
            "### Option A — Peak Threshold (highest raw edge)",
            "",
            f"| | Value |",
            f"|---|---|",
            f"| Buy | {_price(buy)} |",
            f"| Sell | {_price(sell)} |",
            f"| Fill window | {best_fw}s |",
            f"| Edge/session | {_edge(edge)} |",
            f"| Fill rate | {_pct(fill_rate)} |",
            f"| Sell hit rate | {_pct(shr)} |",
            f"| Est. $/session (net) | ${net_session:+.4f} |",
            f"| Est. $/day | ${net_day:+.2f} |",
            f"| Robustness | {rob_line} |",
            "",
        ]
        if flags:
            lines += flags
            lines.append("")

        lines += [
            "```python",
            f"# Option A — config.py",
            f"BUY_PRICE    = {buy}",
            f"SELL_PRICE   = {sell}",
            f"MONITOR_SECS = {best_fw}",
            "```",
            "",
        ]

        # Option B — best neighborhood
        if nb:
            nb_buy  = nb.get("buy")
            nb_sell = nb.get("sell")
            nb_fw   = nb.get("fill_window", best_fw)
            nb_edge = nb.get("neighborhood_mean_edge")
            nb_pct  = nb.get("pct_positive")
            nb_ns, nb_nd = _net(nb_buy, nb_edge, fill_rate, shr)
            nb_agree = nb.get("peak_vs_neighborhood", "—")

            lines += [
                "### Option B — Best Neighborhood (most robust region)",
                "",
                "> Uses the center of the region with the highest average surrounding edge.",
                "> Prefer this when shape=spike or peak vs NB=diverge.",
                "",
                f"| | Value |",
                f"|---|---|",
                f"| Buy | {_price(nb_buy)} |",
                f"| Sell | {_price(nb_sell)} |",
                f"| Fill window | {nb_fw}s |",
                f"| NB mean edge | {_edge(nb_edge)} |",
                f"| % NB positive | {_pct(nb_pct)} |",
                f"| Est. $/session (net, approx) | ${nb_ns:+.4f} |",
                f"| Est. $/day (approx) | ${nb_nd:+.2f} |",
                f"| Peak vs NB | {nb_agree} |",
                "",
                "```python",
                f"# Option B — config.py",
                f"BUY_PRICE    = {nb_buy}",
                f"SELL_PRICE   = {nb_sell}",
                f"MONITOR_SECS = {nb_fw}",
                "```",
            ]
    else:
        lines.append("_Insufficient data to make a recommendation._")

    content = "\n".join(lines)
    with open(path, "w") as f:
        f.write(content)
    log.info("Research report written to %s", path)
    return path
