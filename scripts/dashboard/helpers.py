"""
Shared constants and pure-computation helpers used across dashboard tabs.

Nothing in here renders Streamlit widgets.
"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import plotly.graph_objects as go

from skeptic.research.fetcher import HistoricalSession

# ── Constants ──────────────────────────────────────────────────────────────────

SESSIONS_PER_DAY = 288  # 24h × 60min / 5min per window

OUTCOME_COLORS = {
    "No Fill":  "#6b7280",
    "Sell Hit": "#22c55e",
    "Res Win":  "#3b82f6",
    "Res Loss": "#ef4444",
}
OUTCOME_BG = {
    "No Fill":  "rgba(107,114,128,0.07)",
    "Sell Hit": "rgba(34,197,94,0.12)",
    "Res Win":  "rgba(59,130,246,0.12)",
    "Res Loss": "rgba(239,68,68,0.12)",
}


# ── Fill-window helpers ────────────────────────────────────────────────────────

def _m1(s: HistoricalSession, fill_window: int) -> tuple[list, list]:
    """Return (up_trades, dn_trades) filtered to the fill window."""
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


# ── Window scatter data ────────────────────────────────────────────────────────

def build_window_rows(
    sessions: list[HistoricalSession],
    buy: float,
    sell: float,
    fill_window: int = 60,
) -> pd.DataFrame:
    """One row per session: fill price, max price after fill, and outcome."""
    rows = []
    for s in sessions:
        up_m1, dn_m1 = _m1(s, fill_window)
        up_min = min((p for _, p in up_m1), default=None)
        dn_min = min((p for _, p in dn_m1), default=None)
        up_fill = up_min is not None and up_min <= buy
        down_fill = dn_min is not None and dn_min <= buy

        if not up_fill and not down_fill:
            rows.append({
                "fill_price": min(
                    up_min if up_min is not None else 1.0,
                    dn_min if dn_min is not None else 1.0,
                ),
                "max_after": None,
                "outcome": "No Fill",
            })
            continue

        if up_fill and down_fill:
            up_ts = next((ts for ts, p in up_m1 if p <= buy), None)
            dn_ts = next((ts for ts, p in dn_m1 if p <= buy), None)
            use_up = up_ts is not None and (dn_ts is None or up_ts <= dn_ts)
        else:
            use_up = up_fill

        if use_up:
            fill_ts   = next((ts for ts, p in up_m1 if p <= buy), None)
            fill_price = up_min
            max_after  = _max_after(s.up_trades_all, fill_ts)
            sell_hit   = (max_after or 0.0) >= sell
            res_win    = (s.up_resolution or 0.0) >= 0.9
        else:
            fill_ts    = next((ts for ts, p in dn_m1 if p <= buy), None)
            fill_price = dn_min
            max_after  = _max_after(s.down_trades_all, fill_ts)
            sell_hit   = (max_after or 0.0) >= sell
            res_win    = (s.down_resolution or 0.0) >= 0.9

        if sell_hit:
            outcome = "Sell Hit"
        elif res_win:
            outcome = "Res Win"
        else:
            outcome = "Res Loss"

        rows.append({"fill_price": fill_price, "max_after": max_after, "outcome": outcome})

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Profit table ───────────────────────────────────────────────────────────────

def compute_profit_table(
    results: pd.DataFrame,
    buy: float,
    capital: float,
    position_pct: float,
    spread_cost: float,
) -> pd.DataFrame:
    position_usdc = capital * position_pct
    rows = []
    for _, r in results.iterrows():
        shares = position_usdc / buy
        edge = r["Edge/Session"]
        fill_rate = r["Fill Rate"]
        sell_hit_rate = r["Sell Hit Rate"]
        gross = edge * shares
        spread = fill_rate * spread_cost * shares + fill_rate * sell_hit_rate * spread_cost * shares
        net = gross - spread
        rows.append({
            "Asset": r["Asset"],
            "$/Session (net)": net,
            "$/Day": net * SESSIONS_PER_DAY,
            "$/Week": net * SESSIONS_PER_DAY * 7,
            "$/Month": net * SESSIONS_PER_DAY * 30,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Window classifier & chart ──────────────────────────────────────────────────

def classify_window(
    s: HistoricalSession,
    buy: float,
    sell: float,
    position_usdc: float,
    spread_cost: float,
    fill_window: int = 60,
) -> tuple[str, float]:
    """Return (outcome_label, realized_pnl_usdc) for a single window."""
    up_m1, dn_m1 = _m1(s, fill_window)
    up_min  = min((p for _, p in up_m1), default=None)
    dn_min  = min((p for _, p in dn_m1), default=None)
    up_fill   = up_min is not None and up_min <= buy
    down_fill = dn_min is not None and dn_min <= buy

    if not up_fill and not down_fill:
        return "No Fill", 0.0

    if up_fill and down_fill:
        up_ts = next((ts for ts, p in up_m1 if p <= buy), None)
        dn_ts = next((ts for ts, p in dn_m1 if p <= buy), None)
        use_up = up_ts is not None and (dn_ts is None or up_ts <= dn_ts)
    else:
        use_up = up_fill

    if use_up:
        fill_ts   = next((ts for ts, p in up_m1 if p <= buy), None)
        max_after = _max_after(s.up_trades_all, fill_ts)
        res = s.up_resolution
    else:
        fill_ts   = next((ts for ts, p in dn_m1 if p <= buy), None)
        max_after = _max_after(s.down_trades_all, fill_ts)
        res = s.down_resolution

    shares = position_usdc / buy
    entry_spread = spread_cost * shares

    if (max_after or 0.0) >= sell:
        profit = (sell - buy) * shares - 2 * entry_spread
        return "Sell Hit", profit
    elif (res or 0.0) >= 0.9:
        profit = (1.0 - buy) * shares - entry_spread
        return "Res Win", profit
    else:
        profit = -(buy * shares) - entry_spread
        return "Res Loss", profit


def make_window_chart(
    s: HistoricalSession,
    buy: float,
    sell: float,
    outcome: str,
    profit: float,
    fill_window: int = 60,
) -> go.Figure:
    """Build a compact time-series chart for a single 5-minute window."""
    fig = go.Figure()
    start = s.window_start_ts

    if s.up_trades_all:
        xs = [ts - start for ts, _ in s.up_trades_all]
        ys = [p for _, p in s.up_trades_all]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, name="UP",
            line=dict(color="#3b82f6", width=1.5), mode="lines",
        ))

    if s.down_trades_all:
        xs = [ts - start for ts, _ in s.down_trades_all]
        ys = [p for _, p in s.down_trades_all]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, name="DOWN",
            line=dict(color="#f97316", width=1.5), mode="lines",
        ))

    fig.add_hline(
        y=buy, line_dash="dash", line_color="#f59e0b", line_width=1.5,
        annotation_text=f"buy {buy:.2f}", annotation_position="bottom right",
        annotation_font_size=9,
    )
    fig.add_hline(
        y=sell, line_dash="dash", line_color="#22c55e", line_width=1.5,
        annotation_text=f"sell {sell:.2f}", annotation_position="top right",
        annotation_font_size=9,
    )
    fig.add_vrect(x0=0, x1=fill_window, fillcolor="rgba(255,255,255,0.06)", line_width=0)
    fig.add_vline(
        x=fill_window, line_dash="dot", line_color="black", line_width=1,
        annotation_text=f"{fill_window}s", annotation_position="top",
        annotation_font_size=8,
    )

    outcome_color = OUTCOME_COLORS.get(outcome, "#6b7280")
    profit_str = f"${profit:+.2f}" if outcome != "No Fill" else "—"
    dt = datetime.fromtimestamp(start, tz=timezone.utc).strftime("%m/%d %H:%M")

    fig.update_layout(
        plot_bgcolor=OUTCOME_BG.get(outcome, "white"),
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(
            text=(
                f"<b>{dt} UTC</b>  "
                f"<span style='color:{outcome_color}'><b>{outcome}</b></span>  "
                f"{profit_str}"
            ),
            font=dict(size=11), x=0, xanchor="left",
        ),
        xaxis=dict(title="s", range=[0, 300], tickfont=dict(size=9), dtick=60),
        yaxis=dict(range=[0, 1], tickfont=dict(size=9), tickformat=".2f"),
        showlegend=False,
        height=210,
        margin=dict(t=32, b=28, l=40, r=60),
    )
    return fig
