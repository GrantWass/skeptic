"""Tab — High-Buy Strategy: equity curve and per-window charts for the high-probability buy."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from helpers import OUTCOME_BG, OUTCOME_COLORS, make_window_chart
from skeptic.research.analyzer import sweep_high_buy, grid_search_high_buy
from skeptic.research.fetcher import HistoricalSession


SLIPPAGE = 0.06  # assumed 6-cent slippage per fill


def _classify_high_buy(
    s: HistoricalSession,
    threshold: float,
    position_usdc: float,
) -> tuple[str, float] | None:
    """
    Check if either UP or DOWN touched `threshold` during the session.
    Takes the first side to reach the threshold.
    Returns (outcome, pnl_usdc) or None if no fill.
    Assumes 6-cent slippage: effective fill price = threshold + SLIPPAGE.
    """
    effective_price = threshold + SLIPPAGE
    win_payout  =  1.0 - effective_price
    lose_payout = -effective_price

    up_ts = next((ts for ts, p in s.up_trades_all   if p >= threshold), None)
    dn_ts = next((ts for ts, p in s.down_trades_all if p >= threshold), None)

    if up_ts is None and dn_ts is None:
        return None

    if up_ts is not None and (dn_ts is None or up_ts <= dn_ts):
        resolution = s.up_resolution
        fill_side  = "UP"
    else:
        resolution = s.down_resolution
        fill_side  = "DOWN"

    if resolution is None:
        return None

    shares = position_usdc / effective_price
    win    = resolution >= 0.9
    pnl    = (win_payout if win else lose_payout) * shares
    outcome = "Res Win" if win else "Res Loss"
    return outcome, pnl, fill_side


def _equity_curve(sessions: list[HistoricalSession], threshold: float, position_usdc: float) -> go.Figure:
    """Plotly equity curve showing cumulative PnL over sessions."""
    xs, ys, colors, texts = [], [], [], []
    cumulative = 0.0

    for s in sorted(sessions, key=lambda s: s.window_start_ts):
        result = _classify_high_buy(s, threshold, position_usdc)
        dt = datetime.fromtimestamp(s.window_start_ts, tz=timezone.utc)
        if result is None:
            continue
        outcome, pnl, fill_side = result
        cumulative += pnl
        xs.append(dt)
        ys.append(cumulative)
        colors.append(OUTCOME_COLORS[outcome])
        texts.append(f"{dt.strftime('%m/%d %H:%M')} {fill_side} {outcome} {pnl:+.2f}")

    fig = go.Figure()

    # Zero line
    if xs:
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)

    # Filled area under curve
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines",
        line=dict(color="#3b82f6", width=2),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.15)",
        name="Cum PnL",
        hovertext=texts,
        hoverinfo="text+y",
    ))

    # Colored dots per fill
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers",
        marker=dict(color=colors, size=6),
        name="Fills",
        hovertext=texts,
        hoverinfo="text+y",
        showlegend=False,
    ))

    final = ys[-1] if ys else 0
    fig.update_layout(
        title=dict(
            text=f"Equity Curve — buy @ {threshold:.2f}  |  Final: ${final:+.2f}",
            font=dict(size=13), x=0,
        ),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Cumulative PnL ($)", tickformat="$,.2f"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(t=40, b=40, l=60, r=20),
        hovermode="x unified",
    )
    return fig


def _threshold_edge_chart(sessions: list[HistoricalSession], cutoff_secs: int) -> go.Figure:
    """Line chart: threshold (x) vs edge/session (y) for any-time and late-window modes."""
    thresholds = [round(t, 2) for t in np.arange(0.65, 0.96, 0.05)]

    df_any    = sweep_high_buy(sessions, thresholds=thresholds, min_elapsed_secs=0,           max_elapsed_secs=300, slippage=SLIPPAGE)
    df_late   = sweep_high_buy(sessions, thresholds=thresholds, min_elapsed_secs=cutoff_secs, max_elapsed_secs=300, slippage=SLIPPAGE)
    df_any30  = sweep_high_buy(sessions, thresholds=thresholds, min_elapsed_secs=0,           max_elapsed_secs=270, slippage=SLIPPAGE)
    df_late30 = sweep_high_buy(sessions, thresholds=thresholds, min_elapsed_secs=cutoff_secs, max_elapsed_secs=270, slippage=SLIPPAGE)

    for df in (df_any, df_late, df_any30, df_late30):
        df.sort_values("threshold", inplace=True)
    late_label = f"Last {(300 - cutoff_secs) // 60}m {(300 - cutoff_secs) % 60}s".replace(" 0s", "")

    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)

    fig.add_trace(go.Scatter(
        x=df_any["threshold"], y=df_any["edge_per_session"],
        mode="lines+markers", name="Any time",
        line=dict(color="#3b82f6", width=2), marker=dict(size=7),
        hovertemplate="threshold=%{x:.2f}<br>edge/session=%{y:+.4f}<extra>Any time</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df_any30["threshold"], y=df_any30["edge_per_session"],
        mode="lines+markers", name="Any time −30s",
        line=dict(color="#3b82f6", width=2, dash="dot"), marker=dict(size=7),
        hovertemplate="threshold=%{x:.2f}<br>edge/session=%{y:+.4f}<extra>Any time −30s</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df_late["threshold"], y=df_late["edge_per_session"],
        mode="lines+markers", name=late_label,
        line=dict(color="#f59e0b", width=2), marker=dict(size=7),
        hovertemplate=f"threshold=%{{x:.2f}}<br>edge/session=%{{y:+.4f}}<extra>{late_label}</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df_late30["threshold"], y=df_late30["edge_per_session"],
        mode="lines+markers", name=f"{late_label} −30s",
        line=dict(color="#f59e0b", width=2, dash="dot"), marker=dict(size=7),
        hovertemplate=f"threshold=%{{x:.2f}}<br>edge/session=%{{y:+.4f}}<extra>{late_label} −30s</extra>",
    ))

    fig.update_layout(
        title=dict(text="Edge/Session vs Buy Threshold", font=dict(size=13), x=0),
        xaxis=dict(title="Buy Threshold", tickformat=".2f"),
        yaxis=dict(title="Edge / Session", tickformat="+.4f"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(t=40, b=40, l=70, r=20),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def render(
    all_sessions: dict[str, list[HistoricalSession]],
    selected_assets: list[str],
    capital: float,
    position_pct: float,
) -> None:
    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        asset = st.selectbox("Asset", selected_assets, key="hb_asset")
    with col_b:
        threshold = st.slider("Buy threshold", 0.65, 0.95, 0.65, 0.05, format="%.2f", key="hb_threshold")
    with col_c:
        cutoff_secs = st.slider("Late-window cutoff", 0, 270, 180, 30,
                                format="%ds", key="hb_cutoff",
                                help="Only count triggers after this many seconds into the window (0 = any time)")

    sessions = sorted(all_sessions.get(asset, []), key=lambda s: s.window_start_ts)
    if not sessions:
        st.info(f"No sessions for {asset}.")
        return

    position_usdc = capital * position_pct

    # Classify all sessions
    filled = []
    for s in sessions:
        result = _classify_high_buy(s, threshold, position_usdc)
        if result is not None:
            outcome, pnl, fill_side = result
            filled.append((s, outcome, pnl, fill_side))

    n_wins   = sum(1 for _, o, _, _ in filled if o == "Res Win")
    n_losses = sum(1 for _, o, _, _ in filled if o == "Res Loss")
    total_pnl = sum(p for _, _, p, _ in filled)
    win_rate = n_wins / len(filled) if filled else 0.0
    effective_price = threshold + SLIPPAGE
    break_even = effective_price  # must win > effective_price of the time to profit

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Sessions", len(sessions))
    m2.metric("Fills", len(filled))
    m3.metric("Fill Rate", f"{len(filled)/len(sessions):.1%}")
    m4.metric("Win Rate", f"{win_rate:.1%}", delta=f"{win_rate - break_even:+.1%} vs {break_even:.0%} break-even")
    m5.metric("Wins / Losses", f"{n_wins} / {n_losses}")
    m6.metric("Total PnL", f"${total_pnl:+.2f}")

    st.caption(
        f"threshold={threshold:.2f}  slippage={SLIPPAGE:.2f}  effective fill={effective_price:.2f}  "
        f"position=${position_usdc:.2f}  break-even win rate={break_even:.0%}"
    )
    st.divider()

    # Equity curve
    st.plotly_chart(_equity_curve(sessions, threshold, position_usdc), width="stretch")

    st.divider()

    # Threshold vs edge/session chart (both modes)
    st.plotly_chart(_threshold_edge_chart(sessions, cutoff_secs), width="stretch")

    st.divider()

    # Grid search heatmap: threshold × window cutoff
    st.subheader("Grid Search: Threshold × Entry Window")
    df_grid = grid_search_high_buy(sessions, slippage=SLIPPAGE)
    pivot = df_grid.pivot(index="cutoff_secs", columns="threshold", values="edge_per_session")
    y_labels = [f"Any time" if c == 0 else f"After {c}s" for c in pivot.index]
    fig_grid = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{t:.2f}" for t in pivot.columns],
        y=y_labels,
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="Edge/Session", tickformat="+.3f"),
        hovertemplate="threshold=%{x}<br>cutoff=%{y}<br>edge/session=%{z:+.4f}<extra></extra>",
    ))
    fig_grid.update_layout(
        xaxis=dict(title="Buy Threshold"),
        yaxis=dict(title="Window Entry Point", autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(t=20, b=60, l=80, r=20),
    )
    st.plotly_chart(fig_grid, width="stretch")

    st.divider()

    # Per-window charts for filled sessions only, most recent first
    st.subheader(f"Filled Windows ({len(filled)})")
    show_wins   = st.checkbox("Show wins",   value=True,  key="hb_wins")
    show_losses = st.checkbox("Show losses", value=True,  key="hb_losses")

    visible = [
        (s, outcome, pnl, fill_side)
        for s, outcome, pnl, fill_side in reversed(filled)
        if (outcome == "Res Win" and show_wins) or (outcome == "Res Loss" and show_losses)
    ]

    col_l, col_r = st.columns(2)
    for i, (s, outcome, pnl, fill_side) in enumerate(visible):
        col = col_l if i % 2 == 0 else col_r
        with col:
            # Reuse make_window_chart — pass threshold as both buy (fill level) and sell=1.0
            # but annotate with the high-buy outcome via a custom title override
            fig = make_window_chart(s, buy=threshold, sell=1.0, outcome=outcome, profit=pnl, fill_window=300)
            # Add a horizontal line at threshold to mark the high-buy trigger
            fig.add_hline(
                y=threshold,
                line_dash="solid", line_color="#f59e0b", line_width=2,
                annotation_text=f"buy {threshold:.2f}", annotation_position="top right",
                annotation_font_size=9,
            )
            fig.update_layout(
                title=dict(
                    text=(
                        f"<b>{datetime.fromtimestamp(s.window_start_ts, tz=timezone.utc).strftime('%m/%d %H:%M')} UTC</b>  "
                        f"<span style='color:{OUTCOME_COLORS[outcome]}'><b>{fill_side} → {outcome}</b></span>  "
                        f"${pnl:+.2f}"
                    ),
                    font=dict(size=11), x=0, xanchor="left",
                ),
                plot_bgcolor=OUTCOME_BG.get(outcome, "white"),
            )
            st.plotly_chart(fig, width="stretch")
