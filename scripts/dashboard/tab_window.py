"""Tab 2 — Window Detail: fill-price vs max-after-fill scatter per asset."""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from helpers import OUTCOME_COLORS, build_window_rows
from skeptic.research.fetcher import HistoricalSession


def render(
    all_sessions: dict[str, list[HistoricalSession]],
    selected_assets: list[str],
    buy: float,
    sell: float,
    fill_window: int = 60,
) -> None:
    detail_asset = st.selectbox("Asset", selected_assets, key="detail_asset")
    sessions = all_sessions.get(detail_asset, [])

    if not sessions:
        st.info(f"No sessions loaded for {detail_asset}.")
        return

    df_w = build_window_rows(sessions, buy, sell, fill_window)

    fills = df_w[df_w["outcome"] != "No Fill"].copy()
    no_fills = df_w[df_w["outcome"] == "No Fill"]

    n_total = len(df_w)
    n_fills_count = len(fills)
    n_sell_hits = (fills["outcome"] == "Sell Hit").sum()
    n_res_win = (fills["outcome"] == "Res Win").sum()
    n_res_loss = (fills["outcome"] == "Res Loss").sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Windows", n_total)
    c2.metric("Filled", f"{n_fills_count} ({n_fills_count/n_total:.0%})")
    c3.metric("Sell Hit", f"{n_sell_hits} ({n_sell_hits/n_total:.0%})")
    c4.metric("Res Win", f"{n_res_win} ({n_res_win/n_total:.0%})")
    c5.metric("Res Loss", f"{n_res_loss} ({n_res_loss/n_total:.0%})")

    st.divider()
    st.subheader(f"{detail_asset} — fill price vs max price after fill")
    st.caption(
        "Each dot is one 5-minute window that resulted in a fill. "
        "X = lowest price touched in minute 1 (your buy fills here). "
        "Y = highest price reached after the fill (sell target lives here)."
    )

    if fills.empty:
        st.info("No fills at this buy threshold. Try raising the buy slider.")
    else:
        fig_scatter = go.Figure()

        for outcome, color in OUTCOME_COLORS.items():
            if outcome == "No Fill":
                continue
            subset = fills[fills["outcome"] == outcome]
            if subset.empty:
                continue
            fig_scatter.add_trace(go.Scatter(
                x=subset["fill_price"],
                y=subset["max_after"].fillna(0),
                mode="markers",
                name=outcome,
                marker=dict(color=color, size=7, opacity=0.7),
                hovertemplate=(
                    f"<b>{outcome}</b><br>"
                    "Fill price: %{x:.3f}<br>"
                    "Max after: %{y:.3f}<extra></extra>"
                ),
            ))

        fig_scatter.add_vline(
            x=buy, line_dash="dash", line_color="#f59e0b", line_width=2,
            annotation_text=f"buy={buy:.2f}", annotation_position="top right",
        )
        fig_scatter.add_hline(
            y=sell, line_dash="dash", line_color="#a78bfa", line_width=2,
            annotation_text=f"sell={sell:.2f}", annotation_position="right",
        )
        fig_scatter.update_layout(
            xaxis_title="Fill price (min m1 price)",
            yaxis_title="Max price reached after fill",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=480,
            margin=dict(t=40, b=0),
        )
        st.plotly_chart(fig_scatter, width="stretch")

    if not no_fills.empty:
        with st.expander(f"No-fill windows ({len(no_fills)}) — min m1 price distribution"):
            fig_nofill = px.histogram(
                no_fills, x="fill_price", nbins=30,
                color_discrete_sequence=["#6b7280"],
                labels={"fill_price": "Closest approach to buy threshold (min m1 price)"},
            )
            fig_nofill.add_vline(
                x=buy, line_dash="dash", line_color="#f59e0b",
                annotation_text=f"buy={buy:.2f}", annotation_position="top right",
            )
            fig_nofill.update_layout(margin=dict(t=20, b=0), height=250)
            st.plotly_chart(fig_nofill, width="stretch")
