"""Tab 3 — Threshold Heatmap: grid search over buy×sell edge landscape."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import plotly.graph_objects as go

from skeptic.research import analyzer
from skeptic.research.fetcher import HistoricalSession


def render(
    all_sessions: dict[str, list[HistoricalSession]],
    selected_assets: list[str],
    buy: float,
    sell: float,
) -> None:
    st.subheader("Grid search — edge per session across all threshold pairs")
    st.caption(
        "Runs `optimize_thresholds()` across the full buy × sell grid. "
        "Can take 10–30s for large datasets."
    )

    hm_asset = st.selectbox("Asset", selected_assets, key="hm_asset")
    n_asset_sessions = len(all_sessions.get(hm_asset, []))
    st.caption(f"{n_asset_sessions} sessions available for {hm_asset}")

    gc1, gc2, gc3, gc4 = st.columns(4)
    with gc1:
        hm_buy_min = st.number_input("Buy min", value=0.10, step=0.01, format="%.2f", key="hm_buy_min")
        hm_buy_max = st.number_input("Buy max", value=0.55, step=0.01, format="%.2f", key="hm_buy_max")
    with gc2:
        hm_sell_min = st.number_input("Sell min", value=0.45, step=0.01, format="%.2f", key="hm_sell_min")
        hm_sell_max = st.number_input("Sell max", value=0.95, step=0.01, format="%.2f", key="hm_sell_max")
    with gc3:
        hm_step = st.selectbox("Step size", [0.01, 0.02, 0.05], index=0, key="hm_step")
    with gc4:
        hm_fill_window = st.slider(
            "Fill window (s)", 15, 120, 60, 15, key="hm_fill_window",
            help="Seconds to watch for a fill. Move slider to re-run and compare heatmaps.",
        )

    if st.button("Run grid search", type="primary", key="run_grid"):
        if n_asset_sessions == 0:
            st.warning(f"No sessions for {hm_asset}.")
        else:
            with st.spinner(f"Optimizing thresholds for {hm_asset} (fill window={hm_fill_window}s)…"):
                grid_df = analyzer.optimize_thresholds(
                    all_sessions[hm_asset],
                    buy_range=(hm_buy_min, hm_buy_max),
                    sell_range=(hm_sell_min, hm_sell_max),
                    step=hm_step,
                    fill_window=hm_fill_window,
                )
            st.session_state["grid_df"] = grid_df
            st.session_state["grid_asset"] = hm_asset
            st.session_state["grid_fill_window"] = hm_fill_window

    grid_df = st.session_state.get("grid_df", None)
    grid_asset = st.session_state.get("grid_asset", "")
    grid_fill_window = st.session_state.get("grid_fill_window", 60)

    if grid_df is not None and not grid_df.empty and grid_asset == hm_asset:
        pivot = grid_df.pivot(index="buy", columns="sell", values="edge_per_session")
        x_labels = [f"{v:.2f}" for v in pivot.columns]
        y_labels = [f"{v:.2f}" for v in pivot.index]

        fig_hm = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=x_labels,
            y=y_labels,
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="Edge/Session"),
            hovertemplate="Buy: %{y}  Sell: %{x}<br>Edge: %{z:.6f}<extra></extra>",
        ))

        buy_lbl = f"{buy:.2f}"
        sell_lbl = f"{sell:.2f}"
        if buy_lbl in y_labels and sell_lbl in x_labels:
            fig_hm.add_trace(go.Scatter(
                x=[sell_lbl], y=[buy_lbl],
                mode="markers",
                name="Current thresholds",
                marker=dict(symbol="star", size=16, color="white",
                            line=dict(color="black", width=1)),
                hovertemplate=f"Current: buy={buy:.2f} sell={sell:.2f}<extra></extra>",
            ))

        best = grid_df.iloc[0]
        fig_hm.update_layout(
            xaxis_title="Sell threshold",
            yaxis_title="Buy threshold",
            height=540,
            margin=dict(t=20, b=70),
            annotations=[dict(
                text=(
                    f"Best: buy={best['buy']:.2f}  sell={best['sell']:.2f}  "
                    f"fill window={grid_fill_window}s  "
                    f"edge={best['edge_per_session']:.4f}  "
                    f"fill_rate={best['fill_rate']:.1%}  "
                    f"sell_hit_rate={best['sell_hit_rate']:.1%}"
                ),
                xref="paper", yref="paper", x=0, y=-0.10,
                showarrow=False, font=dict(size=12),
            )],
        )
        st.plotly_chart(fig_hm, width="stretch")

        st.subheader("Top 20 threshold pairs")
        top = grid_df.head(20)[
            ["buy", "sell", "n_sessions", "n_fills", "fill_rate",
             "sell_hit_rate", "n_res_wins", "n_res_losses", "edge_per_session"]
        ].copy()
        top["fill_rate"] = top["fill_rate"].map("{:.2%}".format)
        top["sell_hit_rate"] = top["sell_hit_rate"].map("{:.2%}".format)
        top["edge_per_session"] = top["edge_per_session"].map("{:.6f}".format)
        st.dataframe(top, width="stretch", hide_index=True)
    else:
        st.info("Select an asset and click **Run grid search** to generate the heatmap.")
