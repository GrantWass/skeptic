#!/usr/bin/env python3
"""
Skeptic Research Dashboard

Run with:
    streamlit run scripts/dashboard/__main__.py

Requires data from the price collector:
    python scripts/collect_prices.py
"""
import os
import sys

# Ensure repo root is on the path for skeptic.* imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# Ensure dashboard/ is on the path for sibling module imports
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

from skeptic import config

import tab_heatmap
import tab_timeseries
import tab_high_buy
import tab_timing
import tab_live
from data_loader import get_sessions

st.set_page_config(
    page_title="Skeptic Research Dashboard",
    page_icon="📈",
    layout="wide",
)

MIN_WINDOW_POINTS = 280  # discard windows with fewer data points (incomplete)


def main() -> None:
    st.title("📈 Skeptic Research Dashboard")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Assets")
        cols = st.columns(2)
        selected_assets = [
            asset
            for i, asset in enumerate(config.ASSETS)
            if cols[i % 2].checkbox(asset, value=True, key=f"asset_{asset}")
        ]

        st.header("Thresholds")
        buy_default = float(config.BUY_PRICE) if config.BUY_PRICE is not None else 0.35
        sell_default = float(config.SELL_PRICE) if config.SELL_PRICE is not None else 0.65
        buy = st.slider("Buy threshold", 0.10, 0.50, buy_default, 0.01, format="%.2f")
        sell = st.slider("Sell threshold", 0.45, 0.99, sell_default, 0.01, format="%.2f")

        st.header("Fill Window")
        fill_window = st.slider(
            "Fill window (seconds)", 10, 90, 60, 10,
            help="How long after window open to watch for a fill. Default is 60s.",
        )

        st.header("Capital")
        capital = st.number_input("Starting capital ($)", value=500.0, step=50.0, min_value=10.0)
        position_pct = st.slider(
            "Position size", 1, 20, int(float(config.POSITION_SIZE_PCT) * 100), 1,
            format="%d%%", help="Fraction of capital deployed per window"
        ) / 100
        spread_cost = st.number_input(
            "Spread cost / crossing", value=0.002, step=0.001, format="%.3f",
            help="Half-spread paid per fill. Two crossings per round-trip."
        )

    # ── Load sessions ─────────────────────────────────────────────────────────
    all_sessions = {
        asset: [s for s in sessions if len(s.up_trades_all) >= MIN_WINDOW_POINTS]
        for asset, sessions in get_sessions().items()
    }
    total = sum(len(v) for v in all_sessions.values())

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Threshold Heatmap",
        "📈 Time Series",
        "🎯 High Buy",
        "⏱️ Timing",
        "🤖 Live Bots",
    ])

    if total == 0:
        with tab1, tab2, tab3:
            st.warning(
                "**No price CSV data found.**\n\n"
                "Run the price collector first, then reload this dashboard:\n\n"
                "```\npython scripts/collect_prices.py\n```"
            )
    elif not selected_assets:
        with tab1, tab2, tab3:
            st.info("Select at least one asset in the sidebar.")
    else:
        with tab1:
            tab_heatmap.render(all_sessions, selected_assets, buy, sell)

        with tab2:
            tab_timeseries.render(all_sessions, selected_assets, buy, sell, capital, position_pct, spread_cost, fill_window)

        with tab3:
            tab_high_buy.render(all_sessions, selected_assets, capital, position_pct)

        with tab4:
            tab_timing.render(all_sessions, selected_assets)

    with tab5:
        tab_live.render()


if __name__ == "__main__":
    main()
