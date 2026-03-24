#!/usr/bin/env python3
"""
Skeptic Research Dashboard

Run with:
    streamlit run scripts/dashboard/__main__.py

Requires data from the price collector or paper trading:
    python scripts/collect_prices.py     # preferred — run for hours/days
    python scripts/trade.py --dry-run    # alternative — paper trading
"""
import os
import sys
import time

# Ensure repo root is on the path for skeptic.* imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# Ensure dashboard/ is on the path for sibling module imports
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import streamlit as st

from skeptic import config
from skeptic.research import analyzer

import tab_overview
import tab_window
import tab_heatmap
import tab_timeseries
import tab_live
from collector import get_collector
from data_loader import get_sessions
from helpers import compute_profit_table

st.set_page_config(
    page_title="Skeptic Dashboard",
    page_icon="📈",
    layout="wide",
)

MIN_WINDOW_POINTS = 280  # discard windows with fewer data points (incomplete)


def main() -> None:
    st.title("📈 Skeptic Research Dashboard")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Data Source")
        source_label = st.radio(
            "Load from",
            ["Price CSVs", "Sessions DB"],
            help="Price CSVs (collect_prices.py) give per-second data; DB uses paper/live session snapshots.",
        )
        source = "prices" if source_label == "Price CSVs" else "db"

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
        sell = st.slider("Sell threshold", 0.45, 0.96, sell_default, 0.01, format="%.2f")

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
        for asset, sessions in get_sessions(source).items()
    }
    total = sum(len(v) for v in all_sessions.values())

    if total == 0:
        if source == "prices":
            st.warning(
                "**No price CSV data found.**\n\n"
                "Run the price collector first, then reload this dashboard:\n\n"
                "```\npython scripts/collect_prices.py\n```"
            )
        else:
            st.warning(
                "**No sessions in DB yet.**\n\n"
                "Run paper trading first, then reload:\n\n"
                "```\npython scripts/trade.py --dry-run\n```"
            )
        return

    if not selected_assets:
        st.info("Select at least one asset in the sidebar.")
        return

    # ── Run simulation for selected assets ────────────────────────────────────
    sim_rows = []
    for asset in selected_assets:
        sessions = all_sessions.get(asset, [])
        if not sessions:
            continue
        r = analyzer.simulate(sessions, buy, sell, fill_window=fill_window)
        sim_rows.append({
            "Asset": asset,
            "Sessions": r.n_sessions,
            "Fills": r.n_fills,
            "Fill Rate": r.fill_rate,
            "Sell Hits": r.n_sell_hits,
            "Sell Hit Rate": r.sell_hit_rate,
            "Res Wins": r.n_resolution_wins,
            "Res Losses": r.n_resolution_losses,
            "Edge/Session": r.edge_per_session,
        })

    if not sim_rows:
        st.warning("No sessions found for the selected assets.")
        return

    results = pd.DataFrame(sim_rows)
    profit = compute_profit_table(results, buy, capital, position_pct, spread_cost)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Asset Overview",
        "🔍 Window Detail",
        "🗺️ Threshold Heatmap",
        "📈 Time Series",
        "🔴 Live",
    ])

    with tab1:
        tab_overview.render(results, profit, buy, sell, capital, position_pct, spread_cost)

    with tab2:
        tab_window.render(all_sessions, selected_assets, buy, sell, fill_window)

    with tab3:
        tab_heatmap.render(all_sessions, selected_assets, buy, sell)

    with tab4:
        tab_timeseries.render(all_sessions, selected_assets, buy, sell, capital, position_pct, spread_cost, fill_window)

    with tab5:
        collector = get_collector()
        tab_live.render(collector, buy, sell, capital, position_pct)
        # Auto-refresh every second while collecting
        if collector.running:
            time.sleep(1)
            st.rerun()


if __name__ == "__main__":
    main()
