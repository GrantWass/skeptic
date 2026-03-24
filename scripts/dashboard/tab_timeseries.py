"""Tab 4 — Time Series: per-window price charts for a selected asset."""
import streamlit as st

from helpers import classify_window, make_window_chart
from skeptic.research.fetcher import HistoricalSession


def render(
    all_sessions: dict[str, list[HistoricalSession]],
    selected_assets: list[str],
    buy: float,
    sell: float,
    capital: float,
    position_pct: float,
    spread_cost: float,
    fill_window: int = 60,
) -> None:
    ts_asset = st.selectbox("Asset", selected_assets, key="ts_asset")
    asset_sessions = sorted(
        all_sessions.get(ts_asset, []),
        key=lambda s: s.window_start_ts,
    )
    recent = asset_sessions[::-1]  # most recent first

    if not recent:
        st.info(f"No sessions for {ts_asset}.")
        return

    position_usdc = capital * position_pct
    classified = [classify_window(s, buy, sell, position_usdc, spread_cost, fill_window) for s in recent]

    n_sell_hit = sum(1 for o, _ in classified if o == "Sell Hit")
    n_res_win  = sum(1 for o, _ in classified if o == "Res Win")
    n_res_loss = sum(1 for o, _ in classified if o == "Res Loss")
    n_no_fill  = sum(1 for o, _ in classified if o == "No Fill")
    total_pnl  = sum(p for _, p in classified)

    sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
    sc1.metric("Windows", len(recent))
    sc2.metric("Sell Hit ✅", n_sell_hit)
    sc3.metric("Res Win ✅", n_res_win)
    sc4.metric("Res Loss ❌", n_res_loss)
    sc5.metric("No Fill ⬜", n_no_fill)
    sc6.metric(f"Est. P&L ({len(recent)} windows)", f"${total_pnl:+.2f}")

    st.caption(
        f"buy={buy:.2f}  sell={sell:.2f}  "
        f"position=${position_usdc:.2f}  spread={spread_cost:.3f}/crossing"
    )
    st.divider()

    col_a, col_b = st.columns(2)
    for i, (s, (outcome, profit)) in enumerate(zip(recent, classified)):
        col = col_a if i % 2 == 0 else col_b
        with col:
            fig = make_window_chart(s, buy, sell, outcome, profit)
            st.plotly_chart(fig, width="stretch")
