#!/usr/bin/env python3
"""
Skeptic Research Dashboard

Run with:
    streamlit run scripts/dashboard.py

Requires data from the price collector or paper trading:
    python scripts/collect_prices.py     # preferred — run for hours/days
    python scripts/trade.py --dry-run    # alternative — paper trading
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from skeptic import config
from skeptic.research import analyzer, fetcher
from skeptic.research.fetcher import HistoricalSession

st.set_page_config(
    page_title="Skeptic Dashboard",
    page_icon="📈",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────────

SESSIONS_PER_DAY = 288  # 24h × 60min / 5min per window
OUTCOME_COLORS = {
    "No Fill":  "#6b7280",
    "Sell Hit": "#22c55e",
    "Res Win":  "#3b82f6",
    "Res Loss": "#ef4444",
}

# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading price data…")
def _load_price_sessions(assets_key: str) -> dict[str, list[HistoricalSession]]:
    return fetcher.load_from_price_files(assets_key.split(","))


@st.cache_resource(show_spinner="Loading DB sessions…")
def _load_db_sessions(assets_key: str) -> dict[str, list[HistoricalSession]]:
    assets = assets_key.split(",")
    result: dict[str, list[HistoricalSession]] = {a: [] for a in assets}

    if not os.path.exists(config.DB_PATH):
        return result

    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        for asset in assets:
            rows = conn.execute(
                """
                SELECT s.*,
                    ps1_up.price AS up_m1,
                    ps1_dn.price AS dn_m1
                FROM sessions s
                LEFT JOIN price_snapshots ps1_up
                    ON ps1_up.session_id = s.session_id
                    AND ps1_up.outcome = 'UP' AND ps1_up.minute_mark = 1
                LEFT JOIN price_snapshots ps1_dn
                    ON ps1_dn.session_id = s.session_id
                    AND ps1_dn.outcome = 'DOWN' AND ps1_dn.minute_mark = 1
                WHERE s.asset = ?
                ORDER BY s.window_start_ts
                """,
                (asset,),
            ).fetchall()

            for row in rows:
                hs = HistoricalSession(
                    asset=row["asset"],
                    condition_id=row["condition_id"],
                    window_start_ts=row["window_start_ts"],
                    up_token_id="",
                    down_token_id="",
                )
                # DB only has minute-1 snapshots — use T+30 as a proxy timestamp
                mid_ts = row["window_start_ts"] + 30
                up_m1 = row["up_m1"] or row["up_price_m1"]
                dn_m1 = row["dn_m1"] or row["down_price_m1"]
                if up_m1:
                    hs.up_trades_m1 = [(mid_ts, float(up_m1))]
                    hs.up_trades_all = [(mid_ts, float(up_m1))]
                if dn_m1:
                    hs.down_trades_m1 = [(mid_ts, float(dn_m1))]
                    hs.down_trades_all = [(mid_ts, float(dn_m1))]
                res = row["resolution_price"]
                if res is not None:
                    hs.up_resolution = float(res)
                    hs.down_resolution = 1.0 - float(res)
                result[asset].append(hs)
    finally:
        conn.close()

    return result


def _get_sessions(source: str) -> dict[str, list[HistoricalSession]]:
    key = ",".join(config.ASSETS)
    if source == "prices":
        return _load_price_sessions(key)
    return _load_db_sessions(key)


# ── Per-session scatter data ────────────────────────────────────────────────────

def build_window_rows(
    sessions: list[HistoricalSession],
    buy: float,
    sell: float,
) -> pd.DataFrame:
    """One row per session: the fill price, max price after fill, and outcome."""
    rows = []
    for s in sessions:
        up_fill = s.up_min_m1 is not None and s.up_min_m1 <= buy
        down_fill = s.down_min_m1 is not None and s.down_min_m1 <= buy

        if not up_fill and not down_fill:
            rows.append({
                "fill_price": min(
                    s.up_min_m1 if s.up_min_m1 is not None else 1.0,
                    s.down_min_m1 if s.down_min_m1 is not None else 1.0,
                ),
                "max_after": None,
                "outcome": "No Fill",
            })
            continue

        # Determine which side filled first
        if up_fill and down_fill:
            up_ts = s.up_first_fill_ts(buy)
            dn_ts = s.down_first_fill_ts(buy)
            use_up = up_ts is not None and (dn_ts is None or up_ts <= dn_ts)
        else:
            use_up = up_fill

        if use_up:
            fill_price = s.up_min_m1
            max_after = s.up_max_after_fill(buy)
            sell_hit = (max_after or 0.0) >= sell
            res_win = (s.up_resolution or 0.0) >= 0.9
        else:
            fill_price = s.down_min_m1
            max_after = s.down_max_after_fill(buy)
            sell_hit = (max_after or 0.0) >= sell
            res_win = (s.down_resolution or 0.0) >= 0.9

        if sell_hit:
            outcome = "Sell Hit"
        elif res_win:
            outcome = "Res Win"
        else:
            outcome = "Res Loss"

        rows.append({
            "fill_price": fill_price,
            "max_after": max_after,
            "outcome": outcome,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Profit calculation ─────────────────────────────────────────────────────────

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
            "$/Session (gross)": gross,
            "Spread/Session": spread,
            "$/Session (net)": net,
            "$/Day (net)": net * SESSIONS_PER_DAY,
            "$/Week (net)": net * SESSIONS_PER_DAY * 7,
            "$/Month (net)": net * SESSIONS_PER_DAY * 30,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Main ───────────────────────────────────────────────────────────────────────

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
        buy = st.slider("Buy threshold", 0.20, 0.49, buy_default, 0.01, format="%.2f")
        sell = st.slider("Sell threshold", 0.51, 0.90, sell_default, 0.01, format="%.2f")

        st.header("Capital")
        capital = st.number_input("Starting capital ($)", value=500.0, step=50.0, min_value=10.0)
        position_pct = st.slider(
            "Position size", 0.01, 0.20, float(config.POSITION_SIZE_PCT), 0.01,
            format="%.0f%%", help="Fraction of capital deployed per window"
        )
        spread_cost = st.number_input(
            "Spread cost / crossing", value=0.002, step=0.001, format="%.3f",
            help="Half-spread paid per fill. Two crossings per round-trip."
        )

    # ── Load sessions ─────────────────────────────────────────────────────────
    all_sessions = _get_sessions(source)
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
        r = analyzer.simulate(sessions, buy, sell)
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
    tab1, tab2, tab3 = st.tabs(["📊 Asset Overview", "🔍 Window Detail", "🗺️ Threshold Heatmap"])

    # ════════════════════════════════════════════════════════════════════════════
    # Tab 1 — Asset Overview
    # ════════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader(f"Results at  buy = {buy:.2f}   sell = {sell:.2f}")

        # Top-level metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Sessions", f"{int(results['Sessions'].sum()):,}")
        m2.metric("Avg Fill Rate", f"{results['Fill Rate'].mean():.1%}")
        m3.metric("Avg Sell Hit Rate", f"{results['Sell Hit Rate'].mean():.1%}")
        m4.metric("Total Fills", f"{int(results['Fills'].sum()):,}")
        net_day_total = profit["$/Day (net)"].sum() if not profit.empty else 0.0
        m5.metric("Est. $/Day (net)", f"${net_day_total:+.2f}")

        st.divider()

        # Outcome breakdown stacked bar
        st.subheader("Outcome breakdown per window")
        bd_rows = []
        for _, r in results.iterrows():
            n = r["Sessions"]
            bd_rows.append({
                "Asset": r["Asset"],
                "No Fill":  n - r["Fills"],
                "Sell Hit": r["Sell Hits"],
                "Res Win":  r["Res Wins"],
                "Res Loss": r["Res Losses"],
            })
        bd = pd.DataFrame(bd_rows)

        fig_stack = go.Figure()
        for outcome, color in OUTCOME_COLORS.items():
            fig_stack.add_trace(go.Bar(
                name=outcome, x=bd["Asset"], y=bd[outcome],
                marker_color=color,
                text=bd[outcome], textposition="inside",
            ))
        fig_stack.update_layout(
            barmode="stack",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=0), height=320,
            yaxis_title="Windows",
        )
        st.plotly_chart(fig_stack, use_container_width=True)

        left, right = st.columns(2)

        # Fill rate bar
        with left:
            st.subheader("Fill rate per asset")
            sorted_fr = results.sort_values("Fill Rate", ascending=True)
            fig_fr = px.bar(
                sorted_fr, x="Fill Rate", y="Asset", orientation="h",
                text=sorted_fr["Fill Rate"].map("{:.1%}".format),
                color="Fill Rate", color_continuous_scale="Blues",
            )
            fig_fr.update_layout(
                showlegend=False, coloraxis_showscale=False,
                xaxis=dict(tickformat=".0%"),
                margin=dict(t=0, b=0), height=280,
            )
            st.plotly_chart(fig_fr, use_container_width=True)

        # Edge per session bar
        with right:
            st.subheader("Edge per session")
            sorted_edge = results.sort_values("Edge/Session", ascending=True)
            fig_edge = px.bar(
                sorted_edge, x="Edge/Session", y="Asset", orientation="h",
                text=sorted_edge["Edge/Session"].map("{:.4f}".format),
                color="Edge/Session",
                color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
            )
            fig_edge.update_layout(
                showlegend=False, coloraxis_showscale=False,
                margin=dict(t=0, b=0), height=280,
            )
            st.plotly_chart(fig_edge, use_container_width=True)

        # Profit table
        if not profit.empty:
            st.subheader("Profit estimates")
            position_usdc = capital * position_pct
            st.caption(
                f"Capital: ${capital:,.0f} | Position: {position_pct:.0%} "
                f"(${position_usdc:.2f}/trade) | "
                f"Spread: {spread_cost:.3f}/crossing | {SESSIONS_PER_DAY} windows/day per asset"
            )
            disp = profit.copy()
            for col in ["$/Session (gross)", "$/Session (net)"]:
                disp[col] = disp[col].map("${:+.4f}".format)
            disp["Spread/Session"] = disp["Spread/Session"].map("${:.4f}".format)
            for col in ["$/Day (net)", "$/Week (net)", "$/Month (net)"]:
                disp[col] = disp[col].map("${:+.2f}".format)
            st.dataframe(disp.set_index("Asset"), use_container_width=True)

        with st.expander("Raw simulation numbers"):
            disp2 = results.copy()
            disp2["Fill Rate"] = disp2["Fill Rate"].map("{:.2%}".format)
            disp2["Sell Hit Rate"] = disp2["Sell Hit Rate"].map("{:.2%}".format)
            disp2["Edge/Session"] = disp2["Edge/Session"].map("{:.6f}".format)
            st.dataframe(disp2.set_index("Asset"), use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════════
    # Tab 2 — Window Detail scatter
    # ════════════════════════════════════════════════════════════════════════════
    with tab2:
        detail_asset = st.selectbox("Asset", selected_assets, key="detail_asset")
        sessions = all_sessions.get(detail_asset, [])

        if not sessions:
            st.info(f"No sessions loaded for {detail_asset}.")
        else:
            df_w = build_window_rows(sessions, buy, sell)

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

                # Buy threshold vertical line
                fig_scatter.add_vline(
                    x=buy, line_dash="dash", line_color="#f59e0b", line_width=2,
                    annotation_text=f"buy={buy:.2f}",
                    annotation_position="top right",
                )
                # Sell threshold horizontal line
                fig_scatter.add_hline(
                    y=sell, line_dash="dash", line_color="#a78bfa", line_width=2,
                    annotation_text=f"sell={sell:.2f}",
                    annotation_position="right",
                )

                fig_scatter.update_layout(
                    xaxis_title="Fill price (min m1 price)",
                    yaxis_title="Max price reached after fill",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=480,
                    margin=dict(t=40, b=0),
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            # No-fill distribution
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
                    st.plotly_chart(fig_nofill, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════════
    # Tab 3 — Threshold Heatmap
    # ════════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Grid search — edge per session across all threshold pairs")
        st.caption(
            "Runs `optimize_thresholds()` across the full buy × sell grid. "
            "Can take 10–30s for large datasets."
        )

        hm_asset = st.selectbox("Asset", selected_assets, key="hm_asset")
        n_asset_sessions = len(all_sessions.get(hm_asset, []))
        st.caption(f"{n_asset_sessions} sessions available for {hm_asset}")

        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            hm_buy_min = st.number_input("Buy min", value=0.20, step=0.01, format="%.2f", key="hm_buy_min")
            hm_buy_max = st.number_input("Buy max", value=0.49, step=0.01, format="%.2f", key="hm_buy_max")
        with gc2:
            hm_sell_min = st.number_input("Sell min", value=0.51, step=0.01, format="%.2f", key="hm_sell_min")
            hm_sell_max = st.number_input("Sell max", value=0.90, step=0.01, format="%.2f", key="hm_sell_max")
        with gc3:
            hm_step = st.selectbox("Step size", [0.01, 0.02, 0.05], index=0, key="hm_step")

        if st.button("Run grid search", type="primary", key="run_grid"):
            if n_asset_sessions == 0:
                st.warning(f"No sessions for {hm_asset}.")
            else:
                with st.spinner(f"Optimizing thresholds for {hm_asset}…"):
                    grid_df = analyzer.optimize_thresholds(
                        all_sessions[hm_asset],
                        buy_range=(hm_buy_min, hm_buy_max),
                        sell_range=(hm_sell_min, hm_sell_max),
                        step=hm_step,
                    )
                st.session_state["grid_df"] = grid_df
                st.session_state["grid_asset"] = hm_asset

        grid_df: pd.DataFrame = st.session_state.get("grid_df", pd.DataFrame())
        grid_asset: str = st.session_state.get("grid_asset", "")

        if not grid_df.empty and grid_asset == hm_asset:
            # Heatmap: x = sell threshold, y = buy threshold, z = edge
            pivot = grid_df.pivot(index="buy", columns="sell", values="edge_per_session")

            x_labels = [f"{v:.2f}" for v in pivot.columns]
            y_labels = [f"{v:.2f}" for v in pivot.index]

            fig_hm = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=x_labels,
                y=y_labels,
                colorscale="RdYlGn",
                colorbar=dict(title="Edge/Session"),
                hovertemplate="Buy: %{y}  Sell: %{x}<br>Edge: %{z:.6f}<extra></extra>",
            ))

            # Star at currently selected sidebar thresholds (if in range)
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
                margin=dict(t=20, b=60),
                annotations=[dict(
                    text=(
                        f"Best: buy={best['buy']:.2f}  sell={best['sell']:.2f}  "
                        f"edge={best['edge_per_session']:.4f}  "
                        f"fill_rate={best['fill_rate']:.1%}  "
                        f"sell_hit_rate={best['sell_hit_rate']:.1%}"
                    ),
                    xref="paper", yref="paper", x=0, y=-0.09,
                    showarrow=False, font=dict(size=12),
                )],
            )
            st.plotly_chart(fig_hm, use_container_width=True)

            st.subheader("Top 20 threshold pairs")
            top = grid_df.head(20)[
                ["buy", "sell", "n_sessions", "n_fills", "fill_rate",
                 "sell_hit_rate", "n_res_wins", "n_res_losses", "edge_per_session"]
            ].copy()
            top["fill_rate"] = top["fill_rate"].map("{:.2%}".format)
            top["sell_hit_rate"] = top["sell_hit_rate"].map("{:.2%}".format)
            top["edge_per_session"] = top["edge_per_session"].map("{:.6f}".format)
            st.dataframe(top, use_container_width=True, hide_index=True)
        else:
            st.info("Select an asset and click **Run grid search** to generate the heatmap.")


if __name__ == "__main__":
    main()
