"""Tab 1 — Asset Overview: outcome breakdown, fill rate, edge, profit table."""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from helpers import SESSIONS_PER_DAY, OUTCOME_COLORS


def render(
    results: pd.DataFrame,
    profit: pd.DataFrame,
    buy: float,
    sell: float,
    capital: float,
    position_pct: float,
    spread_cost: float,
) -> None:
    st.subheader(f"Results at  buy = {buy:.2f}   sell = {sell:.2f}")

    # Top-level metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Sessions", f"{int(results['Sessions'].sum()):,}")
    m2.metric("Avg Fill Rate", f"{results['Fill Rate'].mean():.1%}")
    m3.metric("Avg Sell Hit Rate", f"{results['Sell Hit Rate'].mean():.1%}")
    m4.metric("Total Fills", f"{int(results['Fills'].sum()):,}")
    net_day_total = profit["$/Day"].sum() if not profit.empty else 0.0
    m5.metric("Est. $/Day (net, linear)", f"${net_day_total:+.2f}")

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
    st.plotly_chart(fig_stack, width="stretch")

    left, right = st.columns(2)

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
        st.plotly_chart(fig_fr, width="stretch")

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
        st.plotly_chart(fig_edge, width="stretch")

    if not profit.empty:
        st.subheader("Profit estimates")
        position_usdc = capital * position_pct
        st.caption(
            f"Capital: ${capital:,.0f} | Fractional position: {position_pct:.0%} of current capital | "
            f"Spread: {spread_cost:.3f}/crossing | {SESSIONS_PER_DAY} windows/day per asset. "
            f"Returns compounded — capital never goes negative with fractional sizing."
        )
        disp = profit.copy()
        disp["$/Session (net)"] = disp["$/Session (net)"].map("${:+.4f}".format)
        for col in ["$/Day", "$/Week", "$/Month"]:
            disp[col] = disp[col].map("${:+.2f}".format)
        st.dataframe(disp.set_index("Asset"), width="stretch")

    with st.expander("Raw simulation numbers"):
        disp2 = results.copy()
        disp2["Fill Rate"] = disp2["Fill Rate"].map("{:.2%}".format)
        disp2["Sell Hit Rate"] = disp2["Sell Hit Rate"].map("{:.2%}".format)
        disp2["Edge/Session"] = disp2["Edge/Session"].map("{:.6f}".format)
        st.dataframe(disp2.set_index("Asset"), width="stretch")
