"""Tab — Predicted vs Actual: surface where live performance diverges from simulation."""
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[2]))

LIVE_DIR    = Path("data/live")
REPORTS_DIR = Path("data/reports")

ASSETS = ["BTC", "DOGE", "ETH", "SOL", "BNB"]

ASSET_COLORS = {
    "BTC":  "#f59e0b",
    "DOGE": "#8b5cf6",
    "ETH":  "#3b82f6",
    "SOL":  "#06b6d4",
    "BNB":  "#f97316",
}

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _load_momentum_actual() -> pd.DataFrame:
    dfs = []
    for asset in ASSETS:
        path = LIVE_DIR / f"trades_mom_{asset.lower()}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["asset"] = asset
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df[df["status"].isin(["won", "lost", "fok_killed"])].copy()


@st.cache_data(ttl=300)
def _load_threshold_edge() -> pd.DataFrame:
    path = REPORTS_DIR / "threshold_edge.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["asset"] = df["asset"].str.upper()
    df["date"] = pd.to_datetime(df["window_ts"], unit="s", utc=True)
    return df


@st.cache_data(ttl=60)
def _load_model_actual() -> pd.DataFrame:
    dfs = []
    for asset in ASSETS:
        path = LIVE_DIR / f"trades_model_mom_{asset.lower()}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["asset"] = asset
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df["is_live"] = df["order_id"] != "DRY_RUN"
    return df[df["status"].isin(["won", "lost", "fok_killed"])].copy()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _expected_pnl(fill_price: float, fill_usdc: float, win_rate: float | None) -> float:
    """E[PnL] per trade given a fill price, stake, and win-rate baseline."""
    if fill_price <= 0 or win_rate is None:
        return 0.0
    return fill_usdc * (win_rate - fill_price) / fill_price


def _pct(val: float) -> str:
    return f"{val:.1%}"


def _fmt(val: float, decimals: int = 3) -> str:
    return f"{val:.{decimals}f}"


def _delta_color(actual: float, expected: float, higher_is_better: bool = True) -> str:
    if actual > expected:
        return "normal" if higher_is_better else "inverse"
    return "inverse" if higher_is_better else "normal"


# ── Rendering ─────────────────────────────────────────────────────────────────

def render() -> None:
    st.header("Predicted vs Actual")

    mom_actual = _load_momentum_actual()
    te         = _load_threshold_edge()
    mdl_actual = _load_model_actual()

    if mom_actual.empty and mdl_actual.empty:
        st.warning("No trade data found.")
        return

    tab_mom, tab_mdl = st.tabs(["Momentum Strategy", "Model Strategy"])

    # ── MOMENTUM TAB ──────────────────────────────────────────────────────────
    with tab_mom:
        _render_momentum(mom_actual, te)

    # ── MODEL TAB ─────────────────────────────────────────────────────────────
    with tab_mdl:
        _render_model(mdl_actual)


# ── Momentum ─────────────────────────────────────────────────────────────────

def _render_momentum(actual: pd.DataFrame, sim: pd.DataFrame) -> None:
    if actual.empty:
        st.info("No resolved momentum trades yet.")
        return

    resolved_actual = actual[actual["status"].isin(["won", "lost"])].copy()

    # ── Window-level join ────────────────────────────────────────────────────
    # Only compare on windows present in both datasets for each asset.
    # Actual: window_start_ts  |  Sim: window_ts (filtered to SIM_SIGMA=0.5)
    matched_rows = []
    for asset in ASSETS:
        act = resolved_actual[resolved_actual["asset"] == asset]
        if act.empty:
            continue
        if not sim.empty and "sigma_entry" in act.columns:
            # Match each trade to sim rows for its own sigma_entry value.
            # Trades with different sigma_entry values (e.g. BTC ran both 0.25 and 0.5)
            # are each matched against the correct sim group.
            kept = []
            for sigma_val, sigma_group in act.groupby("sigma_entry"):
                sim_windows = set(sim.loc[
                    (sim["asset"] == asset) & (sim["sigma"] == sigma_val), "window_ts"
                ])
                kept.append(sigma_group[sigma_group["window_start_ts"].isin(sim_windows)])
            act = pd.concat(kept) if kept else pd.DataFrame()
        if act.empty:
            continue
        matched_rows.append(act)

    if not matched_rows:
        st.info("No windows with both simulated and real data.")
        return

    matched = pd.concat(matched_rows, ignore_index=True)

    # Sim stats restricted to the same matched windows
    rows = []
    for asset in ASSETS:
        act = matched[matched["asset"] == asset]
        if act.empty:
            continue

        matched_windows = set(act["window_start_ts"])
        if not sim.empty and "sigma_entry" in act.columns:
            # For each sigma_entry group, match sim rows by that sigma value
            sim_parts = []
            for sigma_val, sigma_group in act.groupby("sigma_entry"):
                group_windows = set(sigma_group["window_start_ts"])
                sim_parts.append(sim[
                    (sim["asset"] == asset)
                    & (sim["sigma"] == sigma_val)
                    & (sim["window_ts"].isin(group_windows))
                ])
            sim_asset = pd.concat(sim_parts) if sim_parts else pd.DataFrame()
        else:
            sim_asset = pd.DataFrame()

        wr      = act["status"].eq("won").mean()
        avg_fp  = act["fill_price"].mean()
        avg_sl  = act["slippage"].mean() if "slippage" in act.columns else float("nan")
        avg_et  = act["elapsed_second"].mean()
        tot_pnl = act["pnl_usdc"].sum()

        if not sim_asset.empty:
            s_wr    = sim_asset["won"].mean()
            s_price = sim_asset["pm_price_at_trigger"].mean()
            s_slip  = (sim_asset["pm_price"] - sim_asset["pm_price_at_trigger"]).mean()
            s_et    = sim_asset["trigger_second"].mean()
            exp_pnl = float(act.apply(
                lambda r: _expected_pnl(r["fill_price"], r["fill_usdc"], s_wr), axis=1
            ).sum())
        else:
            s_wr = s_price = s_slip = s_et = exp_pnl = None

        rows.append({
            "Asset":             asset,
            "Windows":           len(matched_windows),
            "Actual Trades":     len(act),
            "Actual WR":         wr,
            "Sim WR":            s_wr,
            "WR Gap":            (wr - s_wr)      if s_wr    is not None else None,
            "Actual Avg Price":  avg_fp,
            "Sim Trigger Price": s_price,
            "Price Gap":         (avg_fp - s_price) if s_price is not None else None,
            "Actual Avg Slip":   avg_sl,
            "Sim Avg Slip":      s_slip,
            "Actual Entry (s)":  avg_et,
            "Sim Entry (s)":     s_et,
            "Actual PnL":        tot_pnl,
            "Expected PnL":      exp_pnl,
            "PnL Gap":           (tot_pnl - exp_pnl) if exp_pnl is not None else None,
        })

    summary = pd.DataFrame(rows)

    # ── Comparison table ─────────────────────────────────────────────────────
    st.subheader("Summary")
    st.caption(
        "Only windows present in both threshold_edge.csv and live trades are included. "
        "Expected PnL uses the simulated win rate applied to actual fill prices and stakes."
    )

    def _color_gap(val, higher_is_better=True):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        color = "#22c55e" if (val >= 0) == higher_is_better else "#ef4444"
        return f"color: {color}"

    display = pd.DataFrame({
        "Asset":            summary["Asset"],
        "Windows":          summary["Windows"],
        "Actual WR":        summary["Actual WR"].map(_pct),
        "Sim WR":           summary["Sim WR"].map(lambda v: _pct(v) if v is not None else "—"),
        "WR Gap":           summary["WR Gap"].map(lambda v: f"{v:+.1%}" if v is not None else "—"),
        "Actual Price":     summary["Actual Avg Price"].map(_fmt),
        "Sim Price":        summary["Sim Trigger Price"].map(lambda v: _fmt(v) if v is not None else "—"),
        "Price Gap":        summary["Price Gap"].map(lambda v: f"{v:+.3f}" if v is not None else "—"),
        "Entry (s) actual": summary["Actual Entry (s)"].map(lambda v: f"{v:.0f}s"),
        "Entry (s) sim":    summary["Sim Entry (s)"].map(lambda v: f"{v:.0f}s" if v is not None else "—"),
        "Actual PnL":       summary["Actual PnL"].map(lambda v: f"${v:.2f}"),
        "Expected PnL":     summary["Expected PnL"].map(lambda v: f"${v:.2f}" if v is not None else "—"),
        "PnL Gap":          summary["PnL Gap"].map(lambda v: f"${v:+.2f}" if v is not None else "—"),
    })

    def _style(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for i, row in summary.iterrows():
            styles.at[i, "WR Gap"]   = _color_gap(row["WR Gap"],    higher_is_better=True)
            styles.at[i, "Price Gap"] = _color_gap(row["Price Gap"], higher_is_better=False)
            styles.at[i, "PnL Gap"]  = _color_gap(row["PnL Gap"],   higher_is_better=True)
        return styles

    st.dataframe(display.style.apply(_style, axis=None), use_container_width='stretch', hide_index=True)

    st.divider()

    # ── Win rate chart ───────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Win Rate: Actual vs Simulated")
        fig = go.Figure()
        assets = summary["Asset"].tolist()
        fig.add_bar(
            x=assets,
            y=summary["Actual WR"].tolist(),
            name="Actual",
            marker_color=[ASSET_COLORS[a] for a in assets],
            text=[_pct(v) for v in summary["Actual WR"]],
            textposition="outside",
        )
        if summary["Sim WR"].notna().any():
            fig.add_bar(
                x=assets,
                y=summary["Sim WR"].tolist(),
                name="Simulated",
                marker_color="rgba(107,114,128,0.4)",
                text=[_pct(v) if v is not None else "" for v in summary["Sim WR"]],
                textposition="outside",
            )
        fig.update_layout(
            barmode="group",
            yaxis=dict(tickformat=".0%", range=[0, 1.05]),
            height=320,
            margin=dict(t=10, b=10, l=40, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width='stretch')

    with c2:
        st.subheader("Avg Price: Actual Fill vs Sim Trigger")
        fig2 = go.Figure()
        fig2.add_bar(
            x=assets,
            y=summary["Actual Avg Price"].tolist(),
            name="Actual Fill",
            marker_color=[ASSET_COLORS[a] for a in assets],
            text=[_fmt(v) for v in summary["Actual Avg Price"]],
            textposition="outside",
        )
        if summary["Sim Trigger Price"].notna().any():
            fig2.add_bar(
                x=assets,
                y=summary["Sim Trigger Price"].tolist(),
                name="Sim Trigger",
                marker_color="rgba(107,114,128,0.4)",
                text=[_fmt(v) if v is not None else "" for v in summary["Sim Trigger Price"]],
                textposition="outside",
            )
        fig2.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 0.9]),
            height=320,
            margin=dict(t=10, b=10, l=40, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig2, use_container_width='stretch')

    st.divider()

    # ── Slippage distribution ────────────────────────────────────────────────
    st.subheader("Slippage Distribution: Actual vs Simulated")
    st.caption(
        "Actual slippage = fill_price − pm_ask at trigger.  "
        "Sim slippage = pm_price_at_fill − pm_price_at_trigger in threshold_edge."
    )

    if "slippage" in resolved_actual.columns and not sim.empty:
        fig3 = go.Figure()
        sim_slip_vals = (sim["pm_price"] - sim["pm_price_at_trigger"]).dropna()
        act_slip_vals = resolved_actual["slippage"].dropna()

        fig3.add_trace(go.Histogram(
            x=act_slip_vals, name="Actual",
            opacity=0.7, nbinsx=40,
            marker_color="rgba(59,130,246,0.7)",
            histnorm="percent",
        ))
        fig3.add_trace(go.Histogram(
            x=sim_slip_vals, name="Simulated",
            opacity=0.7, nbinsx=40,
            marker_color="rgba(107,114,128,0.4)",
            histnorm="percent",
        ))
        fig3.add_vline(x=act_slip_vals.mean(), line_dash="dash",
                       line_color="#3b82f6", annotation_text=f"Actual mean {act_slip_vals.mean():.3f}",
                       annotation_position="top right")
        fig3.add_vline(x=sim_slip_vals.mean(), line_dash="dash",
                       line_color="#6b7280", annotation_text=f"Sim mean {sim_slip_vals.mean():.3f}",
                       annotation_position="top left")
        fig3.update_layout(
            barmode="overlay",
            xaxis_title="Slippage (price units)",
            yaxis_title="% of trades",
            height=300,
            margin=dict(t=10, b=40, l=50, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig3, use_container_width='stretch')

    st.divider()

    # ── Cumulative PnL: actual vs expected ──────────────────────────────────
    st.subheader("Cumulative PnL: Actual vs Expected (sim win rate)")

    # Lookup dicts built from the summary table
    sim_wr_by_asset: dict[str, float] = {
        r["Asset"]: r["Sim WR"] for _, r in summary.iterrows() if r["Sim WR"] is not None
    }

    asset_filter = st.multiselect(
        "Assets", ASSETS, default=ASSETS, key="mom_cum_assets"
    )

    filtered = matched[matched["asset"].isin(asset_filter)].sort_values("date")

    fig4 = go.Figure()
    for asset in asset_filter:
        g = filtered[filtered["asset"] == asset].copy()
        if g.empty:
            continue
        s_wr = sim_wr_by_asset.get(asset)
        g["cum_actual"] = g["pnl_usdc"].cumsum()
        if s_wr is not None:
            g["exp_pnl"] = g.apply(
                lambda r: _expected_pnl(r["fill_price"], r["fill_usdc"], s_wr), axis=1
            )
            g["cum_expected"] = g["exp_pnl"].cumsum()
        color = ASSET_COLORS.get(asset, "#888")
        fig4.add_trace(go.Scatter(
            x=g["date"], y=g["cum_actual"],
            name=f"{asset} actual",
            line=dict(color=color, width=2),
        ))
        if s_wr is not None:
            fig4.add_trace(go.Scatter(
                x=g["date"], y=g["cum_expected"],
                name=f"{asset} expected",
                line=dict(color=color, width=1.5, dash="dash"),
                opacity=0.6,
            ))

    fig4.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", line_width=1)
    fig4.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative PnL (USDC)",
        height=380,
        margin=dict(t=10, b=40, l=60, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig4, use_container_width='stretch')

    # ── Rolling win rate ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Rolling Win Rate (50-trade window)")

    fig5 = go.Figure()
    for asset in asset_filter:
        g = matched[matched["asset"] == asset].sort_values("date").copy()
        if len(g) < 10:
            continue
        g["rolling_wr"] = g["status"].eq("won").rolling(50, min_periods=10).mean()
        s_wr = sim_wr_by_asset.get(asset)
        color = ASSET_COLORS.get(asset, "#888")
        fig5.add_trace(go.Scatter(
            x=g["date"], y=g["rolling_wr"],
            name=asset, line=dict(color=color, width=2),
        ))
        if s_wr is not None:
            # dashed horizontal sim baseline
            fig5.add_shape(
                type="line",
                x0=g["date"].min(), x1=g["date"].max(),
                y0=s_wr, y1=s_wr,
                line=dict(color=color, width=1, dash="dot"),
            )

    fig5.update_layout(
        xaxis_title="Date",
        yaxis=dict(tickformat=".0%", title="Win Rate"),
        height=320,
        margin=dict(t=10, b=40, l=60, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08),
    )
    st.caption("Dotted lines = simulated win rate baseline per asset")
    st.plotly_chart(fig5, use_container_width='stretch')


# ── Model ─────────────────────────────────────────────────────────────────────

def _render_model(actual: pd.DataFrame) -> None:
    if actual.empty:
        st.info("No resolved model trades yet.")
        return

    resolved = actual[actual["status"].isin(["won", "lost"])]
    live = resolved[resolved["is_live"]]
    dry  = resolved[~resolved["is_live"]]

    # ── Per-asset summary ────────────────────────────────────────────────────
    st.subheader("Summary: Live vs Dry-Run")

    rows = []
    for asset in ASSETS:
        l = live[live["asset"] == asset]
        d = dry[dry["asset"] == asset]
        rows.append({
            "Asset":        asset,
            "Live Trades":  len(l),
            "Live WR":      l["status"].eq("won").mean() if len(l) > 0 else None,
            "Live PnL":     l["pnl_usdc"].sum() if len(l) > 0 else 0.0,
            "Live Avg Edge": l["edge"].mean() if len(l) > 0 else None,
            "Dry Trades":   len(d),
            "Dry WR":       d["status"].eq("won").mean() if len(d) > 0 else None,
            "Dry PnL":      d["pnl_usdc"].sum() if len(d) > 0 else 0.0,
            "Dry Avg Edge": d["edge"].mean() if len(d) > 0 else None,
        })

    cols = st.columns(len(rows))
    for col, row in zip(cols, rows):
        l_wr = row["Live WR"]
        d_wr = row["Dry WR"]
        wr_delta = (l_wr - d_wr) if (l_wr is not None and d_wr is not None) else None
        with col:
            st.markdown(f"**{row['Asset']}**")
            if row["Live Trades"] > 0:
                st.metric("Live WR", _pct(l_wr),
                          delta=f"{wr_delta:+.1%} vs dry" if wr_delta is not None else None,
                          delta_color="normal" if (wr_delta or 0) >= 0 else "inverse")
                st.metric("Live PnL", f"${row['Live PnL']:.2f}")
            else:
                st.metric("Live WR", "—")
                st.metric("Live PnL", "$0.00")
            if row["Dry Trades"] > 0:
                st.metric("Dry-run WR", _pct(d_wr), help="Would-be trades if model were fully enabled")
                st.metric("Dry-run PnL", f"${row['Dry PnL']:.2f}")

    st.divider()

    # ── Win rate bar: live vs dry ────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Win Rate: Live vs Dry-Run")
        assets_with_data = [r["Asset"] for r in rows if r["Live Trades"] > 0 or r["Dry Trades"] > 0]
        fig = go.Figure()
        live_wrs = [next(r["Live WR"] for r in rows if r["Asset"] == a) or 0 for a in assets_with_data]
        dry_wrs  = [next(r["Dry WR"]  for r in rows if r["Asset"] == a) or 0 for a in assets_with_data]
        fig.add_bar(
            x=assets_with_data, y=live_wrs, name="Live",
            marker_color=[ASSET_COLORS[a] for a in assets_with_data],
            text=[_pct(v) for v in live_wrs], textposition="outside",
        )
        fig.add_bar(
            x=assets_with_data, y=dry_wrs, name="Dry-Run",
            marker_color="rgba(107,114,128,0.4)",
            text=[_pct(v) for v in dry_wrs], textposition="outside",
        )
        fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig.update_layout(
            barmode="group",
            yaxis=dict(tickformat=".0%", range=[0, 1.05]),
            height=320,
            margin=dict(t=10, b=10, l=40, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width='stretch')

    with c2:
        st.subheader("PnL: Live vs Dry-Run")
        live_pnls = [next(r["Live PnL"] for r in rows if r["Asset"] == a) for a in assets_with_data]
        dry_pnls  = [next(r["Dry PnL"]  for r in rows if r["Asset"] == a) for a in assets_with_data]
        fig2 = go.Figure()
        fig2.add_bar(
            x=assets_with_data, y=live_pnls, name="Live",
            marker_color=[ASSET_COLORS[a] for a in assets_with_data],
            text=[f"${v:.2f}" for v in live_pnls], textposition="outside",
        )
        fig2.add_bar(
            x=assets_with_data, y=dry_pnls, name="Dry-Run",
            marker_color="rgba(107,114,128,0.4)",
            text=[f"${v:.2f}" for v in dry_pnls], textposition="outside",
        )
        fig2.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig2.update_layout(
            barmode="group",
            yaxis_title="PnL (USDC)",
            height=320,
            margin=dict(t=10, b=10, l=50, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig2, use_container_width='stretch')

    st.divider()

    # ── Edge vs outcome ──────────────────────────────────────────────────────
    st.subheader("Edge at Trigger vs Outcome (Live Trades)")

    if len(live) > 0:
        asset_filter = st.multiselect(
            "Assets", ASSETS, default=[a for a in ASSETS if a in live["asset"].unique()],
            key="mdl_edge_assets",
        )
        fig3 = go.Figure()
        for asset in asset_filter:
            g = live[live["asset"] == asset]
            won  = g[g["status"] == "won"]
            lost = g[g["status"] == "lost"]
            color = ASSET_COLORS.get(asset, "#888")
            fig3.add_trace(go.Scatter(
                x=won["edge"], y=won["pnl_usdc"],
                mode="markers", name=f"{asset} won",
                marker=dict(color=color, size=6, opacity=0.7, symbol="circle"),
            ))
            fig3.add_trace(go.Scatter(
                x=lost["edge"], y=lost["pnl_usdc"],
                mode="markers", name=f"{asset} lost",
                marker=dict(color=color, size=6, opacity=0.4, symbol="x"),
            ))
        fig3.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig3.update_layout(
            xaxis_title="Edge at trigger (predicted_win − pm_ask)",
            yaxis_title="Realized PnL (USDC)",
            height=340,
            margin=dict(t=10, b=50, l=60, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig3, use_container_width='stretch')

    st.divider()

    # ── Cumulative PnL ───────────────────────────────────────────────────────
    st.subheader("Cumulative PnL: Live vs Dry-Run")

    asset_filter2 = st.multiselect(
        "Assets", ASSETS,
        default=[a for a in ASSETS if a in resolved["asset"].unique()],
        key="mdl_cum_assets",
    )

    fig4 = go.Figure()
    for asset in asset_filter2:
        l = live[live["asset"] == asset].sort_values("date")
        d = dry[dry["asset"] == asset].sort_values("date")
        color = ASSET_COLORS.get(asset, "#888")
        if len(l) > 0:
            fig4.add_trace(go.Scatter(
                x=l["date"], y=l["pnl_usdc"].cumsum(),
                name=f"{asset} live",
                line=dict(color=color, width=2),
            ))
        if len(d) > 0:
            fig4.add_trace(go.Scatter(
                x=d["date"], y=d["pnl_usdc"].cumsum(),
                name=f"{asset} dry-run",
                line=dict(color=color, width=1.5, dash="dash"),
                opacity=0.55,
            ))

    fig4.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", line_width=1)
    fig4.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative PnL (USDC)",
        height=380,
        margin=dict(t=10, b=40, l=60, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig4, use_container_width='stretch')

    # ── Win rate by edge bucket ───────────────────────────────────────────────
    st.divider()
    st.subheader("Win Rate by Edge Bucket (all resolved model trades)")

    if len(resolved) > 10:
        r2 = resolved.copy()
        r2["edge_bucket"] = pd.cut(r2["edge"], bins=8)
        bucket_stats = (
            r2.groupby(["edge_bucket", "is_live"])["status"]
            .apply(lambda s: s.eq("won").mean())
            .reset_index()
        )
        bucket_stats.columns = ["edge_bucket", "is_live", "win_rate"]
        bucket_stats["label"] = bucket_stats["is_live"].map({True: "Live", False: "Dry-Run"})
        bucket_stats["bucket_str"] = bucket_stats["edge_bucket"].astype(str)

        fig5 = go.Figure()
        for label, color in [("Live", "#3b82f6"), ("Dry-Run", "rgba(107,114,128,0.5)")]:
            sub = bucket_stats[bucket_stats["label"] == label]
            fig5.add_trace(go.Bar(
                x=sub["bucket_str"], y=sub["win_rate"],
                name=label, marker_color=color,
                text=[_pct(v) for v in sub["win_rate"]], textposition="outside",
            ))
        fig5.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig5.update_layout(
            barmode="group",
            xaxis_title="Edge bucket",
            yaxis=dict(tickformat=".0%", range=[0, 1.1]),
            height=320,
            margin=dict(t=10, b=60, l=50, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig5, use_container_width='stretch')
