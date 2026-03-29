"""Tab — Live Trading: real-time monitor for the live high-buy executor."""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

LIVE_DIR    = os.path.join("data", "live")
STATUS_JSON = os.path.join(LIVE_DIR, "status.json")
TRADES_CSV  = os.path.join(LIVE_DIR, "trades.csv")

STATUS_COLOR = {"won": "#22c55e", "lost": "#ef4444", "open": "#3b82f6",
                "unresolved": "#f59e0b", "order_failed": "#ef4444", "dry_run": "#a78bfa"}


def _load_status() -> dict | None:
    """Return the most recently updated status file across all instances."""
    try:
        files = sorted(Path(LIVE_DIR).glob("status_*.json"))
        if not files:
            return None
        # Pick the freshest one
        best = max(files, key=lambda p: json.load(open(p)).get("updated_at", 0))
        return json.load(open(best))
    except Exception:
        return None


def _load_trades() -> pd.DataFrame:
    try:
        frames = [pd.read_csv(f) for f in Path(LIVE_DIR).glob("trades_*.csv")]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values("ts").groupby(["asset", "window_start_ts"], as_index=False).last()
        return df.sort_values("ts", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%m/%d %H:%M UTC")


def _pnl_chart(trades: pd.DataFrame) -> go.Figure:
    """Cumulative PnL over time from resolved trades."""
    resolved = trades[trades["status"].isin(["won", "lost"])].copy()
    resolved = resolved.sort_values("ts")

    fig = go.Figure()
    if resolved.empty:
        fig.update_layout(title="No resolved trades yet", height=220,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        return fig

    cum_pnl = resolved["pnl_usdc"].cumsum()
    colors  = ["#22c55e" if p >= 0 else "#ef4444" for p in cum_pnl]
    texts   = [
        f"{_fmt_ts(r['ts'])} {r['asset']} {r['side']} {r['status'].upper()} ${r['pnl_usdc']:+.2f}"
        for _, r in resolved.iterrows()
    ]

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)
    fig.add_trace(go.Scatter(
        x=list(range(1, len(resolved) + 1)),
        y=cum_pnl.tolist(),
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2),
        marker=dict(color=colors, size=8),
        hovertext=texts, hoverinfo="text+y",
        name="Cum PnL",
    ))
    fig.update_layout(
        title=dict(text=f"Live Equity Curve  |  Total: ${cum_pnl.iloc[-1]:+.2f}", font=dict(size=13), x=0),
        xaxis=dict(title="Trade #"),
        yaxis=dict(title="Cumulative PnL ($)", tickformat="$,.2f"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=280, margin=dict(t=40, b=40, l=60, r=20),
        hovermode="x unified",
    )
    return fig


@st.fragment(run_every=2)
def render() -> None:
    status = _load_status()
    trades = _load_trades()

    # ── Executor not running ─────────────────────────────────────────────────
    if status is None:
        st.info(
            "**Executor not running.**\n\n"
            "Start it with:\n"
            "```\npython scripts/live_high_buy.py --threshold 0.80 --capital 500\n```"
        )
        if not trades.empty:
            st.subheader("Previous Trades")
            _render_trades_table(trades)
            st.plotly_chart(_pnl_chart(trades), width="stretch")
        return

    # ── Stale check ─────────────────────────────────────────────────────────
    age = time.time() - status.get("updated_at", 0)
    if age > 15:
        st.warning(f"Status file is {age:.0f}s old — executor may have stopped.")

    # ── Window progress bar ──────────────────────────────────────────────────
    ws       = status.get("window_start", 0)
    we       = status.get("window_end", 0)
    elapsed  = status.get("elapsed_secs", 0)
    remain   = max(0, status.get("remaining_secs", 0))
    total    = we - ws or 300
    threshold    = status.get("threshold", "—")
    wallet_pct   = status.get("wallet_pct", None)
    position_usdc = status.get("position_usdc", None)
    dry_run      = status.get("dry_run", False)

    col_l, col_r = st.columns([3, 1])
    with col_l:
        label = f"Window: {_fmt_ts(ws)} → {_fmt_ts(we)}  ({elapsed}s elapsed, {remain}s remaining)"
        st.progress(min(elapsed / total, 1.0), text=label)
    with col_r:
        badge = "🔵 DRY RUN" if dry_run else "🔴 LIVE"
        pct_str = f"  {wallet_pct:.1%}/trade" if wallet_pct else ""
        pos_str = f"  (~${position_usdc:.2f})" if position_usdc else ""
        st.markdown(f"**{badge}**  threshold=`{threshold:.2f}`{pct_str}{pos_str}")

    # ── Per-asset price tiles ────────────────────────────────────────────────
    asset_data = status.get("assets", {})
    assets = list(asset_data.keys())
    cols = st.columns(len(assets)) if assets else []

    for col, asset in zip(cols, assets):
        info = asset_data[asset]
        up_p  = info.get("up_price")
        dn_p  = info.get("down_price")
        filled = info.get("filled", False)
        trade  = info.get("trade")

        up_str = f"{up_p:.3f}" if up_p is not None else "—"
        dn_str = f"{dn_p:.3f}" if dn_p is not None else "—"

        with col:
            if trade and trade.get("status") == "open":
                side  = trade["side"]
                price = trade["fill_price"]
                usdc  = trade["fill_usdc"]
                st.metric(
                    label=f"**{asset}**  🟢 FILLED",
                    value=f"{side} @ {price:.3f}",
                    delta=f"${usdc:.2f} USDC",
                )
            elif trade and trade.get("status") in ("won", "lost"):
                pnl = trade.get("pnl_usdc", 0)
                icon = "✅" if trade["status"] == "won" else "❌"
                st.metric(
                    label=f"**{asset}**  {icon} {trade['status'].upper()}",
                    value=f"{trade['side']} @ {trade['fill_price']:.3f}",
                    delta=f"${pnl:+.2f}",
                )
            else:
                # Watching — show live prices
                up_hit   = up_p  is not None and up_p  >= threshold
                down_hit = dn_p  is not None and dn_p  >= threshold
                up_color  = "🔥" if up_hit  else ""
                down_color = "🔥" if down_hit else ""
                st.metric(
                    label=f"**{asset}**  {'⏳ WATCHING' if not filled else '✓ filled'}",
                    value=f"UP {up_str}{up_color}",
                    delta=f"DN {dn_str}{down_color}",
                )

    st.divider()

    # ── PnL summary ──────────────────────────────────────────────────────────
    if not trades.empty:
        resolved = trades[trades["status"].isin(["won", "lost"])]
        total_pnl = resolved["pnl_usdc"].sum() if not resolved.empty else 0.0
        n_won     = (resolved["status"] == "won").sum()
        n_lost    = (resolved["status"] == "lost").sum()
        n_open    = (trades["status"] == "open").sum()
        win_rate  = n_won / (n_won + n_lost) if (n_won + n_lost) > 0 else None

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total PnL", f"${total_pnl:+.2f}")
        m2.metric("Trades", len(resolved))
        m3.metric("Win / Loss", f"{n_won} / {n_lost}")
        m4.metric("Win Rate", f"{win_rate:.1%}" if win_rate is not None else "—")
        m5.metric("Open Positions", n_open)

        st.plotly_chart(_pnl_chart(trades), width="stretch")
        st.divider()

    # ── Trade log ────────────────────────────────────────────────────────────
    st.subheader("Trade Log")
    if trades.empty:
        st.caption("No trades yet this session.")
    else:
        _render_trades_table(trades)



def _render_trades_table(trades: pd.DataFrame) -> None:
    display_cols = ["ts", "asset", "side", "fill_price", "fill_usdc",
                    "status", "resolution", "pnl_usdc", "window_start_ts"]
    cols = [c for c in display_cols if c in trades.columns]
    df = trades[cols].copy()

    # Format timestamps
    if "ts" in df.columns:
        df["ts"] = df["ts"].apply(lambda t: _fmt_ts(float(t)) if pd.notna(t) else "—")
    if "window_start_ts" in df.columns:
        df["window_start_ts"] = df["window_start_ts"].apply(
            lambda t: _fmt_ts(float(t)) if pd.notna(t) else "—"
        )

    # Color rows by status
    def _row_style(row):
        s = row.get("status", "")
        bg = {"won": "rgba(34,197,94,0.12)", "lost": "rgba(239,68,68,0.12)",
              "open": "rgba(59,130,246,0.10)"}.get(s, "")
        return [f"background-color: {bg}"] * len(row)

    st.dataframe(
        df.style.apply(_row_style, axis=1),
        hide_index=True,
        width="stretch",
    )
