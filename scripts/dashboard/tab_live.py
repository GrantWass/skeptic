"""Tab — Live Momentum Bots: real-time grid for all status_mom_*.json executors."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

LIVE_DIR     = Path("data/live")
ASSETS_YAML  = Path("config/assets.yaml")
STALE_SECS   = 12   # warn if status older than this


def _load_model_cfg() -> dict:
    try:
        return (yaml.safe_load(ASSETS_YAML.read_text()) or {}).get("MODEL", {})
    except Exception:
        return {}
CARDS_PER_ROW = 2

STATUS_BG = {
    "won":          "rgba(34,197,94,0.10)",
    "lost":         "rgba(239,68,68,0.10)",
    "open":         "rgba(59,130,246,0.07)",
    "unresolved":   "rgba(245,158,11,0.10)",
    "order_failed": "rgba(239,68,68,0.10)",
    "watching":     "rgba(0,0,0,0)",
}
STATUS_ICON = {
    "won":          "✅",
    "lost":         "❌",
    "open":         "🟢",
    "unresolved":   "⚠️",
    "order_failed": "🔴",
    "watching":     "👁️",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_bots() -> list[tuple[str, dict]]:
    """Return [(coin_name, status_dict), ...] sorted by asset name."""
    bots = []
    for path in sorted(LIVE_DIR.glob("status_mom_*.json")):
        try:
            s = json.loads(path.read_text())
            coin = path.stem.replace("status_mom_", "").upper()
            bots.append((coin, s))
        except Exception:
            pass
    return bots


def _load_trades() -> pd.DataFrame:
    frames = []
    for path in LIVE_DIR.glob("trades_mom_*.csv"):
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            pass
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Prefer final status (won/lost/unresolved) over open for each (asset, window)
    def _prefer_final(group):
        finals = group[group["status"].isin(["won", "lost", "unresolved"])]
        if not finals.empty:
            return finals.sort_values("ts").iloc[-1]
        return group.sort_values("ts").iloc[-1]
    df = df.sort_values("ts").groupby(
        ["asset", "window_start_ts"], as_index=False
    ).apply(_prefer_final).reset_index(drop=True)
    return df.sort_values("ts", ascending=False).reset_index(drop=True)


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M UTC")


def _fmt_coin(price: float) -> str:
    if price >= 1000:    return f"{price:,.1f}"
    if price >= 1:       return f"{price:.3f}"
    if price >= 0.01:    return f"{price:.5f}"
    if price >= 0.0001:  return f"{price:.7f}"
    return f"{price:.9f}"


# ── Bot card components ───────────────────────────────────────────────────────

def _move_gauge(move: float, threshold: float) -> go.Figure:
    """
    Compact horizontal gauge. Threshold markers sit at the ±25% marks (¼ of
    visual width from centre). The bar can grow past them; the screen edge
    represents 200% of the threshold. Bar turns green once threshold is crossed.

    Coordinate space: 0.0 = left edge, 0.5 = centre, 1.0 = right edge.
    Threshold markers at 0.25 (−thresh) and 0.75 (+thresh).
    Full screen edge (±2×thresh) = 0.0 / 1.0.
    """
    if threshold <= 0:
        threshold = 1.0

    # Scale: 1×threshold → 0.25 units from centre; 2×threshold → 0.5 (screen edge)
    pct   = min(abs(move) / threshold * 0.25, 0.5)   # capped at screen edge
    color = "#22c55e" if abs(move) >= threshold else "#60a5fa"

    if move >= 0:
        x0, x1 = 0.5, 0.5 + pct
    else:
        x0, x1 = 0.5 - pct, 0.5

    fig = go.Figure()
    # Background
    fig.add_shape(type="rect", x0=0, x1=1, y0=0, y1=1,
                  fillcolor="#1f2937", line_width=0)
    # Fill
    fig.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=1,
                  fillcolor=color, line_width=0)
    # Centre line
    fig.add_shape(type="line", x0=0.5, x1=0.5, y0=0, y1=1,
                  line=dict(color="#6b7280", width=1))
    # Threshold markers at ±25% (¼ of visual)
    fig.add_shape(type="line", x0=0.25, x1=0.25, y0=0, y1=1,
                  line=dict(color="#f59e0b", width=1, dash="dot"))
    fig.add_shape(type="line", x0=0.75, x1=0.75, y0=0, y1=1,
                  line=dict(color="#f59e0b", width=1, dash="dot"))

    fig.update_layout(
        height=32, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _kv(label: str, value: str, color: str = "#9ca3af") -> str:
    return (f"<span style='color:{color};font-size:0.78em'>{label}</span>"
            f"<span style='font-size:0.85em;margin-left:3px'>{value}</span>")


def _render_bot_card(coin: str, s: dict, idx: int = 0, model_cfg: dict | None = None) -> None:
    _model_cfg = model_cfg or {}
    age       = time.time() - s.get("updated_at", 0)
    ws        = s.get("window_start", 0)
    coin_open  = s.get("coin_open")
    coin_cur   = s.get("coin_current") or coin_open
    coin_move  = s.get("coin_move", 0.0)
    threshold  = s.get("threshold_move", 1.0)
    sigma_val  = s.get("sigma_value")
    sigma_ent  = s.get("sigma_entry")
    max_pm     = s.get("max_pm_price")
    direction  = s.get("direction", "both")
    up_mid     = s.get("up_price")
    dn_mid     = s.get("down_price")
    up_ask     = s.get("up_ask")
    dn_ask     = s.get("down_ask")
    filled        = s.get("filled", False)
    trade         = s.get("trade")
    predicted_win = s.get("predicted_win")
    model_trade   = s.get("model_trade")
    elapsed_now   = s.get("elapsed_secs", 0)
    early_thresh  = _model_cfg.get("early_threshold")
    norm_thresh   = _model_cfg.get("edge_threshold", 0.20)

    trade_status = trade.get("status", "open") if trade else ("watching" if not filled else "open")

    with st.container(border=True):
        # ── Header: coin + all config in one compact line ─────────────────────
        sig_str   = f"{sigma_ent}σ·${_fmt_coin(threshold)}" if sigma_val and sigma_ent else "—"
        ws_clock  = _fmt_ts(ws) if ws else "—"
        stale_str = f"⚠ stale {age:.0f}s" if age > STALE_SECS else f"{age:.0f}s ago"
        thresh_str = (f"edge {'**' + str(early_thresh) + '**' if early_thresh and elapsed_now < 30 else norm_thresh}"
                      f"{'→' + str(norm_thresh) + ' @30s' if early_thresh and elapsed_now < 30 else ''}")
        st.markdown(
            f"**{coin}** &nbsp; "
            + "  &nbsp;·&nbsp;  ".join([
                _kv("move", sig_str),
                _kv("dir", direction),
                _kv("max", str(max_pm)),
                _kv("win", ws_clock),
                _kv("", stale_str, "#f59e0b" if age > STALE_SECS else "#6b7280"),
            ]),
            unsafe_allow_html=True,
        )

        # ── Gauge + sparkline in one figure ───────────────────────────────────
        ph = s.get("price_history") or []
        if coin_open is not None and coin_cur is not None:
            sigmas        = coin_move / sigma_val if sigma_val else 0.0
            pct_of_thresh = abs(coin_move) / threshold * 100 if threshold else 0.0
            dir_arrow     = "▲" if coin_move > 0 else ("▼" if coin_move < 0 else "—")

            if len(ph) >= 3 and sigma_val:
                fig = go.Figure()

                # Price line in σ units
                n   = len(ph)
                xs  = [i / max(n - 1, 1) * elapsed_now for i in range(n)]
                ys  = [(p - (coin_open or ph[0])) / sigma_val for p in ph]

                triggered = filled or (trade is not None)
                line_color = "#22c55e" if triggered else "#60a5fa"
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(color=line_color, width=2),
                    name="move (σ)", yaxis="y1",
                    hovertemplate="%{y:.2f}σ<extra></extra>",
                ))

                # Threshold bands
                sig_ent = sigma_ent or 1.0
                for mult, dash in [(sig_ent, "dash"), (-sig_ent, "dash")]:
                    fig.add_hline(y=mult, line=dict(color="#f59e0b", width=1, dash=dash),
                                  yref="y1")

                # Fill marker
                if trade and trade.get("elapsed_second") is not None:
                    fill_x   = trade.get("elapsed_second", 0)
                    fill_sig = (trade.get("coin_trigger", coin_open or 0) - (coin_open or 0)) / sigma_val
                    fig.add_trace(go.Scatter(
                        x=[fill_x], y=[fill_sig],
                        mode="markers",
                        marker=dict(color="#f59e0b", size=8, symbol="diamond"),
                        name="entry", yaxis="y1",
                        hovertemplate=f"entry @{fill_x}s<extra></extra>",
                    ))

                # PM ask lines on secondary y-axis
                if up_ask is not None:
                    fig.add_hline(y=up_ask, line=dict(color="#34d399", width=1, dash="dot"),
                                  yref="y2", annotation_text=f"UP {up_ask:.2f}",
                                  annotation_position="right",
                                  annotation_font=dict(size=9, color="#34d399"))
                if dn_ask is not None:
                    fig.add_hline(y=dn_ask, line=dict(color="#f87171", width=1, dash="dot"),
                                  yref="y2", annotation_text=f"DN {dn_ask:.2f}",
                                  annotation_position="right",
                                  annotation_font=dict(size=9, color="#f87171"))
                if max_pm is not None:
                    fig.add_hline(y=max_pm, line=dict(color="#9ca3af", width=1, dash="longdash"),
                                  yref="y2", annotation_text=f"max {max_pm:.2f}",
                                  annotation_position="right",
                                  annotation_font=dict(size=9, color="#9ca3af"))

                # "Now" vertical line
                fig.add_vline(x=elapsed_now, line=dict(color="#6b7280", width=1, dash="dot"))

                data_range = max(abs(v) for v in ys) if ys else 0
                sig_range  = max(data_range * 1.4, sig_ent * 1.2, 0.05)
                fig.update_layout(
                    height=130,
                    margin=dict(l=0, r=60, t=4, b=4),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    xaxis=dict(range=[0, 300], showgrid=False, showticklabels=False, zeroline=False),
                    yaxis=dict(
                        range=[-sig_range, sig_range],
                        showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                        zeroline=True, zerolinecolor="#374151", zerolinewidth=1,
                        tickfont=dict(size=9, color="#6b7280"),
                        title=dict(text="σ", font=dict(size=9, color="#6b7280")),
                        tickformat=".1f", side="left",
                    ),
                    yaxis2=dict(
                        range=[0, 1], overlaying="y", side="right",
                        showgrid=False, showticklabels=False, zeroline=False,
                    ),
                )
                st.plotly_chart(fig, config={"displayModeBar": False},
                                key=f"combo_{coin}_{idx}", width="stretch")
            else:
                st.plotly_chart(_move_gauge(coin_move, threshold),
                                width="stretch", config={"displayModeBar": False},
                                key=f"gauge_{coin}_{idx}")

            st.caption(f"{dir_arrow} ${_fmt_coin(abs(coin_move))}  ·  {pct_of_thresh:.0f}% of threshold  ·  {sigmas:+.2f}σ  ·  {thresh_str}")

        # ── PM prices + model prediction in one row ───────────────────────────
        up_ref  = up_ask if up_ask is not None else up_mid
        dn_ref  = dn_ask if dn_ask is not None else dn_mid
        up_edge = (predicted_win - up_ref)         if predicted_win is not None and up_ref is not None else None
        dn_edge = ((1.0 - predicted_win) - dn_ref) if predicted_win is not None and dn_ref is not None else None
        active_thresh = early_thresh if (early_thresh is not None and elapsed_now < 30) else norm_thresh

        def _edge_str(edge):
            if edge is None: return "—"
            fire = edge >= active_thresh
            col  = "#22c55e" if fire else "#9ca3af"
            tag  = " ▲" if fire else ""
            return f"<span style='color:{col}'>{edge:+.3f}{tag}</span>"

        hot_up = up_ask is not None and up_ask >= (max_pm or 1.0)
        hot_dn = dn_ask is not None and dn_ask >= (max_pm or 1.0)
        spread_str = f"{up_ask + dn_ask - 1:+.3f}" if up_ask is not None and dn_ask is not None else "—"
        up_ask_str = f"<span style='color:#ef4444'>{up_ask:.2f}!</span>" if hot_up else (f"{up_ask:.2f}" if up_ask is not None else "—")
        dn_ask_str = f"<span style='color:#ef4444'>{dn_ask:.2f}!</span>" if hot_dn else (f"{dn_ask:.2f}" if dn_ask is not None else "—")

        parts = [
            _kv("UP ask", up_ask_str),
            _kv("DN ask", dn_ask_str),
            _kv("vig", spread_str),
        ]
        if predicted_win is not None:
            parts += [
                _kv("P(UP)", f"{predicted_win:.1%}"),
                _kv("UP edge", _edge_str(up_edge)),
                _kv("DN edge", _edge_str(dn_edge)),
            ]
        st.markdown("&nbsp;&nbsp;".join(parts), unsafe_allow_html=True)

        # ── Model features (collapsed) ────────────────────────────────────────
        features = s.get("features") or {}
        if features:
            with st.expander("Model features", expanded=False):
                FEAT_LABELS = {
                    "move_sigmas": "move(σ)", "elapsed_second": "elapsed",
                    "hour_utc": "hour", "vel_2s": "vel2s", "vel_5s": "vel5s",
                    "vel_10s": "vel10s", "acc_4s": "acc4s", "vel_ratio": "ratio",
                    "vel_decay": "decay", "vol_10s_log": "vol10s",
                }
                items = [(FEAT_LABELS.get(k, k), v) for k, v in features.items()]
                st.markdown(
                    "  &nbsp;·&nbsp;  ".join(
                        _kv(lbl or "", "—" if v is None else str(round(v, 3)))
                        for lbl, v in items
                    ),
                    unsafe_allow_html=True,
                )

        # ── Trade status ──────────────────────────────────────────────────────
        if trade:
            icon       = STATUS_ICON.get(trade_status, "")
            side       = trade.get("side", "")
            fp         = trade.get("fill_price", 0.0)
            fusdc      = trade.get("fill_usdc",  0.0)
            pnl        = trade.get("pnl_usdc")
            slip       = trade.get("slippage", 0.0)
            resolution = trade.get("resolution")
            took_profit = (
                trade_status == "won"
                and resolution is not None
                and float(resolution) not in (0.0, 1.0)
            )
            pnl_str  = f"PnL ${pnl:+.4f}" if pnl is not None else ""
            slip_str = f"slip={slip:+.4f}" if slip else ""
            tp_str   = f"sold @ {float(resolution):.2f} 💰" if took_profit else ""

            parts = [
                f"{icon}",
                f"**{trade_status.upper()}**",
                f"{side}",
                f"@ `{fp:.4f}`",
                f"{fusdc:.2f}",
                tp_str,
                pnl_str,
                slip_str,
            ]

            # Remove empty parts and join with consistent spacing
            line = "  ".join(p for p in parts if p)

            st.markdown(line)
        elif not filled:
            st.caption("Watching…")

        # ── Model trade ───────────────────────────────────────────────────────
        ws_now = int(time.time()) // 300 * 300
        if model_trade and int(model_trade.get("window_start_ts") or 0) == ws_now:
            mt_status = model_trade.get("status", "open")
            mt_side   = model_trade.get("side", "")
            mt_fp     = model_trade.get("fill_price", 0.0)
            mt_fusdc  = model_trade.get("fill_usdc", 0.0)
            mt_slip   = model_trade.get("slippage")
            mt_pnl    = model_trade.get("pnl_usdc")
            pnl_str  = f"  PnL **${mt_pnl:+.4f}**" if mt_pnl is not None else ""
            slip_str = f"  slip={mt_slip:+.4f}" if mt_slip else ""
            icon      = {"won": "✅", "lost": "❌", "open": "🔵", "order_failed": "⚠️"}.get(mt_status, "🔵")
            if mt_status == "order_failed":
                st.markdown(f"**MODEL** {icon} **ORDER FAILED** {mt_side}")
            else:
                st.markdown(
                    f"**MODEL** {icon} **{mt_status.upper()}** {mt_side} @ `{mt_fp:.4f}` ${mt_fusdc:.2f}{pnl_str}{slip_str}"
                )


# ── Shared window bar ────────────────────────────────────────────────────────

def _render_window_bar() -> None:
    """Single progress bar for the current 5-minute window (shared by all bots)."""
    now = time.time()
    ws  = int(now) // 300 * 300
    we  = ws + 300
    elapsed   = int(now - ws)
    remaining = max(0, we - int(now))
    st.progress(
        elapsed / 300,
        text=f"Window  {_fmt_ts(ws)} → {_fmt_ts(we)}  ·  {elapsed}s elapsed  ·  {remaining}s left",
    )


# ── Summary header ────────────────────────────────────────────────────────────

def _streak(df: pd.DataFrame):
    filtered = cast(pd.DataFrame, df[df["status"].isin(["won", "lost"])])
    statuses = filtered.sort_values(by="ts")["status"].tolist()
    if not statuses:
        return None, 0
    last = statuses[-1]
    count = 0
    for s in reversed(statuses):
        if s == last:
            count += 1
        else:
            break
    return last, count


def _render_summary(trades: pd.DataFrame, model_trades: pd.DataFrame, n_bots: int) -> None:
    # Combine resolved trades from both strategies
    frames = []
    if not trades.empty:
        frames.append(trades)
    if not model_trades.empty:
        frames.append(model_trades[model_trades.columns.intersection(trades.columns if not trades.empty else model_trades.columns)])

    if frames and not trades.empty and not model_trades.empty:
        combined_cols = list(set(trades.columns) & set(model_trades.columns))
        combined = pd.concat([trades[combined_cols], model_trades[combined_cols]], ignore_index=True)
    elif not trades.empty:
        combined = trades.copy()
    elif not model_trades.empty:
        combined = model_trades.copy()
    else:
        combined = pd.DataFrame()

    resolved_mom   = trades[trades["status"].isin(["won", "lost"])]      if not trades.empty       else pd.DataFrame()
    resolved_mdl   = model_trades[model_trades["status"].isin(["won", "lost"])] if not model_trades.empty else pd.DataFrame()
    resolved_all   = pd.concat([resolved_mom, resolved_mdl], ignore_index=True) if (not resolved_mom.empty or not resolved_mdl.empty) else pd.DataFrame()

    total_pnl = pd.to_numeric(resolved_all["pnl_usdc"], errors="coerce").sum() if not resolved_all.empty else 0.0

    total_deployed = 0.0
    if not resolved_mom.empty and "fill_usdc" in resolved_mom.columns:
        total_deployed += pd.to_numeric(resolved_mom["fill_usdc"], errors="coerce").sum()
    if not resolved_mdl.empty and "fill_usdc" in resolved_mdl.columns:
        total_deployed += pd.to_numeric(resolved_mdl["fill_usdc"], errors="coerce").sum()
    roi_pct = total_pnl / total_deployed * 100 if total_deployed > 0 else None

    n_won  = (resolved_all["status"] == "won").sum()  if not resolved_all.empty else 0
    n_lost = (resolved_all["status"] == "lost").sum() if not resolved_all.empty else 0
    win_rate = n_won / (n_won + n_lost) if (n_won + n_lost) > 0 else None

    streak_label, streak_count = _streak(combined) if not combined.empty and "ts" in combined.columns and "status" in combined.columns else (None, 0)
    if streak_label == "won":
        streak_str = f"🔥 {streak_count}W"
    elif streak_label == "lost":
        streak_str = f"❄️ {streak_count}L"
    else:
        streak_str = "—"

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Active Bots",  n_bots)
    c2.metric("Total PnL",    f"${total_pnl:+.4f}")
    c3.metric("ROI%",         f"{roi_pct:+.2f}%" if roi_pct is not None else "—")
    c4.metric("Resolved",     f"{n_won}W / {n_lost}L")
    c5.metric("Win Rate",     f"{win_rate:.1%}" if win_rate is not None else "—")
    c6.metric("Streak",       streak_str)

    # Last trade callout
    if not resolved_all.empty and "ts" in resolved_all.columns:
        last_row = resolved_all.sort_values("ts").iloc[-1]
        asset  = last_row.get("asset", "?")
        side   = last_row.get("side", "?")
        status = last_row.get("status", "?")
        pnl    = last_row.get("pnl_usdc")
        ts_val = last_row.get("ts")
        pnl_str  = f"${float(pnl):+.4f}" if pnl is not None and pd.notna(pnl) else "—"
        time_str = _fmt_ts(float(ts_val)) if ts_val is not None and pd.notna(ts_val) else "—"
        st.info(f"Last trade: {asset} {side} {status.upper()}  {pnl_str}  @ {time_str}")


# ── Equity curve ──────────────────────────────────────────────────────────────

def _pnl_chart(trades: pd.DataFrame, model_trades: pd.DataFrame = None) -> go.Figure:
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)", line_width=1)

    titles = []

    def _norm_x(n: int) -> list[float]:
        """Spread n trades evenly across 0–100."""
        return [(i + 1) / n * 100 for i in range(n)]

    def _hover(df: pd.DataFrame, n: int) -> list[str]:
        ts_col     = list(pd.to_numeric(df["ts"],       errors="coerce"))  # type: ignore[arg-type]
        asset_col  = df["asset"].astype(str).tolist()
        side_col   = df["side"].astype(str).str.upper().tolist()
        status_col = df["status"].astype(str).str.upper().tolist()
        pnl_col    = list(pd.to_numeric(df["pnl_usdc"], errors="coerce"))  # type: ignore[arg-type]
        return [
            f"#{i+1}/{n}  {_fmt_ts(float(ts))}  {asset} {side} {status}  ${float(pnl):+.4f}"
            for i, (ts, asset, side, status, pnl)
            in enumerate(zip(ts_col, asset_col, side_col, status_col, pnl_col))
        ]

    def _resolved(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["status"].isin(["won", "lost"])].copy().sort_values("ts")  # type: ignore[arg-type]

    def _cumsum(df: pd.DataFrame) -> "pd.Series[float]":
        return pd.to_numeric(df["pnl_usdc"], errors="coerce").fillna(0.0).cumsum()  # type: ignore[return-value]

    mom_resolved = _resolved(trades) if not trades.empty else pd.DataFrame()
    if not mom_resolved.empty:
        n   = len(mom_resolved)
        cum = _cumsum(mom_resolved)
        colors = ["#22c55e" if v >= 0 else "#ef4444" for v in cum]
        fig.add_trace(go.Scatter(
            x=_norm_x(n), y=cum.tolist(),
            name="Momentum",
            mode="lines+markers",
            line=dict(color="#3b82f6", width=2),
            marker=dict(color=colors, size=7),
            hovertext=_hover(mom_resolved, n), hoverinfo="text+y",
        ))
        titles.append(f"Momentum ${cum.iloc[-1]:+.4f}")

    mdl_resolved = pd.DataFrame()
    if model_trades is not None and not model_trades.empty:
        mdl_resolved = _resolved(model_trades)
        if not mdl_resolved.empty:
            n_m   = len(mdl_resolved)
            cum_m = _cumsum(mdl_resolved)
            colors_m = ["#22c55e" if v >= 0 else "#ef4444" for v in cum_m]
            fig.add_trace(go.Scatter(
                x=_norm_x(n_m), y=cum_m.tolist(),
                name="Model",
                mode="lines+markers",
                line=dict(color="#a855f7", width=2),
                marker=dict(color=colors_m, size=7),
                hovertext=_hover(mdl_resolved, n_m), hoverinfo="text+y",
            ))
            titles.append(f"Model ${cum_m.iloc[-1]:+.4f}")

    title_str = "Equity Curve  —  " + "  ·  ".join(titles) if titles else "No resolved trades yet"
    fig.update_layout(
        title=dict(text=title_str, font=dict(size=13), x=0),
        xaxis=dict(title="Trade Progress (%)", range=[0, 100],
                   tickvals=[0, 25, 50, 75, 100], ticktext=["0%", "25%", "50%", "75%", "100%"]),
        yaxis=dict(title="Cumulative PnL ($)", tickformat="$.4f"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=260, margin=dict(t=36, b=36, l=64, r=20),
        hovermode="x unified", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Strategy stats ───────────────────────────────────────────────────────────

def _strategy_stats(df: pd.DataFrame, extra_cols: list[str] | None = None) -> dict:
    """Compute summary stats from a resolved+open trades DataFrame."""
    num = lambda col: pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)
    resolved = df[df["status"].isin(["won", "lost"])] if not df.empty else df
    n_res  = len(resolved)
    n_won  = (resolved["status"] == "won").sum() if n_res else 0
    pnl    = num("pnl_usdc")

    total_deployed = pd.to_numeric(resolved["fill_usdc"], errors="coerce").sum() if (n_res and "fill_usdc" in resolved.columns) else 0.0
    total_pnl_val  = pnl[df["status"].isin(["won", "lost"])].sum() if n_res else 0.0
    roi_pct = total_pnl_val / total_deployed * 100 if total_deployed > 0 else None

    avg_slip_val = num("slippage").mean() if "slippage" in df.columns else None

    res = {
        "n": len(df),
        "n_resolved": n_res,
        "n_won": int(n_won),
        "win_rate": n_won / n_res if n_res else None,
        "total_pnl": total_pnl_val,
        "avg_pnl":   pnl[df["status"].isin(["won","lost"])].mean() if n_res else None,
        "avg_fill":  num("fill_price").mean() or None,
        "avg_usdc":  num("fill_usdc").mean() or None,
        "avg_slip":  avg_slip_val,
        "roi_pct":   roi_pct,
    }

    for col in (extra_cols or []):
        res[f"avg_{col}"] = num(col).mean() if col in df.columns else None

    # avg_edge_net: edge after slippage
    if "avg_edge" in res and res.get("avg_edge") is not None and avg_slip_val is not None:
        res["avg_edge_net"] = res["avg_edge"] - avg_slip_val
    elif "edge" in (extra_cols or []) and res.get("avg_edge") is not None and avg_slip_val is not None:
        res["avg_edge_net"] = res["avg_edge"] - avg_slip_val
    else:
        res["avg_edge_net"] = None

    # avg_pnl by direction
    if n_res and "side" in resolved.columns:
        up_resolved = resolved[resolved["side"].str.upper() == "UP"]
        dn_resolved = resolved[resolved["side"].str.upper() == "DOWN"]
        res["n_up"]       = len(up_resolved)
        res["n_dn"]       = len(dn_resolved)
        res["avg_pnl_up"] = pd.to_numeric(up_resolved["pnl_usdc"], errors="coerce").mean() if not up_resolved.empty else None
        res["avg_pnl_dn"] = pd.to_numeric(dn_resolved["pnl_usdc"], errors="coerce").mean() if not dn_resolved.empty else None
    else:
        res["n_up"] = res["n_dn"] = 0
        res["avg_pnl_up"] = res["avg_pnl_dn"] = None

    return res


def _render_strategy_stats(trades: pd.DataFrame, model_trades: pd.DataFrame) -> None:
    st.subheader("Strategy Stats")
    mom_col, mdl_col = st.columns(2)

    def _show(col, label: str, s: dict) -> None:
        col.markdown(f"**{label}**")
        rows = [
            ("Win rate",         f"{s['win_rate']:.1%}" if s["win_rate"] is not None else "—"),
            ("ROI%",             f"{s['roi_pct']:+.2f}%" if s["roi_pct"] is not None else "—"),
            ("Avg PnL / trade",  f"${s['avg_pnl']:+.4f}" if s["avg_pnl"] is not None else "—"),
            ("Avg PnL UP",       f"${s['avg_pnl_up']:+.4f} ({s['n_up']})" if s["avg_pnl_up"] is not None else "—"),
            ("Avg PnL DOWN",     f"${s['avg_pnl_dn']:+.4f} ({s['n_dn']})" if s["avg_pnl_dn"] is not None else "—"),
            ("Avg fill price",   f"{s['avg_fill']:.4f}" if s["avg_fill"] is not None else "—"),
            ("Avg fill USDC",    f"${s['avg_usdc']:.4f}" if s["avg_usdc"] is not None else "—"),
            ("Avg slippage",     f"{s['avg_slip']:+.4f}" if s["avg_slip"] is not None else "—"),
        ]
        if "avg_edge" in s:
            rows.insert(4, ("Avg edge", f"{s['avg_edge']:+.4f}" if s["avg_edge"] is not None else "—"))
        if "avg_edge_net" in s and s["avg_edge_net"] is not None:
            edge_idx = next((i for i, r in enumerate(rows) if r[0] == "Avg edge"), len(rows))
            rows.insert(edge_idx + 1, ("Avg net edge (edge − slip)", f"{s['avg_edge_net']:+.4f}"))
        if "avg_predicted_win" in s:
            rows.insert(2, ("Avg P(win)", f"{s['avg_predicted_win']:.1%}" if s["avg_predicted_win"] is not None else "—"))
        if "avg_elapsed_second" in s:
            rows.append(("Avg entry time", f"{s['avg_elapsed_second']:.0f}s" if s["avg_elapsed_second"] is not None else "—"))
        for name, val in rows:
            col.markdown(f"<div style='display:flex;justify-content:space-between'>"
                         f"<span style='color:#9ca3af'>{name}</span><span>{val}</span></div>",
                         unsafe_allow_html=True)

    if not trades.empty:
        _show(mom_col, "Momentum", _strategy_stats(trades))
    else:
        mom_col.caption("No momentum trades yet.")

    if not model_trades.empty:
        _show(mdl_col, "Model", _strategy_stats(model_trades, extra_cols=["edge", "predicted_win", "elapsed_second"]))
    else:
        mdl_col.caption("No model trades yet.")


# ── Per-coin stats ───────────────────────────────────────────────────────────

MODEL_TRADE_COLS = [
    "ts", "asset", "side", "token_id",
    "fill_price", "fill_size", "fill_usdc",
    "predicted_win", "edge",
    "elapsed_second", "coin_open", "coin_trigger", "coin_move",
    "window_start_ts", "window_end_ts",
    "resolution", "pnl_usdc", "status", "order_id", "slippage",
]


def _load_model_trades() -> pd.DataFrame:
    frames = []
    for path in LIVE_DIR.glob("trades_model_*.csv"):
        try:
            # Skip the CSV header row and use canonical column names.
            # Handles old files whose header had fewer columns than the data rows.
            df_raw = pd.read_csv(path, names=MODEL_TRADE_COLS, skiprows=1,
                                 on_bad_lines="skip")
            frames.append(df_raw)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])
    # Keep final status per (asset, window_start_ts) — prefer resolved over open
    def _prefer_final(group):
        finals = group[group["status"].isin(["won", "lost", "unresolved"])]
        if not finals.empty:
            return finals.sort_values("ts").iloc[-1]
        return group.sort_values("ts").iloc[-1]
    df = df.sort_values("ts").groupby(
        ["asset", "window_start_ts"], as_index=False
    ).apply(_prefer_final).reset_index(drop=True)
    return df.sort_values("ts", ascending=False).reset_index(drop=True)


def _render_per_coin_stats(trades: pd.DataFrame) -> None:
    resolved = trades[trades["status"].isin(["won", "lost"])]
    model_trades = _load_model_trades()
    model_resolved = model_trades[model_trades["status"].isin(["won", "lost"])] if not model_trades.empty else pd.DataFrame()

    if resolved.empty and model_resolved.empty:
        return

    st.subheader("Per-Coin Stats")
    assets = sorted(set(
        (resolved["asset"].unique().tolist() if not resolved.empty else []) +
        (model_resolved["asset"].unique().tolist() if not model_resolved.empty else [])
    ))
    cols = st.columns(len(assets))
    for col, asset in zip(cols, assets):
        col.markdown(f"**{asset}**")
        if not resolved.empty and asset in resolved["asset"].values:
            adf     = resolved[resolved["asset"] == asset]
            n_won   = (adf["status"] == "won").sum()
            n_lost  = (adf["status"] == "lost").sum()
            total   = n_won + n_lost
            win_rate = n_won / total if total else 0.0
            pnl     = adf["pnl_usdc"].sum()
            col.metric("Momentum", f"${pnl:+.2f}", f"{win_rate:.0%} ({n_won}W/{n_lost}L)")
        if not model_resolved.empty and asset in model_resolved["asset"].values:
            mdf     = model_resolved[model_resolved["asset"] == asset]
            n_won   = (mdf["status"] == "won").sum()
            n_lost  = (mdf["status"] == "lost").sum()
            total   = n_won + n_lost
            win_rate = n_won / total if total else 0.0
            pnl     = mdf["pnl_usdc"].sum()
            col.metric("Model", f"${pnl:+.2f}", f"{win_rate:.0%} ({n_won}W/{n_lost}L)")


# ── Unified trade log ─────────────────────────────────────────────────────────

def _render_unified_trade_log(trades: pd.DataFrame, model_trades: pd.DataFrame) -> None:
    st.subheader("Trade Log")

    mom_rows = pd.DataFrame()
    if not trades.empty:
        mom_cols_avail = [c for c in ["ts", "asset", "side", "fill_price", "fill_usdc", "slippage", "pnl_usdc", "status"] if c in trades.columns]
        mom_rows = trades[mom_cols_avail].copy()
        mom_rows["strategy"] = "MOM"
        for missing in ["edge", "predicted_win", "elapsed_second"]:
            if missing not in mom_rows.columns:
                mom_rows[missing] = None

    mdl_rows = pd.DataFrame()
    if not model_trades.empty:
        mdl_cols_avail = [c for c in ["ts", "asset", "side", "fill_price", "fill_usdc", "slippage", "edge", "predicted_win", "elapsed_second", "pnl_usdc", "status"] if c in model_trades.columns]
        mdl_rows = model_trades[mdl_cols_avail].copy()
        mdl_rows["strategy"] = "MODEL"

    if mom_rows.empty and mdl_rows.empty:
        st.caption("No trades recorded yet.")
        return

    combined = pd.concat([mom_rows, mdl_rows], ignore_index=True)
    combined["ts"] = pd.to_numeric(combined["ts"], errors="coerce")
    combined = combined.sort_values("ts", ascending=False).reset_index(drop=True)
    if "status" in combined.columns:
        combined = combined[combined["status"] != "order_failed"].reset_index(drop=True)

    # Strategy filter
    strategy_filter = st.radio("Show", ["All", "MOM", "MODEL"], horizontal=True)
    if strategy_filter != "All":
        combined = combined[combined["strategy"] == strategy_filter].reset_index(drop=True)

    # Format columns
    display = combined.copy()
    display["ts"] = display["ts"].apply(lambda t: _fmt_ts(float(t)) if pd.notna(t) else "—")
    display["predicted_win"] = display["predicted_win"].apply(
        lambda v: f"{float(v):.1%}" if pd.notna(v) and v is not None else "—"
    )
    display["edge"] = display["edge"].apply(
        lambda v: f"{float(v):+.3f}" if pd.notna(v) and v is not None else "—"
    )
    # Add directional arrow prefix to side
    def _side_fmt(v):
        if pd.isna(v) or v is None:
            return "—"
        s = str(v).upper()
        if s == "UP":
            return f"▲ {v}"
        if s == "DOWN":
            return f"▼ {v}"
        return str(v)
    display["side"] = display["side"].apply(_side_fmt)

    show_cols = ["ts", "strategy", "asset", "side", "fill_price", "fill_usdc",
                 "slippage", "edge", "predicted_win", "elapsed_second", "status", "pnl_usdc"]
    out = display[[c for c in show_cols if c in display.columns]]

    def _row_color(row):
        bg = {
            "won":  "rgba(34,197,94,0.12)",
            "lost": "rgba(239,68,68,0.12)",
            "open": "rgba(59,130,246,0.09)",
        }.get(row.get("status", ""), "")
        return [f"background-color: {bg}"] * len(row)

    st.dataframe(out.style.apply(_row_color, axis=1), hide_index=True, width="stretch")


# ── Main ─────────────────────────────────────────────────────────────────────

@st.fragment(run_every=2)
def render() -> None:
    bots        = _load_bots()
    trades      = _load_trades()
    model_trades = _load_model_trades()
    model_cfg   = _load_model_cfg()

    st.subheader("🤖 Live Momentum Bots")

    tab_ov, tab_bots, tab_log, tab_analytics = st.tabs(["Overview", "Bots", "Trade Log", "Analytics"])

    # ── Overview ──────────────────────────────────────────────────────────────
    with tab_ov:
        _render_window_bar()
        _render_summary(trades, model_trades, len(bots))
        st.divider()

        if not bots:
            st.info(
                "No momentum bots running.  Start one with:\n"
                "```\npython scripts/live_momentum_buy.py --asset BTC --name mom_btc --dry-run\n```"
            )
        else:
            # Active trades summary: one row per bot, showing current trade state
            ws_now = int(time.time()) // 300 * 300
            active_rows = []
            for coin, s in bots:
                trade       = s.get("trade")
                model_trade = s.get("model_trade")
                if trade:
                    status = trade.get("status", "open")
                    active_rows.append({
                        "asset":    coin,
                        "strategy": "MOM",
                        "side":     ("▲ " if str(trade.get("side","")).upper() == "UP" else "▼ ") + str(trade.get("side", "")),
                        "fill":     f"{trade.get('fill_price', 0):.3f}",
                        "usdc":     f"${trade.get('fill_usdc', 0):.2f}",
                        "status":   status,
                        "pnl":      f"${trade['pnl_usdc']:+.4f}" if trade.get("pnl_usdc") is not None else "—",
                    })
                if model_trade and int(model_trade.get("window_start_ts") or 0) == ws_now:
                    mt_status = model_trade.get("status", "open")
                    active_rows.append({
                        "asset":    coin,
                        "strategy": "MODEL",
                        "side":     ("▲ " if str(model_trade.get("side","")).upper() == "UP" else "▼ ") + str(model_trade.get("side", "")),
                        "fill":     f"{model_trade.get('fill_price', 0):.3f}",
                        "usdc":     f"${model_trade.get('fill_usdc', 0):.2f}",
                        "status":   mt_status,
                        "pnl":      f"${model_trade['pnl_usdc']:+.4f}" if model_trade.get("pnl_usdc") is not None else "—",
                    })

            if active_rows:
                def _row_color(row):
                    bg = {"won": "rgba(34,197,94,0.12)", "lost": "rgba(239,68,68,0.12)",
                          "open": "rgba(59,130,246,0.09)", "order_failed": "rgba(239,68,68,0.12)"
                         }.get(row.get("status", ""), "")
                    return [f"background-color: {bg}"] * len(row)
                df_active = pd.DataFrame(active_rows)
                st.dataframe(df_active.style.apply(_row_color, axis=1), hide_index=True, width="stretch")
            else:
                st.caption("No active trades this window.")

    # ── Bots ──────────────────────────────────────────────────────────────────
    with tab_bots:
        _render_window_bar()
        if not bots:
            st.info(
                "No momentum bots running.  Start one with:\n"
                "```\npython scripts/live_momentum_buy.py --asset BTC --name mom_btc --dry-run\n```"
            )
        else:
            for i in range(0, len(bots), CARDS_PER_ROW):
                row  = bots[i : i + CARDS_PER_ROW]
                cols = st.columns(len(row))
                for j, (col, (coin, s)) in enumerate(zip(cols, row)):
                    with col:
                        _render_bot_card(coin, s, idx=i + j, model_cfg=model_cfg)

    # ── Trade Log ─────────────────────────────────────────────────────────────
    with tab_log:
        _render_unified_trade_log(trades, model_trades)

    # ── Analytics ─────────────────────────────────────────────────────────────
    with tab_analytics:
        if not trades.empty or not model_trades.empty:
            st.plotly_chart(_pnl_chart(trades, model_trades), width="stretch")
            st.divider()
            _render_strategy_stats(trades, model_trades)
            st.divider()
            _render_per_coin_stats(trades)
        else:
            st.caption("No trades yet — analytics will appear once trades are recorded.")
