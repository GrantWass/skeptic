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


def _load_strategy_cfg(asset: str | None = None) -> tuple[dict, dict]:
    """Load strategy config (global + per-asset overrides if asset specified).

    Returns (momentum_cfg, model_cfg) where each is the global section merged
    with per-asset overrides.
    """
    try:
        cfg = yaml.safe_load(ASSETS_YAML.read_text()) or {}
        global_momentum = cfg.get("MOMENTUM", {})
        global_model = cfg.get("MODEL", {})
        if asset is not None:
            # Merge per-asset overrides with global defaults
            per_asset_momentum = cfg.get(asset.upper(), {}).get("momentum", {})
            per_asset_model = cfg.get(asset.upper(), {}).get("model", {})
            return {**global_momentum, **per_asset_momentum}, {**global_model, **per_asset_model}
        return global_momentum, global_model
    except Exception:
        return {}, {}
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


_MOM_NUMERIC_COLS = {
    "ts", "fill_price", "fill_size", "fill_usdc", "fee_usdc",
    "sigma_value", "sigma_entry", "max_pm_price",
    "elapsed_second", "coin_open", "coin_trigger", "coin_move",
    "slippage", "window_start_ts", "window_end_ts",
    "sign_ms", "post_ms", "order_ms",
    "resolution", "pnl_usdc",
}


def _load_trades() -> pd.DataFrame:
    frames = []
    mom_trade_cols = [
        "ts", "asset", "side", "token_id",
        "fill_price", "fill_size", "fill_usdc", "fee_usdc",
        "sigma_value", "sigma_entry", "max_pm_price",
        "elapsed_second", "coin_open", "coin_trigger", "coin_move",
        "slippage", "window_start_ts", "window_end_ts",
        "sign_ms", "post_ms", "order_ms",
        "resolution", "pnl_usdc", "status", "order_id",
    ]
    for path in LIVE_DIR.glob("trades_mom_*.csv"):
        try:
            df_raw = pd.read_csv(path, on_bad_lines="skip")
            for col in mom_trade_cols:
                if col not in df_raw.columns:
                    df_raw[col] = pd.NA
            frames.append(df_raw[mom_trade_cols])
        except Exception:
            pass
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=mom_trade_cols)
    df = pd.concat(frames, ignore_index=True)
    for col in _MOM_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "ts" in df.columns:
        df = df.dropna(subset=["ts"])

    required = {"asset", "window_start_ts", "status", "ts"}
    if not required.issubset(df.columns):
        return df.sort_values("ts", ascending=False).reset_index(drop=True) if "ts" in df.columns else df

    # For each trade (same asset, window, timestamp), prefer final status (won/lost/unresolved) over open
    def _prefer_final(group):
        finals = group[group["status"].isin(["won", "lost", "unresolved", "fok_won", "fok_lost"])]
        if not finals.empty:
            return finals.sort_values("ts").iloc[-1]
        return group.sort_values("ts").iloc[-1]
    df = df.sort_values("ts").groupby(
        ["asset", "window_start_ts", "ts"], as_index=False
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


def _downsample_every_2s(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    """Keep one plotted point per 2-second bucket for compact bot-card charts."""
    if not xs or len(xs) != len(ys):
        return xs, ys

    keep_x: list[float] = []
    keep_y: list[float] = []
    last_bucket = None

    for x, y in zip(xs, ys):
        bucket = int(max(x, 0) // 2)
        if bucket != last_bucket:
            keep_x.append(x)
            keep_y.append(y)
            last_bucket = bucket

    # Always include latest point so "now" aligns with current state.
    if keep_x[-1] != xs[-1]:
        keep_x.append(xs[-1])
        keep_y.append(ys[-1])

    return keep_x, keep_y


def _render_bot_card(coin: str, s: dict, idx: int = 0, momentum_cfg: dict | None = None, model_cfg: dict | None = None) -> None:
    _momentum_cfg = momentum_cfg or {}
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
    norm_thresh   = _model_cfg.get("edge_threshold", 0.20)

    with st.container(border=True):
        # ── Header: coin + all config in one compact line ─────────────────────
        sig_str   = f"{sigma_ent}σ·${_fmt_coin(threshold)}" if sigma_val and sigma_ent else "—"
        ws_clock  = _fmt_ts(ws) if ws else "—"
        stale_str = f"⚠ stale {age:.0f}s" if age > STALE_SECS else f"{age:.0f}s ago"
        thresh_str = f"edge {norm_thresh}"
        momentum_enabled = _momentum_cfg.get("enabled", True)
        model_enabled = _model_cfg.get("enabled", True)
        # Status indicators: enabled = 🚀, disabled = 🔒
        momentum_status = "🚀" if momentum_enabled else "🔒"
        model_status = "🤖" if model_enabled else "🔒"
        st.markdown(
            f"**{coin}** {momentum_status} {model_status} &nbsp; "
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
        ws_now = int(time.time()) // 300 * 300
        _mom_live = s.get("momentum_enabled", _momentum_cfg.get("enabled", True))
        _mdl_live = s.get("model_enabled",    _model_cfg.get("enabled", True))
        _has_live_mom = (
            _mom_live
            and trade is not None
            and trade.get("order_id") != "DRY_RUN"
            and int(trade.get("window_start_ts") or 0) == ws_now
        )
        _has_live_mdl = _mdl_live and any(
            t.get("order_id") != "DRY_RUN"
            for t in (s.get("model_trades") or [])
            if int(t.get("window_start_ts") or 0) == ws_now
        )
        has_live_trade = _has_live_mom or _has_live_mdl

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
                xs, ys = _downsample_every_2s(xs, ys)

                line_color = "#22c55e" if has_live_trade else "#60a5fa"
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

            features = s.get("features") or {}
            caption_text = f"{dir_arrow} ${_fmt_coin(abs(coin_move))}  ·  {pct_of_thresh:.0f}% of threshold  ·  {sigmas:+.2f}σ  ·  {thresh_str}"
            if features:
                FEAT_LABELS = {
                    "move_sigmas": "move(σ)", "elapsed_second": "elapsed",
                    "hour_utc": "hour", "vel_2s": "vel2s", "vel_5s": "vel5s",
                    "vel_10s": "vel10s", "acc_4s": "acc4s", "vel_ratio": "ratio",
                    "vel_decay": "decay", "vol_10s_log": "vol10s",
                }
                feat_html = "  &nbsp;·&nbsp;  ".join(
                    _kv(FEAT_LABELS.get(k, k) or "", "—" if v is None else str(round(v, 3)))
                    for k, v in features.items()
                )
                st.markdown(
                    f"<div style='display:flex;align-items:baseline;gap:8px;font-size:12px;color:#6b7280'>"
                    f"<span>{caption_text}</span>"
                    f"<details><summary style='cursor:pointer;font-size:11px;color:#6b7280;white-space:nowrap'>features</summary>"
                    f"<div style='margin-top:4px;font-size:12px'>{feat_html}</div></details>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption(caption_text)

        # ── PM prices + model prediction in one row ───────────────────────────
        up_ref  = up_ask if up_ask is not None else up_mid
        dn_ref  = dn_ask if dn_ask is not None else dn_mid
        up_edge = (predicted_win - up_ref)         if predicted_win is not None and up_ref is not None else None
        dn_edge = ((1.0 - predicted_win) - dn_ref) if predicted_win is not None and dn_ref is not None else None
        active_thresh = norm_thresh

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

        # ── Trade status ──────────────────────────────────────────────────────
        momentum_live = _mom_live
        model_live    = _mdl_live

        def _trade_row(t: dict, label: str, is_live: bool) -> None:
            status     = t.get("status", "open")
            side       = t.get("side", "")
            fp         = t.get("fill_price", 0.0)
            fusdc      = t.get("fill_usdc", 0.0)
            pnl        = t.get("pnl_usdc")
            slip       = t.get("slippage")
            resolution = t.get("resolution")
            elapsed    = t.get("elapsed_second")
            is_dry     = t.get("order_id") == "DRY_RUN" or not is_live
            icon       = {"won": "✅", "lost": "❌", "open": "🔵", "order_failed": "⚠️", "unresolved": "❓"}.get(status, "🔵")
            took_profit = (status == "won" and resolution is not None and float(resolution) not in (0.0, 1.0))
            theme_base = st.get_option("theme.base")
            is_light = theme_base == "light"

            status_colors = {
                "won": "#15803d" if is_light else "#22c55e",
                "lost": "#b91c1c" if is_light else "#ef4444",
                "open": "#1d4ed8" if is_light else "#60a5fa",
                "order_failed": "#991b1b" if is_light else "#f87171",
                "unresolved": "#b45309" if is_light else "#f59e0b",
                "insufficient_balance_disabled": "#b45309" if is_light else "#f59e0b",
            }

            text_main = "#000000" if is_light else "#ffffff"
            text_muted = text_main
            card_bg = (
                ("#f7fff8" if is_light else "rgba(20,83,45,0.35)")
                if is_dry
                else ("#f6f9ff" if is_light else "rgba(30,58,138,0.35)")
            )
            base_border = (
                ("#86efac" if is_light else "rgba(134,239,172,0.50)")
                if is_dry
                else ("#93c5fd" if is_light else "rgba(147,197,253,0.50)")
            )
            failed_statuses = {"lost", "order_failed", "unresolved", "insufficient_balance_disabled"}
            is_failed = status in failed_statuses
            card_border = "#dc2626" if is_light and is_failed else ("#f87171" if is_failed else base_border)

            slip_html = ""
            if slip is not None:
                slip_val = float(slip)
                slip_color = "#15803d" if slip_val <= 0 else "#b91c1c"
                if not is_light:
                    slip_color = "#86efac" if slip_val <= 0 else "#fca5a5"
                slip_html = f"<span style='color:{slip_color};font-weight:800'>slip {slip_val:+.4f}</span>"

            pnl_html = ""
            if pnl is not None:
                pnl_val = float(pnl)
                pnl_color = "#15803d" if pnl_val >= 0 else "#b91c1c"
                if not is_light:
                    pnl_color = "#86efac" if pnl_val >= 0 else "#fca5a5"
                pnl_html = f"<span style='color:{pnl_color};font-weight:900'>PnL {pnl_val:+.4f}</span>"

            tp_html = ""
            if took_profit and resolution is not None:
                tp_html = f"<span style='color:{'#166534' if is_light else '#86efac'};font-weight:700'>sold @ {float(resolution):.2f}</span>"

            elapsed_html = f"t+{int(elapsed)}s" if elapsed is not None else "t+—"
            row_items = [
                f"<span style='font-weight:800;color:{text_main}'>{label}</span>",
                f"<span style='font-weight:800;color:{text_main}'>{side}</span>",
                f"<span style='color:{text_muted};font-weight:700'>@ {fp:.4f}</span>",
                f"<span style='color:{text_muted};font-weight:700'>${fusdc:.2f}</span>",
                f"<span style='color:{text_muted};font-weight:700'>{elapsed_html}</span>",
            ]
            if slip_html:
                row_items.append(slip_html)
            if pnl_html:
                row_items.append(pnl_html)
            if tp_html:
                row_items.append(tp_html)

            sep = f" <span style='color:{text_muted};font-weight:700'>•</span> "

            st.markdown(
                f"<div style='margin:3px 0 7px 0;padding:5px 8px;border:1px solid {card_border};"
                f"background:{card_bg};border-radius:8px;overflow-x:auto;white-space:nowrap;'>"
                + sep.join(row_items)
                + "</div>",
                unsafe_allow_html=True,
            )

        if trade and int(trade.get("window_start_ts") or 0) == ws_now:
            _trade_row(trade, "MOM", momentum_live)
        elif not filled:
            st.caption("Watching…")

        # ── Model trades (all this window) ────────────────────────────────────
        model_trades_this_window = [
            t for t in (s.get("model_trades") or [])
            if int(t.get("window_start_ts") or 0) == ws_now
        ]
        # Fall back to single model_trade if new field not yet in status JSON
        if not model_trades_this_window and model_trade and int(model_trade.get("window_start_ts") or 0) == ws_now:
            model_trades_this_window = [model_trade]

        for mt in model_trades_this_window:
            _trade_row(mt, "MODEL", model_live)


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


def _render_summary(trades: pd.DataFrame, model_trades: pd.DataFrame) -> None:
    def _stats(df: pd.DataFrame) -> dict:
        resolved = df[df["status"].isin(["won", "lost"])] if not df.empty else pd.DataFrame()
        total_pnl = pd.to_numeric(resolved["pnl_usdc"], errors="coerce").sum() if not resolved.empty else 0.0
        total_deployed = pd.to_numeric(resolved["fill_usdc"], errors="coerce").sum() if (not resolved.empty and "fill_usdc" in resolved.columns) else 0.0
        roi_pct = total_pnl / total_deployed * 100 if total_deployed > 0 else None
        n_won = int((resolved["status"] == "won").sum()) if not resolved.empty else 0
        n_lost = int((resolved["status"] == "lost").sum()) if not resolved.empty else 0
        win_rate = n_won / (n_won + n_lost) if (n_won + n_lost) > 0 else None

        streak_label, streak_count = _streak(df) if (not df.empty and "ts" in df.columns and "status" in df.columns) else (None, 0)
        if streak_label == "won":
            streak_str = f"🔥 {streak_count}W"
        elif streak_label == "lost":
            streak_str = f"❄️ {streak_count}L"
        else:
            streak_str = "—"

        return {
            "total_pnl": total_pnl,
            "roi_pct": roi_pct,
            "n_won": n_won,
            "n_lost": n_lost,
            "win_rate": win_rate,
            "streak": streak_str,
            "resolved": resolved,
            "slippage": df["slippage"].mean() if "slippage" in df.columns and not df.empty else None,
        }

    def _stat_card(label: str, s: dict) -> str:
        pnl = s["total_pnl"]
        pnl_color = "#22c55e" if pnl >= 0 else "#ef4444"
        pnl_str   = f"${pnl:+.4f}"
        roi_str   = f"{s['roi_pct']:+.2f}%" if s["roi_pct"] is not None else "—"
        wl_str    = f"{s['n_won']}W / {s['n_lost']}L"
        wr_str    = f"{s['win_rate']:.1%}" if s["win_rate"] is not None else "—"
        wr_color  = "#22c55e" if (s["win_rate"] or 0) >= 0.5 else "#ef4444"
        slip_str  = f"{s['slippage']:.4f}" if s["slippage"] is not None else "—"
        streak    = s["streak"]

        def stat(key: str, val: str, val_color: str = "inherit") -> str:
            return (
                f"<span style='color:#6b7280;font-size:14px'>{key}&nbsp;"
                f"<b style='color:{val_color};font-size:15px'>{val}</b></span>"
            )

        return f"""
<div style='background:rgba(128,128,128,0.1);border:1px solid rgba(128,128,128,0.18);border-radius:8px;padding:14px 18px;margin-bottom:6px'>
  <div style='font-size:12px;color:#6b7280;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px'>{label}</div>
  <div style='display:flex;gap:24px;flex-wrap:wrap;align-items:baseline'>
    <span style='font-size:26px;font-weight:700;color:{pnl_color};letter-spacing:-.01em'>{pnl_str}</span>
    {stat("ROI", roi_str)}
    {stat("W/L", wl_str)}
    {stat("Win Rate", wr_str, wr_color)}
    {stat("Streak", streak)}
    {stat("Slip", slip_str)}
  </div>
</div>"""

    mom_stats = _stats(trades)
    mdl_stats = _stats(model_trades)

    st.markdown(
        _stat_card("Momentum", mom_stats) + _stat_card("Model", mdl_stats),
        unsafe_allow_html=True,
    )

    # Last trade callout (combined)
    mom_resolved = mom_stats["resolved"].copy()
    mdl_resolved = mdl_stats["resolved"].copy()
    if not mom_resolved.empty:
        mom_resolved["strategy"] = "MOMENTUM"
    if not mdl_resolved.empty:
        mdl_resolved["strategy"] = "MODEL"
    resolved_all = pd.concat([mom_resolved, mdl_resolved], ignore_index=True)
    if not resolved_all.empty and "ts" in resolved_all.columns:
        last_row = resolved_all.sort_values("ts").iloc[-1]
        asset  = last_row.get("asset", "?")
        side   = last_row.get("side", "?")
        status = last_row.get("status", "?")
        pnl    = last_row.get("pnl_usdc")
        ts_val = last_row.get("ts")
        strat  = str(last_row.get("strategy", "?"))
        pnl_str  = f"${float(pnl):+.4f}" if pnl is not None and pd.notna(pnl) else "—"
        time_str = _fmt_ts(float(ts_val)) if ts_val is not None and pd.notna(ts_val) else "—"
        st.info(f"Last trade: {strat}  {asset} {side} {status.upper()}  {pnl_str}  @ {time_str}")


# ── Overview charts ───────────────────────────────────────────────────────────

def _render_overview_charts(trades: pd.DataFrame, model_trades: pd.DataFrame) -> None:
    """Overview charts: pnl by coin/strategy, win/loss pie, distribution, and 30s entry-bucket pnl."""
    # Combine resolved trades from both strategies
    def _resolved(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        if df.empty or "status" not in df.columns:
            return pd.DataFrame()
        r = df[df["status"].isin(["won", "lost"])].copy()
        r["strategy"] = strategy
        for col in ["pnl_usdc", "fill_price", "slippage"]:
            if col in r.columns:
                r[col] = pd.to_numeric(r[col], errors="coerce")
        return r

    mom_res = _resolved(trades, "Momentum")
    mdl_res = _resolved(model_trades, "Model")
    all_res = pd.concat([mom_res, mdl_res], ignore_index=True)

    if all_res.empty:
        return

    c1, c2, c3 = st.columns(3)

    # ── 1. PnL by coin × strategy (grouped bar) ───────────────────────────────
    with c1:
        assets = sorted(all_res["asset"].dropna().unique()) if "asset" in all_res.columns else []
        strategies = ["Momentum", "Model"]

        fig = go.Figure()
        for strat in strategies:
            sdf = all_res[all_res["strategy"] == strat]
            if sdf.empty:
                continue
            pnls = [
                float(sdf[sdf["asset"] == a]["pnl_usdc"].sum()) if a in sdf["asset"].values else 0.0
                for a in assets
            ]
            bar_colors = ["#22c55e" if p >= 0 else "#ef4444" for p in pnls]
            fig.add_trace(go.Bar(
                name=strat, x=assets, y=pnls,
                marker_color=bar_colors,
                opacity=0.85 if strat == "Momentum" else 0.55,
                hovertemplate="%{x}  $%{y:+.4f}<extra>" + strat + "</extra>",
            ))

        fig.update_layout(
            title=dict(text="PnL by Coin & Strategy", font=dict(size=12), x=0),
            barmode="group",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=220, margin=dict(t=32, b=28, l=40, r=8),
            yaxis=dict(tickformat="$.2f", gridcolor="rgba(255,255,255,0.06)", zeroline=True,
                       zerolinecolor="rgba(255,255,255,0.2)"),
            xaxis=dict(showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(size=10)),
            showlegend=len([s for s in strategies if not all_res[all_res["strategy"]==s].empty]) > 1,
        )
        st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch")

    # ── 2. Win / loss pie ─────────────────────────────────────────────────────
    with c2:
        n_won  = int((all_res["status"] == "won").sum())
        n_lost = int((all_res["status"] == "lost").sum())
        if n_won + n_lost > 0:
            fig2 = go.Figure(go.Pie(
                labels=["Won", "Lost"],
                values=[n_won, n_lost],
                marker_colors=["#22c55e", "#ef4444"],
                hole=0.55,
                textinfo="label+percent",
                textfont=dict(size=12),
                hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
            ))
            fig2.update_layout(
                title=dict(text=f"Win / Loss  ({n_won}W · {n_lost}L)", font=dict(size=12), x=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=220, margin=dict(t=32, b=8, l=8, r=8),
                showlegend=False,
                annotations=[dict(
                    text=f"{n_won/(n_won+n_lost):.0%}",
                    x=0.5, y=0.5, font_size=18, showarrow=False,
                    font_color="#22c55e" if n_won >= n_lost else "#ef4444",
                )],
            )
            st.plotly_chart(fig2, config={"displayModeBar": False}, width="stretch")

    # ── 3. PnL distribution (histogram) ──────────────────────────────────────
    with c3:
        pnl_vals = all_res[all_res["pnl_usdc"] > 0]["pnl_usdc"].dropna()
        if not pnl_vals.empty:
            # Clip at 95th percentile to suppress right-tail outliers
            _p95       = float(pnl_vals.quantile(0.98))
            _n_clipped = int((pnl_vals > _p95).sum())
            _visible   = pnl_vals[pnl_vals <= _p95]

            _pnl_range = float(_visible.max() - _visible.min())
            _bin_size  = max(0.002, min(0.01, _pnl_range / 60)) if _pnl_range > 0 else 0.005
            _bin_start = max(0.0, float(_visible.min()) - _bin_size / 2)

            fig3 = go.Figure()
            for strat, color in [("Momentum", "#3b82f6"), ("Model", "#a855f7")]:
                sdf = all_res[
                    (all_res["strategy"] == strat) &
                    (all_res["pnl_usdc"] > 0) &
                    (all_res["pnl_usdc"] <= _p95)
                ]["pnl_usdc"].dropna()
                if sdf.empty:
                    continue
                fig3.add_trace(go.Histogram(
                    x=sdf, name=strat, marker_color=color, opacity=0.7,
                    xbins=dict(start=_bin_start, size=_bin_size),
                    hovertemplate="$%{x:.3f}: %{y} trades<extra>" + strat + "</extra>",
                ))

            _clip_note = f"  (+{_n_clipped} outlier{'s' if _n_clipped != 1 else ''} > ${_p95:.3f})" if _n_clipped else ""
            fig3.update_layout(
                title=dict(text=f"Profit Distribution{_clip_note}", font=dict(size=11), x=0),
                barmode="overlay",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=220, margin=dict(t=32, b=28, l=40, r=8),
                xaxis=dict(tickformat="$.2f", showgrid=False, range=[0, _p95 + _bin_size]),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            font=dict(size=10)),
                showlegend=True,
            )
            st.plotly_chart(fig3, config={"displayModeBar": False}, width="stretch")

    # ── 4. PnL by 30s entry bucket, grouped by strategy ─────────────────────
    bucket_df = all_res.copy()
    if "elapsed_second" in bucket_df.columns:
        bucket_df["elapsed_second"] = pd.to_numeric(bucket_df["elapsed_second"], errors="coerce")
        bucket_df = bucket_df.dropna(subset=["elapsed_second", "pnl_usdc"])
        bucket_df = bucket_df[(bucket_df["elapsed_second"] >= 0) & (bucket_df["elapsed_second"] < 300)]

        if not bucket_df.empty:
            bucket_df["bucket_start"] = (bucket_df["elapsed_second"] // 30).astype(int) * 30
            bucket_df["bucket"] = bucket_df["bucket_start"].astype(int).astype(str) + "-" + (bucket_df["bucket_start"] + 29).astype(int).astype(str) + "s"

            bucket_order = [f"{b}-{b+29}s" for b in range(0, 300, 30)]
            fig4 = go.Figure()
            for strat, color in [("Momentum", "#3b82f6"), ("Model", "#a855f7")]:
                sdf = bucket_df[bucket_df["strategy"] == strat]
                if sdf.empty:
                    continue
                pnl_by_bucket = sdf.groupby("bucket", observed=True)["pnl_usdc"].sum()
                ys = [float(pnl_by_bucket.get(lbl, 0.0)) for lbl in bucket_order]
                fig4.add_trace(go.Bar(
                    name=strat,
                    x=bucket_order,
                    y=ys,
                    marker_color=color,
                    opacity=0.85,
                    hovertemplate="%{x}: $%{y:+.4f}<extra>" + strat + "</extra>",
                ))

            if fig4.data:
                fig4.add_hline(y=0, line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"))
                fig4.update_layout(
                    title=dict(text="PnL by 30s Entry Bucket (per Strategy)", font=dict(size=12), x=0),
                    barmode="group",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=260,
                    margin=dict(t=34, b=36, l=56, r=8),
                    yaxis=dict(title="PnL ($)", tickformat="$.4f", gridcolor="rgba(255,255,255,0.06)"),
                    xaxis=dict(title="Entry bucket (elapsed second)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                font=dict(size=10)),
                )
                st.plotly_chart(fig4, config={"displayModeBar": False}, width="stretch")


# ── Equity curve ──────────────────────────────────────────────────────────────

def _pnl_chart(trades: pd.DataFrame, model_trades: pd.DataFrame | None = None) -> go.Figure:
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
        fig.add_trace(go.Scatter(
            x=_norm_x(n), y=cum.tolist(),
            name="Momentum",
            mode="lines",
            line=dict(color="#3b82f6", width=2),
            hovertext=_hover(mom_resolved, n), hoverinfo="text+y",
        ))
        titles.append(f"Momentum ${cum.iloc[-1]:+.4f}")

    mdl_resolved = pd.DataFrame()
    if model_trades is not None and not model_trades.empty:
        mdl_resolved = _resolved(model_trades)
        if not mdl_resolved.empty:
            n_m   = len(mdl_resolved)
            cum_m = _cumsum(mdl_resolved)
            fig.add_trace(go.Scatter(
                x=_norm_x(n_m), y=cum_m.tolist(),
                name="Model",
                mode="lines",
                line=dict(color="#a855f7", width=2),
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
    def num(col):
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)
    def _v(series):
        v = series.mean()
        return None if pd.isna(v) else float(v)

    resolved = df[df["status"].isin(["won", "lost"])] if not df.empty else df
    n_res  = len(resolved)
    n_won  = int((resolved["status"] == "won").sum()) if n_res else 0
    pnl    = num("pnl_usdc")

    total_deployed = pd.to_numeric(resolved["fill_usdc"], errors="coerce").sum() if (n_res and "fill_usdc" in resolved.columns) else 0.0
    total_pnl_val  = float(pnl[df["status"].isin(["won", "lost"])].sum()) if n_res else 0.0
    roi_pct = total_pnl_val / total_deployed * 100 if total_deployed > 0 else None

    avg_slip_val = _v(pd.to_numeric(df["slippage"], errors="coerce")) if "slippage" in df.columns else None

    # Entry edge: max_pm_price - fill_price (momentum) or model edge column
    entry_edge = None
    if "max_pm_price" in df.columns and "fill_price" in df.columns:
        entry_edge = _v(pd.to_numeric(df["max_pm_price"], errors="coerce") - pd.to_numeric(df["fill_price"], errors="coerce"))

    # Timing — exclude dry-run rows, then drop sub-1ms noise
    _real_orders = df[df["order_id"] != "DRY_RUN"] if "order_id" in df.columns else df
    def _timing(col):
        s = pd.to_numeric(_real_orders[col], errors="coerce").dropna() if col in _real_orders.columns else pd.Series(dtype=float)
        s = s[s >= 1.0]  # < 1ms is residual noise — treat as missing
        if s.empty:
            return None, None, None
        return round(float(s.median()), 0), round(float(s.min()), 0), round(float(s.max()), 0)

    order_med, order_min, order_max   = _timing("order_ms")
    sign_med,  sign_min,  sign_max    = _timing("sign_ms")
    post_med,  post_min,  post_max    = _timing("post_ms")

    res = {
        "n": len(df),
        "n_open": int((df["status"] == "open").sum()) if not df.empty else 0,
        "n_resolved": n_res,
        "n_won": n_won,
        "win_rate": n_won / n_res if n_res else None,
        "total_pnl": total_pnl_val,
        "avg_pnl":   _v(pnl[df["status"].isin(["won","lost"])]) if n_res else None,
        "avg_usdc":  _v(num("fill_usdc")),
        "avg_slip":  avg_slip_val,
        "entry_edge": entry_edge,
        "roi_pct":   roi_pct,
        "avg_elapsed": _v(num("elapsed_second")),
        # timing
        "order_med": order_med, "order_min": order_min, "order_max": order_max,
        "sign_med":  sign_med,  "sign_min":  sign_min,  "sign_max":  sign_max,
        "post_med":  post_med,  "post_min":  post_min,  "post_max":  post_max,
    }

    for col in (extra_cols or []):
        v = _v(num(col))
        res[f"avg_{col}"] = None if v is None or pd.isna(v) else v

    # avg_pnl by direction
    if n_res and "side" in resolved.columns:
        up_res = resolved[resolved["side"].str.upper() == "UP"]
        dn_res = resolved[resolved["side"].str.upper() == "DOWN"]
        res["n_up"]       = len(up_res)
        res["n_dn"]       = len(dn_res)
        res["avg_pnl_up"] = _v(pd.to_numeric(up_res["pnl_usdc"], errors="coerce")) if not up_res.empty else None
        res["avg_pnl_dn"] = _v(pd.to_numeric(dn_res["pnl_usdc"], errors="coerce")) if not dn_res.empty else None
    else:
        res["n_up"] = res["n_dn"] = 0
        res["avg_pnl_up"] = res["avg_pnl_dn"] = None

    return res


def _render_strategy_stats(trades: pd.DataFrame, model_trades: pd.DataFrame) -> None:
    st.subheader("Strategy Stats")
    mom_col, mdl_col = st.columns(2)

    def _row(col, name, val):
        col.markdown(
            f"<div style='display:flex;justify-content:space-between'>"
            f"<span style='color:#9ca3af'>{name}</span><span>{val}</span></div>",
            unsafe_allow_html=True,
        )

    def _timing_str(med, mn, mx):
        if med is None:
            return None
        if mn is not None and mx is not None:
            return f"{med:.0f} ms  <span style='color:#6b7280;font-size:11px'>({mn:.0f}–{mx:.0f})</span>"
        return f"{med:.0f} ms"

    def _show(col, label: str, s: dict, is_model: bool = False) -> None:
        n_label = f"{s['n_resolved']} resolved"
        if s["n_open"]:
            n_label += f" · {s['n_open']} open"
        col.markdown(f"**{label}** <span style='color:#6b7280;font-size:12px'>{n_label}</span>",
                     unsafe_allow_html=True)

        # ── Performance ───────────────────────────────────────────────────────
        col.markdown("<div style='color:#6b7280;font-size:15px;margin-top:8px'>PERFORMANCE</div>",
                     unsafe_allow_html=True)
        _row(col, "Win rate", f"{s['win_rate']:.1%}  ({s['n_won']}W / {s['n_resolved'] - s['n_won']}L)" if s["win_rate"] is not None else "—")
        _row(col, "ROI", f"{s['roi_pct']:+.2f}%" if s["roi_pct"] is not None else "—")
        _row(col, "Total PnL", f"${s['total_pnl']:+.4f}")
        _row(col, "Avg PnL / trade", f"${s['avg_pnl']:+.4f}" if s["avg_pnl"] is not None else "—")
        _row(col, "Avg PnL UP", f"${s['avg_pnl_up']:+.4f} ({s['n_up']})" if s["avg_pnl_up"] is not None else "—")
        _row(col, "Avg PnL DOWN", f"${s['avg_pnl_dn']:+.4f} ({s['n_dn']})" if s["avg_pnl_dn"] is not None else "—")

        # ── Entry quality ─────────────────────────────────────────────────────
        col.markdown("<div style='color:#6b7280;font-size:15px;margin-top:8px'>ENTRY QUALITY</div>",
                     unsafe_allow_html=True)
        if is_model and s.get("avg_predicted_win") is not None:
            _row(col, "Avg P(win)", f"{s['avg_predicted_win']:.1%}")
        if is_model and s.get("avg_edge") is not None:
            edge_net = s["avg_edge"] - (s["avg_slip"] or 0)
            _row(col, "Avg edge (net slip)", f"{s['avg_edge']:+.4f}  →  {edge_net:+.4f}")
        if not is_model and s.get("entry_edge") is not None:
            _row(col, "Avg entry edge", f"{s['entry_edge']:+.4f}")
        _row(col, "Avg fill USDC", f"${s['avg_usdc']:.4f}" if s["avg_usdc"] is not None else "—")
        _row(col, "Avg slippage", f"{s['avg_slip']:+.4f}" if s["avg_slip"] is not None else "—")
        elapsed = s.get("avg_elapsed") or (s.get("avg_elapsed_second") if is_model else None)
        if elapsed is not None:
            _row(col, "Avg trigger time", f"{elapsed:.0f}s into window")

        # ── Latency ───────────────────────────────────────────────────────────
        order_str = _timing_str(s["order_med"], s["order_min"], s["order_max"])
        sign_str  = _timing_str(s["sign_med"],  s["sign_min"],  s["sign_max"])
        post_str  = _timing_str(s["post_med"],  s["post_min"],  s["post_max"])
        if any(v is not None for v in [order_str, sign_str, post_str]):
            col.markdown("<div style='color:#6b7280;font-size:15px;margin-top:8px'>LATENCY (median · min–max)</div>",
                         unsafe_allow_html=True)
        if order_str:
            _row(col, "Total order", order_str)
        if sign_str:
            label = "Sign+POST" if is_model else "Sign (fresh)"
            _row(col, label, sign_str)
        if post_str:
            _row(col, "HTTP POST (presigned)", post_str)

    if not trades.empty:
        _show(mom_col, "Momentum", _strategy_stats(trades, extra_cols=["order_ms", "post_ms", "sign_ms"]))
    else:
        mom_col.caption("No momentum trades yet.")

    if not model_trades.empty:
        _show(mdl_col, "Model",
              _strategy_stats(model_trades, extra_cols=["edge", "predicted_win", "elapsed_second", "order_ms", "sign_ms", "post_ms"]),
              is_model=True)
    else:
        mdl_col.caption("No model trades yet.")


# ── Per-coin stats ───────────────────────────────────────────────────────────

MODEL_TRADE_COLS = [
    "ts", "asset", "side", "token_id",
    "fill_price", "fill_size", "fill_usdc", "fee_usdc",
    "predicted_win", "edge",
    "elapsed_second", "coin_open", "coin_trigger", "coin_move",
    "window_start_ts", "window_end_ts",
    "sign_ms", "post_ms", "order_ms",
    "resolution", "pnl_usdc", "status", "order_id", "slippage",
]

_MODEL_NUMERIC_COLS = {
    "ts", "fill_price", "fill_size", "fill_usdc", "fee_usdc",
    "predicted_win", "edge", "elapsed_second",
    "coin_open", "coin_trigger", "coin_move",
    "window_start_ts", "window_end_ts",
    "sign_ms", "post_ms", "order_ms",
    "resolution", "pnl_usdc", "slippage",
}


def _load_model_trades() -> pd.DataFrame:
    frames = []
    for path in LIVE_DIR.glob("trades_model_*.csv"):
        try:
            df_raw = pd.read_csv(path, on_bad_lines="skip")
            for col in MODEL_TRADE_COLS:
                if col not in df_raw.columns:
                    df_raw[col] = pd.NA
            frames.append(df_raw[MODEL_TRADE_COLS])
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for col in _MODEL_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ts"])
    # Keep final status per (asset, window_start_ts) — prefer resolved over open
    def _prefer_final(group):
        finals = group[group["status"].isin(["won", "lost", "unresolved", "fok_won", "fok_lost"])]
        if not finals.empty:
            return finals.sort_values("ts").iloc[-1]
        return group.sort_values("ts").iloc[-1]
    # For each trade (same asset, window, timestamp), prefer final status (won/lost/unresolved) over open
    df = df.sort_values("ts").groupby(
        ["asset", "window_start_ts", "ts"], as_index=False
    ).apply(_prefer_final).reset_index(drop=True)
    return df.sort_values("ts", ascending=False).reset_index(drop=True)


def _render_per_coin_stats(trades: pd.DataFrame, model_trades: pd.DataFrame) -> None:
    if trades.empty or "status" not in trades.columns:
        resolved = pd.DataFrame()
    else:
        resolved = trades[trades["status"].isin(["won", "lost"])]
    model_resolved = (
        model_trades[model_trades["status"].isin(["won", "lost"])]
        if (not model_trades.empty and "status" in model_trades.columns)
        else pd.DataFrame()
    )

    if resolved.empty and model_resolved.empty:
        return

    st.subheader("Per-Coin Stats")
    assets = sorted(set(
        (resolved["asset"].unique().tolist() if not resolved.empty else []) +
        (model_resolved["asset"].unique().tolist() if not model_resolved.empty else [])
    ))
    cols = st.columns(len(assets))

    def _sum_pnl(df: pd.DataFrame) -> float:
        return float(pd.to_numeric(df["pnl_usdc"], errors="coerce").fillna(0.0).sum())

    for col, asset in zip(cols, assets):
        col.markdown(f"**{asset}**")
        if not resolved.empty and asset in resolved["asset"].values:
            adf     = resolved[resolved["asset"] == asset]
            n_won   = (adf["status"] == "won").sum()
            n_lost  = (adf["status"] == "lost").sum()
            total   = n_won + n_lost
            win_rate = n_won / total if total else 0.0
            pnl     = _sum_pnl(adf)
            col.metric("Momentum", f"${pnl:+.2f}", f"{win_rate:.0%} ({n_won}W/{n_lost}L)")
        if not model_resolved.empty and asset in model_resolved["asset"].values:
            mdf     = model_resolved[model_resolved["asset"] == asset]
            n_won   = (mdf["status"] == "won").sum()
            n_lost  = (mdf["status"] == "lost").sum()
            total   = n_won + n_lost
            win_rate = n_won / total if total else 0.0
            pnl     = _sum_pnl(mdf)
            col.metric("Model", f"${pnl:+.2f}", f"{win_rate:.0%} ({n_won}W/{n_lost}L)")


# ── Unified trade log ─────────────────────────────────────────────────────────

def _render_unified_trade_log(trades: pd.DataFrame, model_trades: pd.DataFrame) -> None:
    st.subheader("Trade Log")

    mom_rows = pd.DataFrame()
    if not trades.empty:
        mom_cols_avail = [c for c in ["ts", "asset", "side", "fill_price", "fill_usdc", "slippage", "elapsed_second", "sign_ms", "post_ms", "order_ms", "pnl_usdc", "status"] if c in trades.columns]
        mom_rows = trades[mom_cols_avail].copy()
        mom_rows["strategy"] = "MOM"
        for missing in ["edge", "predicted_win", "elapsed_second"]:
            if missing not in mom_rows.columns:
                mom_rows[missing] = None

    mdl_rows = pd.DataFrame()
    if not model_trades.empty:
        mdl_cols_avail = [c for c in ["ts", "asset", "side", "fill_price", "fill_usdc", "slippage", "edge", "predicted_win", "elapsed_second", "sign_ms", "post_ms", "order_ms", "pnl_usdc", "status"] if c in model_trades.columns]
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

    # Mask timing for dry-run rows and sub-1ms noise
    _is_dry = display["order_id"] == "DRY_RUN" if "order_id" in display.columns else pd.Series(False, index=display.index)
    for _tc in ("sign_ms", "post_ms", "order_ms"):
        if _tc in display.columns:
            _s = pd.to_numeric(display[_tc], errors="coerce")
            display[_tc] = _s.where((_s >= 1.0) & ~_is_dry, other=None)

    show_cols = ["ts", "strategy", "asset", "side", "fill_price", "fill_usdc",
                 "slippage", "edge", "predicted_win", "elapsed_second",
                 "sign_ms", "post_ms", "order_ms", "status", "pnl_usdc"]
    out = display[[c for c in show_cols if c in display.columns]]

    _STATUS_ROW_BG = {
        "won":  "rgba(34,197,94,0.15)",
        "lost": "rgba(239,68,68,0.15)",
        "open": "rgba(59,130,246,0.09)",
    }

    # Streamlit's dataframe path can choke on Arrow metadata for some numpy-backed
    # columns. Render as HTML here so the log stays stable, with per-row colors.
    filled = out.fillna("—")
    headers = "".join(f"<th>{c}</th>" for c in filled.columns)
    rows_html = []
    for _, row in filled.iterrows():
        bg = _STATUS_ROW_BG.get(str(row.get("status", "")), "")
        row_style = f' style="background-color:{bg}"' if bg else ""
        cells = "".join(f"<td>{v}</td>" for v in row.values)
        rows_html.append(f"<tr{row_style}>{cells}</tr>")
    html = f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows_html)}</tbody></table>"

    st.markdown(
        "<div style='overflow-x:auto;'>"
        "<style>"
        "table { width: 100%; border-collapse: collapse; }"
        "th, td { padding: 0.35rem 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.08); text-align: left; }"
        "th { position: sticky; top: 0; background: rgba(15,23,42,0.95); color: #e2e8f0; font-weight: 600; z-index: 1; }"
        "</style>"
        f"{html}"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

@st.fragment(run_every=2)
def render() -> None:
    bots        = _load_bots()
    trades      = _load_trades()
    model_trades = _load_model_trades()

    st.subheader("🤖 Live Momentum Bots")

    # ── Include dry-run trades toggle ─────────────────────────────────────────
    include_dry_run = st.toggle("Include dry-run trades", value=True, label_visibility="collapsed")

    # Apply dry-run filter to momentum and model trades
    def _apply_dry_run_filter(df: pd.DataFrame, include: bool) -> pd.DataFrame:
        if df.empty or "order_id" not in df.columns:
            return df
        if not include:
            return df[df["order_id"] != "DRY_RUN"].reset_index(drop=True)
        return df

    trades_filtered = _apply_dry_run_filter(trades, include_dry_run)
    model_trades_filtered = _apply_dry_run_filter(model_trades, include_dry_run)

    _FOK_STATUSES = {"fok_killed", "fok_won", "fok_lost"}

    # Exclude order_failed and fok_* rows from all main calculations and display
    def _drop_failed(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "status" not in df.columns:
            return df
        return df[~df["status"].isin({"order_failed"} | _FOK_STATUSES)].reset_index(drop=True)

    # Separate FOK rows before stripping them — used for the FOK analysis section
    def _fok_only(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "status" not in df.columns:
            return pd.DataFrame()
        return df[df["status"].isin(_FOK_STATUSES)].reset_index(drop=True)

    fok_trades       = _fok_only(trades_filtered)
    fok_model_trades = _fok_only(model_trades_filtered)
    trades_filtered       = _drop_failed(trades_filtered)
    model_trades_filtered = _drop_failed(model_trades_filtered)

    tab_ov, tab_bots, tab_log, tab_analytics = st.tabs(["Overview", "Bots", "Trade Log", "Analytics"])

    # ── Overview ──────────────────────────────────────────────────────────────
    with tab_ov:
        _render_window_bar()
        _render_summary(trades_filtered, model_trades_filtered)
        _render_overview_charts(trades_filtered, model_trades_filtered)
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
                if trade and int(trade.get("window_start_ts") or 0) == ws_now:
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
            ws_now = int(time.time()) // 300 * 300
            for i in range(0, len(bots), CARDS_PER_ROW):
                row  = bots[i : i + CARDS_PER_ROW]
                cols = st.columns(len(row))
                for j, (col, (coin, s)) in enumerate(zip(cols, row)):
                    with col:
                        # Load per-asset strategy config (merges global defaults with per-coin overrides)
                        momentum_cfg, model_cfg = _load_strategy_cfg(coin)

                        # Count active trades for this coin in current window
                        mom_active = trades_filtered[
                            (trades_filtered["asset"] == coin) &
                            (trades_filtered["window_start_ts"] == ws_now) &
                            (trades_filtered["status"] == "open")
                        ]
                        model_active = model_trades_filtered[
                            (model_trades_filtered["asset"] == coin) &
                            (model_trades_filtered["window_start_ts"] == ws_now) &
                            (model_trades_filtered["status"] == "open")
                        ]

                        _render_bot_card(coin, s, idx=i + j, momentum_cfg=momentum_cfg, model_cfg=model_cfg)

                        # Show multi-trade badges if applicable
                        if len(mom_active) > 1 or len(model_active) > 1:
                            badge_text = ""
                            if len(mom_active) > 1:
                                badge_text += f"🚀 **{len(mom_active)} momentum trades** "
                            if len(model_active) > 1:
                                badge_text += f"🤖 **{len(model_active)} model trades**"
                            st.caption(badge_text)

    # ── Trade Log ─────────────────────────────────────────────────────────────
    with tab_log:
        _render_unified_trade_log(trades_filtered, model_trades_filtered)

    # ── Analytics ─────────────────────────────────────────────────────────────
    with tab_analytics:
        # ── Time filter ───────────────────────────────────────────────────────


        # Dynamic time window: slider from 30 min up to all-time
        # Snap points give clean labels; values between snaps are also valid.
        _SNAPS = [
            (30 * 60,        "30m"),
            (1 * 3600,       "1h"),
            (2 * 3600,       "2h"),
            (4 * 3600,       "4h"),
            (8 * 3600,       "8h"),
            (12 * 3600,      "12h"),
            (24 * 3600,      "24h"),
            (48 * 3600,      "48h"),
            (7 * 24 * 3600,  "7d"),
            (0,              "All time"),
        ]
        _snap_labels = [s[1] for s in _SNAPS]
        tw_col, _ = st.columns([3, 4])
        selected_label = tw_col.select_slider(
            "Time window", options=_snap_labels, value="All time",
        )
        cutoff_secs = dict((v, k) for k, v in _SNAPS)[selected_label] or None

        def _time_filter(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty or cutoff_secs is None or "ts" not in df.columns:
                return df
            cutoff = time.time() - cutoff_secs
            return df[pd.to_numeric(df["ts"], errors="coerce") >= cutoff].reset_index(drop=True)

        at = _time_filter(trades_filtered)
        amt = _time_filter(model_trades_filtered)

        # Asset filter
        _all_assets = sorted(set(
            (at["asset"].dropna().unique().tolist() if not at.empty and "asset" in at.columns else []) +
            (amt["asset"].dropna().unique().tolist() if not amt.empty and "asset" in amt.columns else [])
        ))
        if _all_assets:
            asset_col, _ = st.columns([5, 2])
            selected_assets = asset_col.multiselect(
                "Assets", options=_all_assets, default=_all_assets, label_visibility="collapsed",
                placeholder="Select assets…",
            )
            if selected_assets and selected_assets != _all_assets:
                at  = at[at["asset"].isin(selected_assets)].reset_index(drop=True)  if not at.empty  else at
                amt = amt[amt["asset"].isin(selected_assets)].reset_index(drop=True) if not amt.empty else amt

        if not at.empty or not amt.empty:
            st.plotly_chart(_pnl_chart(at, amt), width="stretch")
            st.divider()
            _render_strategy_stats(at, amt)
            st.divider()
            _render_per_coin_stats(at, amt)
        else:
            st.caption("No trades recorded yet — analytics will appear once trades are recorded.")

        # ── FOK analysis ──────────────────────────────────────────────────────
        fok_all = pd.concat([
            fok_trades.assign(strategy="MOM") if not fok_trades.empty else pd.DataFrame(),
            fok_model_trades.assign(strategy="MODEL") if not fok_model_trades.empty else pd.DataFrame(),
        ], ignore_index=True)

        if not fok_all.empty:
            st.divider()
            st.subheader("FOK-Killed Orders — Hypothetical Analysis")
            st.caption(
                "Orders killed by the CLOB before filling. Fill price assumes +0.25 slippage. "
                "Excluded from all other statistics."
            )

            fok_resolved = fok_all[fok_all["status"].isin(["fok_won", "fok_lost"])].copy()
            fok_killed   = fok_all[fok_all["status"] == "fok_killed"]

            c1, c2, c3, c4 = st.columns(4)
            n_total   = len(fok_all)
            n_res     = len(fok_resolved)
            n_pending = len(fok_killed)
            c1.metric("Total FOK", n_total)
            c2.metric("Resolved", n_res)
            c3.metric("Pending", n_pending)

            if not fok_resolved.empty:
                pnl_col   = pd.to_numeric(fok_resolved["pnl_usdc"], errors="coerce")
                total_pnl = float(pnl_col.sum())
                c4.metric("Hypothetical PnL", f"${total_pnl:+.4f}")

                # Per-strategy breakdown
                for strat in ["MOM", "MODEL"]:
                    sdf = fok_resolved[fok_resolved["strategy"] == strat]
                    if sdf.empty:
                        continue
                    s_pnl  = pd.to_numeric(sdf["pnl_usdc"], errors="coerce")
                    s_won  = int((sdf["status"] == "fok_won").sum())
                    s_lost = int((sdf["status"] == "fok_lost").sum())
                    s_wr   = s_won / (s_won + s_lost) if (s_won + s_lost) > 0 else None
                    wr_str    = f"  ({s_wr:.1%} win)" if s_wr is not None else ""
                    pnl_sum   = s_pnl.sum()
                    pnl_avg   = s_pnl.mean()
                    pnl_color = "#22c55e" if pnl_sum >= 0 else "#ef4444"
                    pnl_sum_s = f"{pnl_sum:+.4f}".replace("-", "&#8722;")
                    pnl_avg_s = f"{pnl_avg:+.4f}".replace("-", "&#8722;")
                    st.markdown(
                        f"<span style='font-size:14px'>"
                        f"<b>{strat}</b> — {s_won}W / {s_lost}L{wr_str}"
                        f"&nbsp;&nbsp;·&nbsp; Hyp. PnL <b style='color:{pnl_color}'>{pnl_sum_s}</b>"
                        f"&nbsp;&nbsp;·&nbsp; Avg <b>{pnl_avg_s}</b> / trade"
                        f"</span>",
                        unsafe_allow_html=True,
                    )

                # Table
                show = ["ts", "strategy", "asset", "side", "fill_price", "slippage",
                        "fill_usdc", "pnl_usdc", "status"]
                fok_disp = fok_resolved[[c for c in show if c in fok_resolved.columns]].copy()
                fok_disp["ts"] = fok_disp["ts"].apply(
                    lambda t: _fmt_ts(float(t)) if pd.notna(t) else "—"
                )
                fok_disp = fok_disp.sort_values("ts", ascending=False) if "ts" in fok_disp.columns else fok_disp
                st.dataframe(fok_disp, use_container_width=True, hide_index=True)
