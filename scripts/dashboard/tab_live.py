"""
Tab 5 — Live: collector controls, real-time price table, and paper trading.

Paper trading state machine:
  fill_window → watching (on fill within 60s) or no_fill (after 60s)
  Each window is finalized on the next window transition.

All display widgets are wrapped in an st.empty() placeholder so that
st.session_state.update() calls (which trigger extra reruns) replace
content atomically rather than appending duplicate widgets.
"""
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from skeptic import config
from collector import LiveCollector


def render(
    collector: LiveCollector,
    buy: float,
    sell: float,
    capital: float,
    position_pct: float,
) -> None:
    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl_left, ctrl_right = st.columns([3, 1])
    with ctrl_left:
        if not collector.running:
            live_assets = st.multiselect(
                "Assets to collect", config.ASSETS, default=config.ASSETS, key="live_assets"
            )
    with ctrl_right:
        st.write("")  # vertical spacing
        if collector.running:
            if st.button("⏹ Stop", key="live_stop"):
                collector.stop()
                st.rerun()
        else:
            if st.button("▶ Start collecting", type="primary", key="live_start"):
                collector.start(live_assets)
                st.rerun()

    if collector.error:
        st.error(f"Collector error: {collector.error}")

    st.divider()

    # ── Window status ──────────────────────────────────────────────────────────
    if collector.status != "Stopped":
        st.caption(f"**Status:** {collector.status}")

    if collector.window_ts:
        wc1, wc2, wc3, wc4 = st.columns(4)
        wc1.metric(
            "Window",
            datetime.fromtimestamp(collector.window_ts, tz=timezone.utc).strftime("%H:%M UTC"),
        )
        wc2.metric("Elapsed", f"{collector.window_elapsed}s / 300s")
        wc3.metric("Remaining", f"{collector.window_remaining}s")
        wc4.metric("Rows written", collector.rows_this_window)

        progress = min(collector.window_elapsed / config.WINDOW_SECS, 1.0)
        st.progress(progress)
        if 0 < collector.window_elapsed <= config.MONITOR_SECS:
            st.info(f"🟡 **Fill monitoring window** — first {config.MONITOR_SECS} seconds")

    # ── Live price table ───────────────────────────────────────────────────────
    if collector.prices:
        price_rows = []
        for asset in sorted(collector.prices):
            p = collector.prices[asset]
            up = p.get("up")
            dn = p.get("down")
            near_buy_up = up is not None and up <= buy
            near_buy_dn = dn is not None and dn <= buy
            hit_sell_up = up is not None and up >= sell
            hit_sell_dn = dn is not None and dn >= sell

            def _fmt(v, hit_sell, near_buy):
                if v is None:
                    return "—"
                tag = " 🟢" if hit_sell else (" 🟡" if near_buy else "")
                return f"{v:.4f}{tag}"

            price_rows.append({
                "Asset": asset,
                "UP":   _fmt(up, hit_sell_up, near_buy_up),
                "DOWN": _fmt(dn, hit_sell_dn, near_buy_dn),
            })
        st.dataframe(
            pd.DataFrame(price_rows).set_index("Asset"),
            width="stretch",
        )
        st.caption(f"🟡 at or below buy ({buy:.2f})   🟢 at or above sell ({sell:.2f})")

    # ── Paper trading ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📝 Paper Trading")

    _pt_defaults: dict = {
        "pt_active": False,
        "pt_capital": capital,
        "pt_initial_capital": capital,
        "pt_results": [],
        "pt_window_ts": None,
        "pt_filled": False,
        "pt_filled_side": None,
        "pt_fill_elapsed": None,
        "pt_sell_hit": False,
        "pt_sell_elapsed": None,
        "pt_phase": "fill_window",
        "pt_last_prices": {},
        "pt_history": [],
    }
    for _k, _v in _pt_defaults.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # Controls row
    pt_ctrl_l, pt_ctrl_r = st.columns([4, 1])
    with pt_ctrl_l:
        if not st.session_state["pt_active"]:
            st.selectbox("Asset to paper trade", config.ASSETS, key="pt_asset")
        else:
            pt_display_asset = st.session_state.get("pt_asset", config.ASSETS[0])
            st.write(
                f"**Tracking:** {pt_display_asset}  |  "
                f"buy ≤ {buy:.2f}  |  sell ≥ {sell:.2f}  |  "
                f"position {position_pct:.0%}"
            )
    with pt_ctrl_r:
        st.write("")
        if st.session_state["pt_active"]:
            if st.button("⏹ Stop", key="pt_stop"):
                st.session_state["pt_active"] = False
                st.rerun()
        else:
            if not collector.running:
                st.button("▶ Start", disabled=True, key="pt_start_off",
                          help="Start the collector first")
            elif st.button("▶ Start", type="primary", key="pt_start"):
                st.session_state.update({
                    "pt_active": True,
                    "pt_capital": capital,
                    "pt_initial_capital": capital,
                    "pt_results": [],
                    "pt_window_ts": None,
                    "pt_filled": False,
                    "pt_filled_side": None,
                    "pt_fill_elapsed": None,
                    "pt_sell_hit": False,
                    "pt_sell_elapsed": None,
                    "pt_phase": "fill_window",
                    "pt_last_prices": {},
                    "pt_history": [],
                })
                st.rerun()

    if not st.session_state["pt_active"]:
        return

    if not collector.running:
        st.warning("Collector stopped — paper trading paused.")
        return

    pt_asset      = st.session_state.get("pt_asset", config.ASSETS[0])
    pt_capital    = st.session_state["pt_capital"]
    position_usdc = pt_capital * position_pct
    shares        = position_usdc / buy if buy > 0 else 0.0

    # ── Tick the state machine ─────────────────────────────────────────────────
    current_wts  = collector.window_ts
    prev_wts     = st.session_state["pt_window_ts"]
    elapsed      = collector.window_elapsed
    asset_prices = collector.prices.get(pt_asset, {})
    up_price     = asset_prices.get("up")
    down_price   = asset_prices.get("down")

    # Update last-known prices and history in-place — in-place mutations
    # are not detected by Streamlit as state changes, so no extra rerun is triggered.
    if current_wts == prev_wts:
        st.session_state["pt_last_prices"].clear()
        st.session_state["pt_last_prices"].update(asset_prices)
        if up_price is not None or down_price is not None:
            st.session_state["pt_history"].append((elapsed, up_price, down_price))

    # Window transition: finalise the just-completed window
    if current_wts and current_wts != prev_wts:
        if prev_wts is not None:
            _filled      = st.session_state["pt_filled"]
            _filled_side = st.session_state["pt_filled_side"]
            _sell_hit    = st.session_state["pt_sell_hit"]
            _last        = st.session_state["pt_last_prices"]
            _pnl         = 0.0
            _outcome     = "no fill"

            if _filled:
                if _sell_hit:
                    _pnl     = shares * (sell - buy)
                    _outcome = "sell hit"
                else:
                    _side_key = "up" if _filled_side == "UP" else "down"
                    _side_px  = _last.get(_side_key)
                    if _side_px is not None and _side_px >= 0.9:
                        _pnl     = shares * (1.0 - buy)
                        _outcome = "res win"
                    elif _side_px is not None and _side_px <= 0.1:
                        _pnl     = -(position_usdc)
                        _outcome = "res loss"
                    else:
                        _pnl     = 0.0
                        _outcome = "res unclear"

            st.session_state["pt_capital"] += _pnl
            st.session_state["pt_results"].append({
                "window_ts":    prev_wts,
                "filled":       _filled,
                "filled_side":  _filled_side,
                "fill_elapsed": st.session_state.get("pt_fill_elapsed"),
                "sell_hit":     _sell_hit,
                "sell_elapsed": st.session_state.get("pt_sell_elapsed"),
                "outcome":      _outcome,
                "pnl":          _pnl,
                "capital_after": st.session_state["pt_capital"],
            })

        # Reset for the new window
        st.session_state.update({
            "pt_window_ts":    current_wts,
            "pt_filled":       False,
            "pt_filled_side":  None,
            "pt_fill_elapsed": None,
            "pt_sell_hit":     False,
            "pt_sell_elapsed": None,
            "pt_phase":        "fill_window",
            "pt_last_prices":  {},
            "pt_history":      [],
        })

    # State machine transitions
    _phase  = st.session_state["pt_phase"]
    _filled = st.session_state["pt_filled"]

    if _phase == "fill_window":
        if elapsed > config.MONITOR_SECS:
            st.session_state["pt_phase"] = "no_fill"
        elif not _filled:
            _up_hit   = up_price  is not None and up_price  <= buy
            _down_hit = down_price is not None and down_price <= buy
            if _up_hit or _down_hit:
                if _up_hit and _down_hit:
                    _side = "UP" if (up_price or 1.0) <= (down_price or 1.0) else "DOWN"
                else:
                    _side = "UP" if _up_hit else "DOWN"
                st.session_state.update({
                    "pt_filled":       True,
                    "pt_filled_side":  _side,
                    "pt_fill_elapsed": elapsed,
                    "pt_phase":        "watching",
                })
                _filled = True

    if (_phase == "watching" or _filled) and not st.session_state["pt_sell_hit"]:
        _filled_side = st.session_state["pt_filled_side"]
        _monitor_px  = up_price if _filled_side == "UP" else down_price
        if _monitor_px is not None and _monitor_px >= sell:
            st.session_state.update({
                "pt_sell_hit":     True,
                "pt_sell_elapsed": elapsed,
            })

    # ── Display (wrapped in st.empty so reruns replace content, not append) ────
    _phase       = st.session_state["pt_phase"]
    _filled      = st.session_state["pt_filled"]
    _filled_side = st.session_state["pt_filled_side"]
    _sell_hit    = st.session_state["pt_sell_hit"]
    _pt_capital  = st.session_state["pt_capital"]
    _total_pnl   = _pt_capital - st.session_state["pt_initial_capital"]

    if "pt_display" not in st.session_state:
        st.session_state["pt_display"] = st.empty()
    _pt_display = st.session_state["pt_display"]

    with _pt_display.container():
        pw1, pw2, pw3, pw4 = st.columns(4)
        pw1.metric("Capital", f"${_pt_capital:,.2f}", delta=f"${_total_pnl:+.2f}")

        if _filled:
            _fe = st.session_state.get("pt_fill_elapsed", 0)
            pw2.metric("Position", f"{_filled_side} @ t+{_fe}s")
        elif _phase == "fill_window":
            pw2.metric("Fill window", f"{elapsed}s / {config.MONITOR_SECS}s")
        else:
            pw2.metric("Fill window", "No fill")

        if _sell_hit:
            _se  = st.session_state.get("pt_sell_elapsed", 0)
            _est = shares * (sell - buy)
            pw3.metric("Sell", f"✓ t+{_se}s", delta=f"+${_est:.2f}")
        elif _filled:
            _mk  = "up" if _filled_side == "UP" else "down"
            _mpx = collector.prices.get(pt_asset, {}).get(_mk)
            pw3.metric(f"Watching ≥ {sell:.2f}", f"{_mpx:.4f}" if _mpx else "—")
        else:
            pw3.metric("Sell", "—")

        pw4.metric("Windows", len(st.session_state["pt_results"]))

        # Completed windows table
        _results = st.session_state["pt_results"]
        if _results:
            _tbl_rows = []
            for _r in reversed(_results):
                _ts = datetime.fromtimestamp(_r["window_ts"], tz=timezone.utc).strftime("%H:%M")
                _fe_str = f"t+{_r['fill_elapsed']}s" if _r["fill_elapsed"] is not None else "—"
                _se_str = f"t+{_r['sell_elapsed']}s" if _r.get("sell_elapsed") is not None else "—"
                _tbl_rows.append({
                    "Window":   _ts,
                    "Side":     _r["filled_side"] or "—",
                    "Filled @": _fe_str,
                    "Sell @":   _se_str if _r["sell_hit"] else "—",
                    "Outcome":  _r["outcome"],
                    "P&L":      f"${_r['pnl']:+.2f}" if _r["filled"] else "—",
                    "Capital":  f"${_r['capital_after']:,.2f}",
                })
            st.dataframe(pd.DataFrame(_tbl_rows), hide_index=True, width="stretch")

        # Live chart
        _pt_chart_history = st.session_state.get("pt_history", [])
        _fig_live = go.Figure()
        if _pt_chart_history:
            _adf = pd.DataFrame(_pt_chart_history, columns=["elapsed", "up", "down"])
            _fig_live.add_trace(go.Scatter(
                x=_adf["elapsed"], y=_adf["up"].dropna(), name="UP",
                line=dict(color="#3b82f6", width=2), mode="lines",
            ))
            _fig_live.add_trace(go.Scatter(
                x=_adf["elapsed"], y=_adf["down"].dropna(), name="DOWN",
                line=dict(color="#f97316", width=2), mode="lines",
            ))
        _fig_live.add_hline(
            y=buy, line_dash="dash", line_color="#f59e0b", line_width=1.5,
            annotation_text=f"buy {buy:.2f}", annotation_position="bottom right",
        )
        _fig_live.add_hline(
            y=sell, line_dash="dash", line_color="#22c55e", line_width=1.5,
            annotation_text=f"sell {sell:.2f}", annotation_position="top right",
        )
        _fig_live.add_vline(
            x=60, line_dash="dot", line_color="black", line_width=1,
            annotation_text="1m", annotation_position="top",
        )
        _fig_live.update_layout(
            title=dict(
                text=f"<b>{pt_asset}</b> — current window ({len(_pt_chart_history)} pts)",
                font=dict(size=12), x=0,
            ),
            xaxis=dict(title="seconds", range=[0, 300], dtick=60),
            yaxis=dict(range=[0, 1], tickformat=".2f"),
            height=320,
            margin=dict(t=36, b=40, l=40, r=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(_fig_live, width="stretch", key="pt_live_chart")
