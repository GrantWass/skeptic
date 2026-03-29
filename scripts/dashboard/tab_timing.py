"""Tab — Trigger Timing Analysis: when within the 5-min window does the trigger fire, and does it matter?"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from skeptic.research.analyzer import analyze_timing_buckets
from skeptic.research.fetcher import HistoricalSession

SLIPPAGE = 0.05


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


BUCKET_SECS    = 60
BUCKET_CENTERS = [30, 90, 150, 210, 270]
BUCKET_LABELS  = ["0–60s", "60–120s", "120–180s", "180–240s", "240–300s"]
ALL_THRESHOLDS = [0.65, 0.68, 0.71, 0.74, 0.77, 0.80, 0.83, 0.86, 0.89, 0.92]


def _threshold_color(t: float) -> str:
    """Map threshold 0.65–0.95 onto a blue→red gradient."""
    frac = (t - 0.65) / (0.95 - 0.65)
    r = int(59  + frac * (239 - 59))
    g = int(130 + frac * (68  - 130))
    b = int(246 + frac * (68  - 246))
    return f"rgb({r},{g},{b})"


def _build_df(sessions: list[HistoricalSession], thresholds: list[float]) -> pd.DataFrame:
    """Run analyze_timing_buckets for every threshold and return one combined DataFrame."""
    total_sessions = len(sessions)
    rows = []
    for t in thresholds:
        df = analyze_timing_buckets(sessions, threshold=t, bucket_secs=BUCKET_SECS, slippage=SLIPPAGE)
        df["threshold"] = t
        df["break_even"] = t + SLIPPAGE
        df["bucket_center"] = BUCKET_CENTERS
        df["win_rate_vs_be"] = df["win_rate"] - df["break_even"]
        df["edge_per_session"] = df["edge_per_fill"] * df["n_fills"] / total_sessions if total_sessions > 0 else 0.0
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ── Chart builders ────────────────────────────────────────────────────────────

def _heatmap(df: pd.DataFrame, value_col: str, title: str, fmt: str, zmid: float | None = None,
             colorscale: str = "RdYlGn") -> go.Figure:
    pivot = df.pivot(index="threshold", columns="bucket_label", values=value_col)
    pivot = pivot.reindex(columns=BUCKET_LABELS)
    pivot = pivot.reindex(index=sorted(pivot.index))

    # Custom hover text
    hover = pd.DataFrame(index=pivot.index, columns=pivot.columns)
    for t in pivot.index:
        be = t + SLIPPAGE
        for b in BUCKET_LABELS:
            val = pivot.loc[t, b]
            if pd.isna(val):
                hover.loc[t, b] = "no fills"
            elif value_col == "edge_per_session":
                row = df[(df["threshold"] == t) & (df["bucket_label"] == b)]
                n = int(row["n_fills"].iloc[0]) if not row.empty else 0
                epf = row["edge_per_fill"].iloc[0] if not row.empty else None
                epf_str = f"{epf:+.4f}" if epf is not None else "—"
                hover.loc[t, b] = f"edge/session {val:+.4f}  ({n} fills, edge/fill {epf_str})"
            elif value_col == "win_rate_vs_be":
                wr = val + be  # reconstruct absolute win rate for context
                hover.loc[t, b] = f"{val:+.1%} vs B/E  (win {wr:.1%}, need {be:.0%})"
            elif value_col == "edge_per_fill":
                hover.loc[t, b] = f"edge {val:+.4f}"
            elif value_col == "n_fills":
                fill_rate = val / df[df["threshold"] == t]["n_fills"].sum() if df[df["threshold"] == t]["n_fills"].sum() > 0 else 0
                hover.loc[t, b] = f"{int(val)} fills"
            else:
                hover.loc[t, b] = f"{val:{fmt.strip('{:}')}}"

    fig = go.Figure(go.Heatmap(
        z=pivot.values.tolist(),
        x=BUCKET_LABELS,
        y=[f"T={t:.2f}" for t in pivot.index],
        colorscale=colorscale,
        zmid=zmid,
        colorbar=dict(title=title, tickformat=fmt, len=0.8),
        text=hover.values.tolist(),
        hovertemplate="<b>%{y}  |  %{x}</b><br>%{text}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13), x=0),
        xaxis=dict(title="Trigger bucket (elapsed since window open)", side="bottom"),
        yaxis=dict(title="Buy threshold", autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=max(300, len(df["threshold"].unique()) * 18 + 100),
        margin=dict(t=36, b=50, l=70, r=20),
    )
    return fig


def _win_rate_lines(df: pd.DataFrame) -> go.Figure:
    """Win rate over/under break-even per bucket (positive = above break-even)."""
    fig = go.Figure()

    # Single zero line = break-even for all thresholds
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(255,255,255,0.5)",
        line_width=1.5,
        annotation_text="break-even",
        annotation_position="top right",
        annotation_font_size=10,
        annotation_font_color="rgba(255,255,255,0.6)",
    )

    for t in sorted(df["threshold"].unique()):
        be = t + SLIPPAGE
        sub = df[df["threshold"] == t].sort_values("bucket_center")
        mask = sub["n_fills"] > 0
        color = _threshold_color(t)

        # Confidence interval band (±1 SE of proportion, shifted to vs-BE space)
        n = sub["n_fills"].values
        wr = sub["win_rate"].fillna(be).values
        se = np.where(n > 0, np.sqrt(wr * (1 - wr) / np.maximum(n, 1)), 0)
        wr_vs_be = wr - be
        upper = wr_vs_be + se
        lower = wr_vs_be - se
        xs = sub["bucket_center"].values

        fig.add_trace(go.Scatter(
            x=np.concatenate([xs, xs[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor=_hex_to_rgba(color, 0.12),
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=sub.loc[mask, "bucket_center"],
            y=sub.loc[mask, "win_rate_vs_be"],
            mode="lines+markers",
            name=f"T={t:.2f}  (B/E={be:.0%})",
            line=dict(color=color, width=2),
            marker=dict(size=sub.loc[mask, "n_fills"].clip(upper=60) / 4 + 5, color=color),
            customdata=np.stack([
                sub.loc[mask, "n_fills"],
                sub.loc[mask, "win_rate"],
                [be] * mask.sum(),
            ], axis=-1),
            hovertemplate=(
                "<b>T=%{meta}  |  bucket center %{x}s</b><br>"
                "vs break-even: <b>%{y:+.1%}</b><br>"
                "win rate: %{customdata[1]:.1%}  (need {customdata[2]:.0%})<br>"
                "fills: %{customdata[0]:.0f}<extra></extra>"
            ),
            meta=f"{t:.2f}",
        ))

    fig.update_layout(
        title=dict(text="Win Rate vs Break-Even by Trigger Time  (0 = break-even)", font=dict(size=13), x=0),
        xaxis=dict(
            title="Seconds elapsed since window open (bucket centre)",
            tickvals=BUCKET_CENTERS,
            ticktext=BUCKET_LABELS,
        ),
        yaxis=dict(title="Win Rate over/under Break-Even", tickformat="+.1%"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(t=40, b=60, l=70, r=20),
        hovermode="closest",
        legend=dict(orientation="h", y=-0.22, x=0),
    )
    return fig


def _edge_lines(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # Bucket boundary gridlines
    for boundary in [60, 120, 180, 240]:
        fig.add_vline(
            x=boundary,
            line_color="rgba(255,255,255,0.12)",
            line_width=1,
            line_dash="dot",
        )

    # Zero / break-even line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(255,255,255,0.5)",
        line_width=1.5,
        annotation_text="break-even",
        annotation_position="top right",
        annotation_font_size=10,
        annotation_font_color="rgba(255,255,255,0.6)",
    )

    for t in sorted(df["threshold"].unique()):
        sub = df[df["threshold"] == t].sort_values("bucket_center")
        mask = sub["n_fills"] > 0
        color = _threshold_color(t)

        fig.add_trace(go.Scatter(
            x=sub.loc[mask, "bucket_center"],
            y=sub.loc[mask, "edge_per_fill"],
            mode="lines+markers",
            name=f"T={t:.2f}",
            line=dict(color=color, width=2.5),
            marker=dict(
                size=sub.loc[mask, "n_fills"].clip(upper=500) / 30 + 7,
                color=color,
                line=dict(color="rgba(0,0,0,0.4)", width=1),
            ),
            customdata=np.stack([
                sub.loc[mask, "win_rate"],
                sub.loc[mask, "n_fills"],
                [t + SLIPPAGE] * mask.sum(),
            ], axis=-1),
            hovertemplate=(
                "<b>T=%{meta}  |  %{x}s into window</b><br>"
                "Edge/fill: <b>%{y:+.4f}</b><br>"
                "Win rate: %{customdata[0]:.1%}  "
                "(B/E = %{customdata[2]:.0%})<br>"
                "Fills: %{customdata[1]:.0f}<extra></extra>"
            ),
            meta=f"{t:.2f}",
        ))

    # Bucket labels along the top
    for center, label in zip(BUCKET_CENTERS, BUCKET_LABELS):
        fig.add_annotation(
            x=center, y=1.04, xref="x", yref="paper",
            text=label, showarrow=False,
            font=dict(size=10, color="rgba(255,255,255,0.5)"),
            align="center",
        )

    fig.update_layout(
        title=dict(
            text="Edge / Fill  by Trigger Time — all thresholds",
            font=dict(size=14), x=0,
        ),
        xaxis=dict(
            title="Seconds elapsed since window open",
            tickvals=[0, 60, 120, 180, 240, 300],
            ticktext=["0s", "60s", "120s", "180s", "240s", "300s"],
            range=[-5, 305],
            showgrid=False,
        ),
        yaxis=dict(
            title="Edge / Fill",
            tickformat="+.3f",
            zeroline=False,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=480,
        margin=dict(t=55, b=60, l=75, r=20),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.16, x=0),
    )
    return fig


def _fill_dist_bars(df: pd.DataFrame) -> go.Figure:
    """Stacked bar showing how fills distribute across buckets per threshold."""
    fig = go.Figure()
    for t in sorted(df["threshold"].unique()):
        sub = df[df["threshold"] == t].sort_values("bucket_center")
        color = _threshold_color(t)
        fig.add_trace(go.Bar(
            x=BUCKET_LABELS,
            y=sub["n_fills"].values,
            name=f"T={t:.2f}",
            marker_color=color,
            opacity=0.8,
            hovertemplate="<b>T=%{meta}  |  %{x}</b><br>%{y} fills<extra></extra>",
            meta=f"{t:.2f}",
        ))

    fig.update_layout(
        title=dict(text="Fill Count Distribution by Trigger Bucket", font=dict(size=13), x=0),
        barmode="group",
        xaxis=dict(title="Trigger bucket"),
        yaxis=dict(title="Number of fills"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=max(300, len(df["threshold"].unique()) * 18 + 100),
        margin=dict(t=40, b=50, l=60, r=20),
        legend=dict(orientation="h", y=-0.22, x=0),
    )
    return fig


def _summary_table(df: pd.DataFrame, total_sessions: int) -> pd.DataFrame:
    rows = []
    for t in sorted(df["threshold"].unique()):
        be = t + SLIPPAGE
        sub = df[df["threshold"] == t]
        total_fills = sub["n_fills"].sum()
        # Overall win rate (weighted by fills)
        total_wins = sub["n_wins"].sum()
        overall_wr = total_wins / total_fills if total_fills > 0 else None
        overall_edge_fill = ((overall_wr - be) if overall_wr is not None else None)
        overall_edge_session = (overall_edge_fill * total_fills / total_sessions
                                if overall_edge_fill is not None and total_sessions > 0 else None)

        best_row = sub[sub["n_fills"] > 0].sort_values("win_rate", ascending=False)
        best_bucket = best_row.iloc[0]["bucket_label"] if not best_row.empty else "—"
        best_wr     = best_row.iloc[0]["win_rate"]     if not best_row.empty else None

        rows.append({
            "Threshold": f"{t:.2f}",
            "B/E Win%": f"{be:.0%}",
            "Total Fills": int(total_fills),
            "Fill Rate": f"{total_fills / total_sessions:.1%}" if total_sessions > 0 else "—",
            "Overall Win%": f"{overall_wr:.1%}" if overall_wr is not None else "—",
            "Edge/Fill": f"{overall_edge_fill:+.4f}" if overall_edge_fill is not None else "—",
            "Edge/Session": f"{overall_edge_session:+.4f}" if overall_edge_session is not None else "—",
            "Best Bucket": best_bucket,
            "Best Win%": f"{best_wr:.1%}" if best_wr is not None else "—",
        })
    return pd.DataFrame(rows)


# ── Main render ───────────────────────────────────────────────────────────────

def render(
    all_sessions: dict[str, list[HistoricalSession]],
    selected_assets: list[str],
) -> None:
    st.subheader("⏱️ Trigger Timing Analysis")
    st.caption(
        "For each buy threshold, shows what happens when the trigger fires in each 60-second "
        "bucket of the 5-minute window. Does it matter *when* you get filled?"
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        asset = st.selectbox("Asset", selected_assets, key="tt_asset")
    with col_b:
        min_fills = st.number_input(
            "Min fills to show", min_value=1, max_value=50, value=5, step=1, key="tt_minfills",
            help="Hide buckets with fewer than this many fills",
        )

    thresholds = ALL_THRESHOLDS

    sessions = all_sessions.get(asset, [])
    if not sessions:
        st.info(f"No sessions for {asset}.")
        return

    df = _build_df(sessions, thresholds)
    if df.empty:
        st.warning("No data to display.")
        return

    # Apply min-fills mask for display (keep underlying data intact)
    df_display = df.copy()
    df_display.loc[df_display["n_fills"] < min_fills, ["win_rate", "edge_per_fill"]] = None

    total_sessions = len(sessions)

    # ── Edge/Fill vs Timing — hero chart ─────────────────────────────────────
    st.plotly_chart(_edge_lines(df_display), width="stretch")

    st.divider()

    # ── Summary metrics row ───────────────────────────────────────────────────
    st.markdown("#### Summary")
    summary_df = _summary_table(df, total_sessions)

    def _colour_edge(val: str) -> str:
        try:
            v = float(val)
            if v > 0:
                return "color: #22c55e"
            elif v < 0:
                return "color: #ef4444"
        except (ValueError, TypeError):
            pass
        return ""

    st.dataframe(
        summary_df.style
            .applymap(_colour_edge, subset=["Edge/Fill", "Edge/Session"])
            .set_properties(**{"text-align": "center"}),
        width="stretch",
        hide_index=True,
    )

    st.divider()

    # ── Heatmaps (side by side) ───────────────────────────────────────────────
    st.markdown("#### Heatmaps")
    st.caption("Green = profitable, Red = negative. Edge/Session accounts for fill frequency — a bucket that rarely triggers has less impact even at high edge/fill.")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            _heatmap(df_display, "edge_per_session", "Edge/Session by Bucket", "+.4f", zmid=0.0),
            width="stretch",
        )
    with col2:
        st.plotly_chart(
            _heatmap(df_display, "edge_per_fill", "Edge/Fill by Bucket", "+.3f", zmid=0.0),
            width="stretch",
        )

    # Fill count heatmap (full width)
    st.plotly_chart(
        _heatmap(df, "n_fills", "Fill Count by Bucket", "d", zmid=None, colorscale="Blues"),
        width="stretch",
    )

    st.divider()

    # # ── Line charts ───────────────────────────────────────────────────────────
    # st.markdown("#### Win Rate vs Trigger Time")
    # st.caption(
    #     "Each line is a threshold. Dashed horizontal = break-even for that threshold. "
    #     "Shaded band = ±1 standard error. Marker size ∝ fill count."
    # )
    # st.plotly_chart(_win_rate_lines(df_display), width="stretch")

    # st.divider()

    # ── Fill distribution ─────────────────────────────────────────────────────
    st.markdown("#### Fill Distribution")
    st.caption("Where do fills concentrate within the window? Early fills at high thresholds are rare.")
    st.plotly_chart(_fill_dist_bars(df), width="stretch")

    st.divider()