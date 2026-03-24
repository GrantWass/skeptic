"""
Session data loaders for the dashboard.

Two sources:
  - Price CSVs  (collect_prices.py output)  → _load_price_sessions
  - Sessions DB (paper/live trading bot)    → _load_db_sessions
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

from skeptic import config
from skeptic.research import fetcher
from skeptic.research.fetcher import HistoricalSession


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


def get_sessions(source: str) -> dict[str, list[HistoricalSession]]:
    key = ",".join(config.ASSETS)
    if source == "prices":
        return _load_price_sessions(key)
    return _load_db_sessions(key)
