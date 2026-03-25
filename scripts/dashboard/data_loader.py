"""
Session data loader for the dashboard — Price CSVs only.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

from skeptic import config
from skeptic.research import fetcher
from skeptic.research.fetcher import HistoricalSession


@st.cache_resource(show_spinner="Loading price data…")
def get_sessions() -> dict[str, list[HistoricalSession]]:
    return fetcher.load_from_price_files(config.ASSETS)
