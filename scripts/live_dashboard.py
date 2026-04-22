#!/usr/bin/env python3
"""
Live momentum bot monitor.

Run with:
    streamlit run scripts/live_dashboard.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dashboard"))

import streamlit as st
import tab_live
import tab_discrepancy

st.set_page_config(
    page_title="Skeptic — Live Bots",
    page_icon="🤖",
    layout="wide",
)

tab_live_tab, tab_disc_tab = st.tabs(["Live", "Predicted vs Actual"])

with tab_live_tab:
    tab_live.render()

with tab_disc_tab:
    if st.button("Run Analysis", type="primary"):
        tab_discrepancy.render()
    else:
        st.caption("Click **Run Analysis** to load the Predicted vs Actual report.")
