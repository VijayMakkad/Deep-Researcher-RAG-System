"""Deep Researcher — Streamlit Frontend.

Run with::

    streamlit run frontend/app.py
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from frontend.api_client import BACKEND_LABELS, query_documents
from frontend.components.results import render_results
from frontend.components.sidebar import init_session_state, render_sidebar
from frontend.state import S

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Deep Researcher — RAG Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load custom CSS
# ---------------------------------------------------------------------------

_css_path = Path(__file__).parent / "style.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Initialise session state (fixes bug #6 — all state defined upfront)
# ---------------------------------------------------------------------------

init_session_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

render_sidebar()

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.markdown('<h1 class="main-header">🧠 Deep Researcher</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Intelligent Document Search & Question Answering</p>',
    unsafe_allow_html=True,
)

# Backend status banner
current_backend = st.session_state.get(S.CURRENT_BACKEND, "lexrank")
current_label = BACKEND_LABELS.get(current_backend, "Unknown")
diverse_indicator = ""
if st.session_state.get(S.DIVERSE_MODE, False):
    diverse_indicator = '<span class="diverse-mode-enabled">🔥 Diverse Mode</span>'

st.markdown(
    f'<div class="backend-status">Active Backend: {current_label}{diverse_indicator}</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Search interface
# ---------------------------------------------------------------------------

col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input(
        "🔍 What would you like to research?",
        placeholder="e.g., What are the latest developments in artificial intelligence?",
        label_visibility="collapsed",
    )
with col2:
    search_button = st.button("🚀 Search", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Search results
# ---------------------------------------------------------------------------

if search_button:
    if not query or not query.strip():
        st.warning("⚠️ Please enter a research question to get started.")
    else:
        diverse_mode = st.session_state.get(S.DIVERSE_MODE, False)
        status_msg = "with Diverse Mode 🔥" if diverse_mode else ""

        with st.spinner(f"🔍 Searching through documents {status_msg}..."):
            start_time = time.time()

            success, data, error = query_documents(
                query=query,
                top_k=st.session_state.get(S.TOP_K, 5),
                max_words=st.session_state.get(S.WORD_LIMIT, 0) or None,
                backend=current_backend,
                diverse=diverse_mode,
                timeout=st.session_state.get(S.TIMEOUT, 120),
            )

            processing_time = time.time() - start_time

        if success and data:
            render_results(data, processing_time, query)
        else:
            st.error(f"❌ {error}")
            if diverse_mode:
                st.info("💡 Try disabling Diverse Mode or adjusting parameters.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
_, center, _ = st.columns([1, 2, 1])
with center:
    footer = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    if st.session_state.get(S.DIVERSE_MODE, False):
        footer += " • 🔥 Diverse Mode Active"
    st.markdown(
        f'<div style="text-align: center; color: #666; font-size: 0.9rem;">{footer}</div>',
        unsafe_allow_html=True,
    )
