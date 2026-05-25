"""Sidebar component — upload, backend selection, search params."""

from __future__ import annotations

import time

import streamlit as st

from frontend.api_client import (
    BACKEND_LABELS,
    get_backend_status,
    set_backend,
    upload_files,
)
from frontend.state import S


def init_session_state() -> None:
    """Initialise all session-state keys with defaults (called once at top)."""
    defaults = {
        S.CURRENT_BACKEND: "lexrank",
        S.DIVERSE_MODE: False,
        S.SHOW_SCORES: True,
        S.SHOW_CHUNK_IDS: True,
        S.TOP_K: 5,
        S.WORD_LIMIT: 0,
        S.TIMEOUT: 120,
        S.DIVERSITY_THRESHOLD: 0.7,
        S.DIVERSITY_PENALTY: 0.3,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> None:
    """Render the full sidebar UI."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        _render_upload_section()
        st.markdown("---")
        _render_backend_status()
        _render_backend_selector()
        _render_search_params()
        _render_about()


# ---------------------------------------------------------------------------
# Sub-sections
# ---------------------------------------------------------------------------


def _render_upload_section() -> None:
    st.markdown("## 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        with st.spinner("Processing files..."):
            success, message, _ = upload_files(uploaded_files)
            if success:
                st.success(message)
            else:
                st.error(message)


def _render_backend_status() -> None:
    st.markdown("**Backend Status:**")
    status = get_backend_status()
    icon = "🟢" if status.is_connected else "🔴"
    st.markdown(icon)

    if status.is_connected:
        st.success(status.display)
        # Sync session state with actual backend
        if status.backend_key in BACKEND_LABELS:
            st.session_state[S.CURRENT_BACKEND] = status.backend_key
    else:
        st.error(status.display)
        if status.error:
            with st.expander("Connection Details"):
                st.caption(f"Error: {status.error}")

    if st.button("🔄 Refresh Status", help="Check backend status again"):
        st.rerun()


def _render_backend_selector() -> None:
    st.markdown("### 🔧 Backend Selection")

    current = st.session_state[S.CURRENT_BACKEND]
    current_index = list(BACKEND_LABELS.keys()).index(current) if current in BACKEND_LABELS else 0

    backend_choice = st.selectbox(
        "Choose Summarizer Backend",
        options=list(BACKEND_LABELS.keys()),
        index=current_index,
        format_func=lambda x: BACKEND_LABELS[x],
        help="Select which backend to use for generating answers",
    )

    if backend_choice != st.session_state[S.CURRENT_BACKEND]:
        with st.spinner(f"🔄 Switching to {BACKEND_LABELS[backend_choice]}..."):
            success, message = set_backend(backend_choice)
            if success:
                st.session_state[S.CURRENT_BACKEND] = backend_choice
                st.success(f"✅ Backend set to {backend_choice}")
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"❌ {message}")

    st.info(f"🎯 Currently using: **{BACKEND_LABELS[st.session_state[S.CURRENT_BACKEND]]}**")


def _render_search_params() -> None:
    st.markdown("### 🔧 Search Parameters")

    st.session_state[S.TOP_K] = st.slider(
        "Number of results to retrieve:",
        1,
        20,
        st.session_state[S.TOP_K],
        help="More results may provide better context but slower response",
    )

    st.session_state[S.WORD_LIMIT] = st.number_input(
        "Summarize in ~N words (optional):",
        min_value=0,
        step=10,
        value=st.session_state[S.WORD_LIMIT],
        help="Set to 0 for no word limit",
    )

    st.session_state[S.DIVERSE_MODE] = st.checkbox(
        "🔥 Enable Diverse Mode (MMR Re-ranking)",
        st.session_state[S.DIVERSE_MODE],
        help="Use Maximal Marginal Relevance for more diverse retrieval results",
    )

    if st.session_state[S.DIVERSE_MODE]:
        st.markdown(
            '<div style="background: linear-gradient(135deg, #ff6b6b, #4ecdc4); '
            'color: white; padding: 10px; border-radius: 8px; margin: 10px 0;">'
            "<strong>🔥 Diverse Mode Active</strong><br>"
            "<small>Using MMR diversity enhancement for better topic coverage</small>"
            "</div>",
            unsafe_allow_html=True,
        )

    with st.expander("🔬 Advanced Options"):
        st.session_state[S.TIMEOUT] = st.number_input(
            "Request timeout (seconds):", 30, 300, st.session_state[S.TIMEOUT]
        )
        st.session_state[S.SHOW_SCORES] = st.checkbox(
            "Show relevance scores", st.session_state[S.SHOW_SCORES]
        )
        st.session_state[S.SHOW_CHUNK_IDS] = st.checkbox(
            "Show chunk IDs", st.session_state[S.SHOW_CHUNK_IDS]
        )


def _render_about() -> None:
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        """
    This RAG (Retrieval-Augmented Generation) system searches through documents
    to provide contextual answers to your questions.

    **Features:**
    - Multiple backend support (LexRank, DistilBART, Ollama)
    - Real-time search with source transparency
    - 🔥 **Diverse Mode**: MMR-based diversity enhancement
    """
    )
