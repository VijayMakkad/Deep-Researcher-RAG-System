"""Results display component — answer, sources, and export."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from frontend.api_client import BACKEND_LABELS
from frontend.components.charts import render_relevance_chart
from frontend.state import S


def render_results(data: dict, processing_time: float, query: str) -> None:
    """Render the full results section: metrics, answer, sources, chart, export."""
    diverse_mode = st.session_state.get(S.DIVERSE_MODE, False)
    answer = data.get("answer", "No answer provided")
    sources = data.get("sources", [])
    sources_count = len(sources)
    avg_score = sum(s.get("score", 0) for s in sources) / max(sources_count, 1)
    actual_word_count = len(answer.split()) if answer else 0
    word_limit = st.session_state.get(S.WORD_LIMIT, 0)

    # -- Metrics row --
    cols = st.columns(6)
    with cols[0]:
        st.metric("⏱️ Response Time", f"{processing_time:.2f}s")
    with cols[1]:
        st.metric("📄 Sources Found", sources_count)
    with cols[2]:
        st.metric("📊 Avg. Relevance", f"{avg_score:.3f}")
    with cols[3]:
        st.metric("🎯 Word Limit", f"~{word_limit}" if word_limit > 0 else "None")
    with cols[4]:
        st.metric("📝 Actual Words", actual_word_count)
    with cols[5]:
        st.metric("🔥 Diverse Mode", "🔥 Enabled" if diverse_mode else "❌ Disabled")

    # -- Answer --
    st.markdown("## 💡 Answer")
    answer_bg = (
        "linear-gradient(135deg, #fff5f5 0%, #ffeaa7 100%)"
        if diverse_mode
        else "linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)"
    )
    border_color = "#ff6b6b" if diverse_mode else "#667eea"

    st.markdown(
        f'<div style="background: {answer_bg}; border-left: 5px solid {border_color}; '
        f'padding: 2rem; border-radius: 10px; margin: 1rem 0;">'
        f'<div style="color: #333; font-size: 1.1rem; line-height: 1.6;">'
        f"{answer}</div></div>",
        unsafe_allow_html=True,
    )

    # -- Diverse mode info --
    if diverse_mode and sources:
        st.markdown("### 🔥 Diverse Mode Results")
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"**Diversity Threshold:** {st.session_state.get(S.DIVERSITY_THRESHOLD, 0.7)}")
        with c2:
            st.info(f"**Diversity Penalty:** {st.session_state.get(S.DIVERSITY_PENALTY, 0.3)}")

        score_std = pd.DataFrame(sources)["score"].std()
        st.success(f"**Score Diversity (Std Dev):** {score_std:.3f}")

    # -- Sources --
    if sources:
        _render_sources(sources, diverse_mode)
        render_relevance_chart(sources, diverse_mode)
        _render_export(query, answer, sources, processing_time, actual_word_count, diverse_mode)
    else:
        st.warning("📭 No relevant sources found for your query.")


def _render_sources(sources: list[dict], diverse_mode: bool) -> None:
    """Render source cards."""
    st.markdown("## 📚 Retrieved Sources")
    suffix = " (Enhanced with Diverse Mode 🔥)" if diverse_mode else ""
    st.caption(f"Showing top {len(sources)} most relevant sources{suffix}")

    show_scores = st.session_state.get(S.SHOW_SCORES, True)
    show_chunk_ids = st.session_state.get(S.SHOW_CHUNK_IDS, True)

    for idx, source in enumerate(sources, 1):
        card_border = "2px solid #ff6b6b" if diverse_mode else "2px solid #dee2e6"

        st.markdown(
            f'<div style="background: white; border: {card_border}; border-radius: 10px; '
            f'padding: 1.5rem; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns([3, 1])
        with c1:
            filename = source.get("filename", "Unknown")
            st.markdown(f"**📖 Source {idx}: {filename}**")
            if show_chunk_ids:
                st.caption(
                    f"Chunk: {source.get('chunk_idx', 'N/A')} | Page: {source.get('page', 'N/A')}"
                )
        with c2:
            if show_scores:
                score = source.get("score", 0)
                if diverse_mode:
                    color = "#ff6b6b" if score > 0.7 else "#4ecdc4" if score > 0.5 else "#fd79a8"
                else:
                    color = "#28a745" if score > 0.7 else "#ffc107" if score > 0.5 else "#dc3545"

                st.markdown(
                    f'<span style="background:{color};color:white;padding:6px 12px;'
                    f"border-radius:20px;font-size:0.85rem;font-weight:600;"
                    f'display:inline-block;">Score: {score:.3f}</span>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        snippet = source.get("snippet", "No content available")
        bg = "#fff5f5" if diverse_mode else "#f8f9fa"
        border = "#ff6b6b" if diverse_mode else "#667eea"
        st.markdown(
            f'<div style="color: #444; font-style: italic; line-height: 1.6; '
            f"padding: 1rem; background: {bg}; border-radius: 6px; "
            f'border-left: 4px solid {border};">{snippet}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div><br>", unsafe_allow_html=True)


def _render_export(
    query: str,
    answer: str,
    sources: list[dict],
    processing_time: float,
    word_count: int,
    diverse_mode: bool,
) -> None:
    """Render export buttons for markdown and CSV."""
    st.markdown("## 📥 Export Results")

    backend_key = st.session_state.get(S.CURRENT_BACKEND, "lexrank")
    backend_label = BACKEND_LABELS.get(backend_key, backend_key)
    word_limit = st.session_state.get(S.WORD_LIMIT, 0)
    top_k = st.session_state.get(S.TOP_K, 5)

    export_md = (
        f"# Deep Researcher Results\n\n"
        f"**Query:** {query}\n"
        f"**Backend:** {backend_label}\n"
        f"**Diverse Mode:** {'🔥 Enabled' if diverse_mode else '❌ Disabled'}\n"
        f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"**Processing Time:** {processing_time:.2f}s\n"
        f"**Word Limit:** {word_limit if word_limit > 0 else 'None'}\n"
        f"**Actual Words:** {word_count}\n"
        f"**Top K:** {top_k}\n\n"
        f"## Answer\n\n{answer}\n\n## Retrieved Sources\n\n"
    )
    for idx, src in enumerate(sources, 1):
        export_md += (
            f"### Source {idx}: {src.get('filename', 'Unknown')}\n"
            f"- **Chunk:** {src.get('chunk_idx', 'N/A')}\n"
            f"- **Page:** {src.get('page', 'N/A')}\n"
            f"- **Score:** {src.get('score', 0):.3f}\n"
            f"- **Content:** {src.get('snippet', '')}\n\n"
        )

    c1, c2 = st.columns(2)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "diverse_" if diverse_mode else ""

    with c1:
        st.download_button(
            "⬇️ Download Results (Markdown)",
            export_md,
            file_name=f"deep_researcher_results_{prefix}{ts}.md",
            mime="text/markdown",
        )

    with c2:
        if sources:
            df = pd.DataFrame(sources)
            df["diverse_mode"] = diverse_mode
            st.download_button(
                "⬇️ Download Sources (CSV)",
                df.to_csv(index=False),
                file_name=f"deep_researcher_sources_{prefix}{ts}.csv",
                mime="text/csv",
            )
