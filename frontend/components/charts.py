"""Chart component — relevance score visualisation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def render_relevance_chart(sources: list[dict], diverse_mode: bool = False) -> None:
    """Render a bar chart of source relevance scores."""
    if not sources:
        return

    st.markdown("## 📊 Source Relevance Analysis")

    df = pd.DataFrame(sources)
    fig, ax = plt.subplots(figsize=(12, 7))

    if diverse_mode:
        palette = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#f39c12", "#e74c3c", "#9b59b6"]
        colors = [palette[i % len(palette)] for i in range(len(df))]
    else:
        colors = [
            "#28a745" if s > 0.7 else "#ffc107" if s > 0.5 else "#dc3545" for s in df["score"]
        ]

    # Create labels from filename + chunk index
    labels = [
        f"{row.get('filename', '?')[:15]}…c{row.get('chunk_idx', i)}"
        for i, row in enumerate(sources)
    ]

    bars = ax.bar(labels, df["score"], color=colors)

    ax.set_xlabel("Source Chunk", fontsize=12)
    ax.set_ylabel("Relevance Score", fontsize=12)

    title = "Retrieved Sources Relevance Scores"
    if diverse_mode:
        title += " (🔥 Diverse Mode)"

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(1.0, df["score"].max() * 1.1))

    for bar, score in zip(bars, df["score"]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
