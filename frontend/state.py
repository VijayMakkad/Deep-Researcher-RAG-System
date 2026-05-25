"""Session state key constants — eliminates magic strings."""

from __future__ import annotations


class S:
    """Namespace for all ``st.session_state`` keys."""

    BACKEND = "backend"
    UPLOAD_STATUS = "upload_status"
    QUERY_HISTORY = "query_history"
    DIVERSE_MODE = "diverse_mode"
    BACKEND_STATUS_CACHE = "backend_status_cache"
    CURRENT_BACKEND = "current_backend"
    SHOW_SCORES = "show_scores"
    SHOW_CHUNK_IDS = "show_chunk_ids"
    TOP_K = "top_k"
    WORD_LIMIT = "word_limit"
    TIMEOUT = "timeout"
    DIVERSITY_THRESHOLD = "diversity_threshold"
    DIVERSITY_PENALTY = "diversity_penalty"
