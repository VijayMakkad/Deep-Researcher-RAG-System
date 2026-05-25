"""FAISS retrieval service — metadata-aware with MMR re-ranking.

Fixes from old ``retreiver.py``:
- Filename typo corrected.
- ``doc_id`` is now the real filename from metadata, not ``doc_{i//10}``.
- Accepts ``(text, metadata)`` pairs via :meth:`build_index`.
- Maximal Marginal Relevance (MMR) re-ranking for diverse retrieval.
"""

from __future__ import annotations

from collections import Counter

import faiss
import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from backend.core.config import settings

log = structlog.get_logger()


class Retriever:
    """Manages a FAISS flat-L2 index over sentence-transformer embeddings."""

    def __init__(self, model_name: str | None = None) -> None:
        model_name = model_name or settings.embedding_model
        log.info("loading_embedding_model", model=model_name)
        self.model = SentenceTransformer(model_name)
        self.index: faiss.IndexFlatL2 | None = None
        self.chunks: list[dict] = []  # each item: {"text": str, "metadata": dict}
        self._embeddings: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[dict]) -> None:
        """Build a FAISS index from chunks with metadata.

        Each chunk must have the shape::

            {"text": str, "metadata": {"filename": str, "page": int|None, "chunk_idx": int}}
        """
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.chunks = chunks
        self._embeddings = embeddings
        log.info(
            "index_built",
            chunk_count=len(texts),
            embedding_dim=dim,
            embedding_model=settings.embedding_model,
        )

    def add_to_index(self, chunks: list[dict]) -> None:
        """Add more chunks to an existing index."""
        if self.index is None:
            self.build_index(chunks)
            return

        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self._embeddings = (
            np.vstack([self._embeddings, embeddings])
            if self._embeddings is not None
            else embeddings
        )
        log.info("index_updated", added=len(texts), total=len(self.chunks))

    def clear_index(self) -> None:
        """Remove all documents from the index."""
        self.index = None
        self.chunks = []
        self._embeddings = None
        log.info("index_cleared")

    @property
    def is_ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        q: str,
        top_k: int = 5,
        diverse: bool = False,
        lambda_param: float = 0.7,
    ) -> list[dict]:
        """Search the index for chunks relevant to query *q*.

        Returns a list of dicts with keys: ``score``, ``text``, ``metadata``.
        If *diverse* is ``True``, results are re-ranked using MMR.
        """
        if self.index is None:
            return []

        q_emb = self.model.encode([q], convert_to_numpy=True)

        if diverse and self._embeddings is not None:
            return self._mmr_rerank(q_emb, top_k=top_k, lambda_param=lambda_param)

        # Standard nearest-neighbour search
        fetch_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(q_emb, fetch_k)

        results: list[dict] = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append(
                    {
                        "score": float(np.exp(-dist)),  # distance → similarity-like score
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                    }
                )
        return results

    def get_document_stats(self) -> list[dict]:
        """Return per-document chunk counts."""
        counter: Counter[str] = Counter()
        for chunk in self.chunks:
            counter[chunk["metadata"]["filename"]] += 1
        return [{"filename": fn, "chunk_count": cnt} for fn, cnt in counter.items()]

    # ------------------------------------------------------------------
    # MMR re-ranking
    # ------------------------------------------------------------------

    def _mmr_rerank(
        self,
        query_emb: np.ndarray,
        top_k: int = 5,
        lambda_param: float = 0.7,
    ) -> list[dict]:
        """Maximal Marginal Relevance for diversity-aware retrieval.

        Fetches ``top_k * 3`` candidates then re-ranks to balance relevance
        and diversity.
        """
        assert self.index is not None
        assert self._embeddings is not None

        fetch_k = min(top_k * 3, self.index.ntotal)
        distances, indices = self.index.search(query_emb, fetch_k)

        # Build candidate pool
        candidates: list[dict] = []
        candidate_embeddings: list[np.ndarray] = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                candidates.append(
                    {
                        "score": float(np.exp(-dist)),
                        "text": self.chunks[idx]["text"],
                        "metadata": self.chunks[idx]["metadata"],
                        "_idx": int(idx),
                    }
                )
                candidate_embeddings.append(self._embeddings[idx])

        if not candidates:
            return []

        candidate_embs = np.array(candidate_embeddings)
        query_vec = query_emb[0]

        # Normalise for cosine similarity
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        cand_norms = candidate_embs / (
            np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-10
        )

        # Relevance to query
        relevance = cand_norms @ query_norm  # (n_candidates,)

        selected: list[int] = []
        remaining = list(range(len(candidates)))

        for _ in range(min(top_k, len(candidates))):
            if not remaining:
                break

            if not selected:
                # Pick the most relevant candidate first
                best = max(remaining, key=lambda i: relevance[i])
            else:
                best_score = -float("inf")
                best = remaining[0]
                selected_embs = cand_norms[selected]

                for i in remaining:
                    # Max similarity to already-selected items
                    sim_to_selected = float(np.max(cand_norms[i] @ selected_embs.T))
                    mmr_score = lambda_param * relevance[i] - (1 - lambda_param) * sim_to_selected
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best = i

            selected.append(best)
            remaining.remove(best)

        return [
            {
                "score": candidates[i]["score"],
                "text": candidates[i]["text"],
                "metadata": candidates[i]["metadata"],
            }
            for i in selected
        ]
