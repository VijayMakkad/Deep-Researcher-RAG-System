"""Retriever unit tests."""

from __future__ import annotations

import pytest


@pytest.fixture()
def loaded_retriever(sample_chunks):
    """A retriever with a pre-built index from sample chunks."""
    from backend.services.retriever import Retriever

    r = Retriever()
    r.build_index(sample_chunks)
    return r


class TestBuildAndQuery:
    def test_build_index_sets_state(self, loaded_retriever):
        assert loaded_retriever.is_ready
        assert loaded_retriever.index is not None
        assert len(loaded_retriever.chunks) == 5

    def test_query_returns_results(self, loaded_retriever):
        results = loaded_retriever.query("What is machine learning?", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3

        for r in results:
            assert "score" in r
            assert "text" in r
            assert "metadata" in r
            assert "filename" in r["metadata"]
            assert "chunk_idx" in r["metadata"]

    def test_query_scores_are_sorted(self, loaded_retriever):
        """Results should be returned in relevance order (highest first)."""
        results = loaded_retriever.query("neural networks deep learning", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_metadata_has_real_filenames(self, loaded_retriever):
        """Bug #2 fix: doc_id should be real filenames, not doc_{i//10}."""
        results = loaded_retriever.query("machine learning", top_k=3)
        filenames = {r["metadata"]["filename"] for r in results}
        valid_names = {"ml_intro.pdf", "nlp_guide.pdf", "cv_paper.pdf", "rl_notes.txt"}
        assert filenames.issubset(valid_names)


class TestEmptyIndex:
    def test_empty_index_is_not_ready(self):
        from backend.services.retriever import Retriever

        r = Retriever.__new__(Retriever)
        r.index = None
        r.chunks = []
        r._embeddings = None
        assert not r.is_ready

    def test_empty_index_query_returns_empty(self):
        from unittest.mock import MagicMock

        from backend.services.retriever import Retriever

        r = Retriever.__new__(Retriever)
        r.index = None
        r.chunks = []
        r._embeddings = None
        r.model = MagicMock()
        results = r.query("anything", top_k=5)
        assert results == []


class TestMMR:
    def test_mmr_returns_diverse_results(self, loaded_retriever):
        """MMR should return results from multiple documents."""
        results = loaded_retriever.query(
            "artificial intelligence", top_k=4, diverse=True, lambda_param=0.5
        )
        assert len(results) > 0
        filenames = {r["metadata"]["filename"] for r in results}
        assert len(filenames) >= 1

    def test_mmr_result_count(self, loaded_retriever):
        results = loaded_retriever.query("learning", top_k=3, diverse=True)
        assert len(results) <= 3


class TestDocumentStats:
    def test_get_document_stats(self, loaded_retriever):
        stats = loaded_retriever.get_document_stats()
        assert len(stats) > 0
        for s in stats:
            assert "filename" in s
            assert "chunk_count" in s


class TestAddToIndex:
    def test_add_to_existing_index(self, loaded_retriever):
        initial_count = len(loaded_retriever.chunks)
        new_chunks = [
            {
                "text": "Transformers use self-attention mechanisms.",
                "metadata": {"filename": "transformers.pdf", "page": 1, "chunk_idx": 5},
            }
        ]
        loaded_retriever.add_to_index(new_chunks)
        assert len(loaded_retriever.chunks) == initial_count + 1

    def test_add_to_empty_index(self, sample_chunks):
        from backend.services.retriever import Retriever

        r = Retriever()
        r.add_to_index(sample_chunks)
        assert r.is_ready
        assert len(r.chunks) == len(sample_chunks)


class TestClearIndex:
    def test_clear_index(self, loaded_retriever):
        loaded_retriever.clear_index()
        assert not loaded_retriever.is_ready
        assert len(loaded_retriever.chunks) == 0
