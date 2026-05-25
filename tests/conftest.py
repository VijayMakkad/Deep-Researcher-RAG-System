"""Pytest fixtures for Deep Researcher tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.services.generation.base import BaseGenerator


class MockGenerator(BaseGenerator):
    """A simple mock generator for testing."""

    @property
    def name(self) -> str:
        return "lexrank"

    def generate(self, query: str, contexts: list[str], **kwargs: object) -> str:
        return f"Mock answer for: {query}"


@pytest.fixture()
def mock_generator() -> MockGenerator:
    return MockGenerator()


@pytest.fixture()
def app_client():
    """Create a test FastAPI app with overridden dependencies.

    Yields the app, test_retriever, test_state, and test_factory.
    """
    from backend.api.deps import (
        BackendState,
        get_backend_state,
        get_generator_factory,
        get_retriever,
    )
    from backend.main import create_app
    from backend.services.generation import GeneratorFactory
    from backend.services.retriever import Retriever

    app = create_app()

    # Override deps to avoid loading heavy models
    test_retriever = Retriever.__new__(Retriever)
    test_retriever.index = None
    test_retriever.chunks = []
    test_retriever._embeddings = None
    test_retriever.model = MagicMock()

    test_state = BackendState(default="lexrank")

    class MockFactory(GeneratorFactory):
        def get(self, backend: str) -> BaseGenerator:
            return MockGenerator()

    test_factory = MockFactory()

    app.dependency_overrides[get_retriever] = lambda: test_retriever
    app.dependency_overrides[get_backend_state] = lambda: test_state
    app.dependency_overrides[get_generator_factory] = lambda: test_factory

    return app, test_retriever, test_state, test_factory


@pytest.fixture()
async def client(app_client):
    """Async HTTP client for the test app."""
    from httpx import ASGITransport, AsyncClient

    app, *_ = app_client
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture()
def sample_chunks() -> list[dict]:
    """Sample chunk data for testing the retriever."""
    return [
        {
            "text": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"filename": "ml_intro.pdf", "page": 1, "chunk_idx": 0},
        },
        {
            "text": "Deep learning uses neural networks with many layers.",
            "metadata": {"filename": "ml_intro.pdf", "page": 1, "chunk_idx": 1},
        },
        {
            "text": "Natural language processing deals with text data.",
            "metadata": {"filename": "nlp_guide.pdf", "page": 2, "chunk_idx": 2},
        },
        {
            "text": "Computer vision focuses on image understanding.",
            "metadata": {"filename": "cv_paper.pdf", "page": 1, "chunk_idx": 3},
        },
        {
            "text": "Reinforcement learning trains agents via rewards.",
            "metadata": {"filename": "rl_notes.txt", "page": None, "chunk_idx": 4},
        },
    ]
