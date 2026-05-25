"""Generation layer tests."""

from __future__ import annotations

import pytest

from backend.services.generation import AVAILABLE_BACKENDS, GeneratorFactory
from backend.services.generation.base import BaseGenerator
from backend.services.generation.lexrank import LexRankGenerator


class TestLexRankGenerator:
    def test_implements_interface(self):
        gen = LexRankGenerator()
        assert isinstance(gen, BaseGenerator)
        assert gen.name == "lexrank"

    def test_generate_returns_string(self):
        gen = LexRankGenerator()
        contexts = [
            "Machine learning is a field of computer science. "
            "It focuses on the development of algorithms. "
            "These algorithms can learn from data. "
            "They make predictions based on patterns. "
            "The field has grown rapidly in recent years.",
        ]
        result = gen.generate("What is ML?", contexts)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_diverse_mode(self):
        gen = LexRankGenerator()
        contexts = [
            "Deep learning is a subset of machine learning. "
            "It uses neural networks with multiple layers. "
            "Convolutional networks are used for images. "
            "Recurrent networks handle sequential data. "
            "Transformers have revolutionised NLP tasks.",
        ]
        result = gen.generate("What is deep learning?", contexts, diverse=True)
        assert isinstance(result, str)
        # Diverse mode returns multiple summaries separated by ---
        assert "---" in result


class TestGeneratorFactory:
    def test_factory_returns_generator(self):
        factory = GeneratorFactory()
        gen = factory.get("lexrank")
        assert isinstance(gen, BaseGenerator)
        assert gen.name == "lexrank"

    def test_factory_singleton(self):
        """Same backend should return the same instance."""
        factory = GeneratorFactory()
        gen1 = factory.get("lexrank")
        gen2 = factory.get("lexrank")
        assert gen1 is gen2

    def test_factory_invalid_backend(self):
        factory = GeneratorFactory()
        with pytest.raises(ValueError, match="Unknown backend"):
            factory.get("nonexistent")

    def test_available_backends_list(self):
        assert "lexrank" in AVAILABLE_BACKENDS
        assert "distilbart" in AVAILABLE_BACKENDS
        assert "ollama" in AVAILABLE_BACKENDS
        # Bug #4 fix: "t5" should NOT be a valid backend name
        assert "t5" not in AVAILABLE_BACKENDS
