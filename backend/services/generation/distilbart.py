"""DistilBART abstractive summarisation backend.

This was previously called ``"t5"`` in the codebase even though it loads
``sshleifer/distilbart-cnn-12-6``. The canonical name is now ``"distilbart"``
everywhere.

Requires the ``distilbart`` optional dependency group::

    pip install "deep-researcher[distilbart]"
"""

from __future__ import annotations

import structlog

from backend.services.generation.base import BaseGenerator

log = structlog.get_logger()

DISTILBART_MODEL = "sshleifer/distilbart-cnn-12-6"


class DistilBARTGenerator(BaseGenerator):
    """Abstractive summariser using the DistilBART HuggingFace pipeline."""

    def __init__(self) -> None:
        self._pipeline = self._load_pipeline()
        log.info("generator_loaded", backend="distilbart", model=DISTILBART_MODEL)

    @property
    def name(self) -> str:
        return "distilbart"

    def generate(self, query: str, contexts: list[str], **kwargs: object) -> str:
        """Generate an abstractive answer via DistilBART.

        Falls back to a simple concatenation if the pipeline is unavailable.

        Keyword Args:
            max_words: Approximate word limit for the answer.
        """
        if self._pipeline is None:
            log.warning("distilbart_unavailable_fallback")
            return "[DistilBART unavailable — install torch and transformers]"

        combined = "\n\n".join(contexts)
        prompt = (
            f"Answer the query based on context:\n"
            f"Question: {query}\n\n"
            f"Context:\n{combined}\n\n"
            f"Provide a concise summary."
        )

        # Truncate to stay within model context limits
        if len(prompt) > 3000:
            prompt = prompt[:3000]

        max_words = kwargs.get("max_words")
        max_length = 200
        if max_words and isinstance(max_words, int):
            max_length = max_words * 2  # heuristic: tokens ≈ words × 1.3

        try:
            out = self._pipeline(prompt, max_length=max_length, min_length=30, do_sample=False)
            return str(out[0]["summary_text"])  # type: ignore[index]
        except Exception as e:
            log.error("distilbart_generation_failed", error=str(e))
            return f"[DistilBART generation failed: {e}]"

    # ------------------------------------------------------------------

    @staticmethod
    def _load_pipeline():  # type: ignore[return]
        """Attempt to load the HuggingFace summarisation pipeline."""
        try:
            from transformers import pipeline as hf_pipeline

            log.info("loading_distilbart", model=DISTILBART_MODEL)
            return hf_pipeline("summarization", model=DISTILBART_MODEL, device=-1)
        except ImportError:
            log.warning("transformers_not_installed")
            return None
        except Exception as e:
            log.error("distilbart_load_failed", error=str(e))
            return None
