"""Ollama LLM generation backend.

Supports both standard and streaming responses for local LLM inference.
"""

from __future__ import annotations

from collections.abc import Generator

import structlog

from backend.core.config import settings
from backend.core.exceptions import BackendUnavailableError
from backend.services.generation.base import BaseGenerator

log = structlog.get_logger()


class OllamaGenerator(BaseGenerator):
    """Answer generator using a local Ollama LLM instance."""

    def __init__(self) -> None:
        self._model = settings.ollama_model
        self._client = self._create_client()
        log.info("generator_loaded", backend="ollama", model=self._model)

    @property
    def name(self) -> str:
        return "ollama"

    def generate(self, query: str, contexts: list[str], **kwargs: object) -> str:
        """Generate an answer via the Ollama chat API.

        Keyword Args:
            max_words: Approximate word limit for the answer.
        """
        if self._client is None:
            raise BackendUnavailableError("ollama", "Ollama package not installed")

        max_words = kwargs.get("max_words")
        prompt = self._build_prompt(query, contexts, max_words)

        try:
            response = self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
            answer: str = response["message"]["content"]

            # Expand if answer is too short relative to requested word count
            if max_words and isinstance(max_words, int):
                word_count = len(answer.split())
                if word_count < int(max_words * 0.8):
                    answer = self._expand_answer(answer, max_words, word_count)

            return answer
        except Exception as e:
            log.error("ollama_generation_failed", model=self._model, error=str(e))
            raise BackendUnavailableError("ollama", str(e)) from e

    def generate_stream(
        self, query: str, contexts: list[str], **kwargs: object
    ) -> Generator[str, None, None]:
        """Yield answer tokens as they arrive from Ollama (SSE-friendly)."""
        if self._client is None:
            raise BackendUnavailableError("ollama", "Ollama package not installed")

        max_words = kwargs.get("max_words")
        prompt = self._build_prompt(query, contexts, max_words)

        try:
            stream = self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                if token:
                    yield token
        except Exception as e:
            log.error("ollama_stream_failed", model=self._model, error=str(e))
            raise BackendUnavailableError("ollama", str(e)) from e

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(
        query: str,
        contexts: list[str],
        max_words: object = None,
    ) -> str:
        combined = "\n\n".join(contexts)

        word_instruction = ""
        if max_words and isinstance(max_words, int):
            low = int(max_words * 0.9)
            high = int(max_words * 1.1)
            word_instruction = (
                f"Write the answer in about {max_words} words. "
                f"Ensure it's between {low} and {high} words."
            )

        return (
            "You are an expert research assistant.\n"
            "Using only the provided context, produce a clear, well-structured academic answer.\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{combined}\n\n"
            "Guidelines for your answer:\n"
            "1. Start with a crisp definition of the term or concept.\n"
            "2. Explain the architecture, methodology, or approach in 2-3 sentences.\n"
            "3. Present the main contributions, results, or findings as bullet points.\n"
            "4. Highlight applications, strengths, or limitations if mentioned in context.\n"
            "5. End with a 1-2 sentence conclusion summarizing its overall significance.\n"
            "6. Maintain a formal, academic tone and avoid repetition.\n"
            "7. Do not add any information outside of the provided context.\n\n"
            f"{word_instruction}"
        )

    def _expand_answer(self, answer: str, max_words: int, current_wc: int) -> str:
        """Ask Ollama to expand a too-short answer."""
        expand_prompt = (
            f"The previous answer had {current_wc} words. "
            f"Expand and refine it to ~{max_words} words while keeping it structured."
        )
        try:
            response = self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": expand_prompt}],
            )
            return str(response["message"]["content"])
        except Exception as e:
            log.warning("ollama_expand_failed", error=str(e))
            return answer  # Return original if expansion fails

    @staticmethod
    def _create_client():  # type: ignore[return]
        """Create an Ollama client, handling import failures gracefully."""
        try:
            import ollama as ollama_lib

            host = settings.ollama_host
            if host:
                # Ensure the host has a scheme
                if not host.startswith(("http://", "https://")):
                    host = f"http://{host}"
                return ollama_lib.Client(host=host)
            return ollama_lib.Client()
        except ImportError:
            log.warning("ollama_not_installed")
            return None
        except Exception as e:
            log.error("ollama_client_creation_failed", error=str(e))
            return None
