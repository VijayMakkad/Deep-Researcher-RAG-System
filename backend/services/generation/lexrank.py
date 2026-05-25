"""LexRank extractive summarisation backend."""

from __future__ import annotations

import random

import structlog
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer

from backend.services.generation.base import BaseGenerator

log = structlog.get_logger()


class LexRankGenerator(BaseGenerator):
    """Extractive summariser using the LexRank algorithm via ``sumy``."""

    def __init__(self) -> None:
        self._summarizer = LexRankSummarizer()
        log.info("generator_loaded", backend="lexrank")

    @property
    def name(self) -> str:
        return "lexrank"

    def generate(self, query: str, contexts: list[str], **kwargs: object) -> str:
        """Extract key sentences from *contexts* to answer *query*.

        Keyword Args:
            diverse: If ``True``, generate multiple summaries with varying
                sentence counts for broader coverage.
        """
        combined = "\n\n".join(contexts)
        diverse = bool(kwargs.get("diverse", False))

        if diverse:
            outputs: list[str] = []
            for _ in range(3):
                sentence_count = random.choice([2, 3, 4, 5])
                outputs.append(self._summarise(combined, sentence_count))
            return "\n\n---\n\n".join(outputs)

        return self._summarise(combined, sentence_count=5)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _summarise(self, text: str, sentence_count: int = 5) -> str:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        sentences = self._summarizer(parser.document, sentences_count=sentence_count)
        return " ".join(str(s) for s in sentences)
