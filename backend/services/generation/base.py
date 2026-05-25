"""Abstract base class for answer generators (Strategy pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    """All generation backends must implement this interface."""

    @abstractmethod
    def generate(self, query: str, contexts: list[str], **kwargs: object) -> str:
        """Generate an answer for *query* given retrieved *contexts*."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical backend name (e.g. ``"lexrank"``, ``"distilbart"``, ``"ollama"``)."""
        ...
