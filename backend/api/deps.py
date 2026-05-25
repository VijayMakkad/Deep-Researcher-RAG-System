"""FastAPI dependency injection — shared state and service providers.

Replaces the old module-level globals with proper DI.  The
:class:`BackendState` singleton manages the active backend name
thread-safely.
"""

from __future__ import annotations

import threading

import structlog

from backend.core.config import settings
from backend.services.generation import GeneratorFactory, generator_factory
from backend.services.generation.base import BaseGenerator
from backend.services.retriever import Retriever

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Singletons (created once at import time)
# ---------------------------------------------------------------------------

_retriever = Retriever()


class BackendState:
    """Thread-safe container for the currently active backend name."""

    def __init__(self, default: str | None = None) -> None:
        self._backend = default or settings.summarizer_backend
        self._lock = threading.Lock()

    @property
    def backend(self) -> str:
        with self._lock:
            return self._backend

    @backend.setter
    def backend(self, value: str) -> None:
        with self._lock:
            self._backend = value
            log.info("backend_switched", backend=value)


_backend_state = BackendState()


# ---------------------------------------------------------------------------
# FastAPI dependency callables
# ---------------------------------------------------------------------------


def get_retriever() -> Retriever:
    """Return the shared :class:`Retriever` instance."""
    return _retriever


def get_backend_state() -> BackendState:
    """Return the shared :class:`BackendState` singleton."""
    return _backend_state


def get_generator_factory() -> GeneratorFactory:
    """Return the shared :class:`GeneratorFactory`."""
    return generator_factory


def get_current_generator() -> BaseGenerator:
    """Return the generator for the currently active backend."""
    return generator_factory.get(_backend_state.backend)
