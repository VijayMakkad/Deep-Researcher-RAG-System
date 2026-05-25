"""Generator factory — thread-safe singleton cache per backend type.

Fixes bug #3: replaces global mutable ``generator`` variable with a factory
that holds one singleton instance per backend type and swaps on demand.
"""

from __future__ import annotations

import threading

import structlog

from backend.services.generation.base import BaseGenerator

log = structlog.get_logger()

AVAILABLE_BACKENDS = ("lexrank", "distilbart", "ollama")


class GeneratorFactory:
    """Creates and caches :class:`BaseGenerator` instances.

    Each backend type is instantiated at most once (lazy singleton).
    Thread-safe via a lock.
    """

    def __init__(self) -> None:
        self._instances: dict[str, BaseGenerator] = {}
        self._lock = threading.Lock()

    def get(self, backend: str) -> BaseGenerator:
        """Return a generator for the given *backend* name.

        Creates the instance on first request; subsequent calls return the
        cached singleton.

        Raises:
            ValueError: If *backend* is not one of the known backends.
        """
        backend = backend.lower()
        if backend not in AVAILABLE_BACKENDS:
            raise ValueError(
                f"Unknown backend '{backend}'. Available: {', '.join(AVAILABLE_BACKENDS)}"
            )

        with self._lock:
            if backend not in self._instances:
                self._instances[backend] = self._create(backend)
            return self._instances[backend]

    # ------------------------------------------------------------------

    @staticmethod
    def _create(backend: str) -> BaseGenerator:
        log.info("creating_generator", backend=backend)

        if backend == "lexrank":
            from backend.services.generation.lexrank import LexRankGenerator

            return LexRankGenerator()

        if backend == "distilbart":
            from backend.services.generation.distilbart import DistilBARTGenerator

            return DistilBARTGenerator()

        if backend == "ollama":
            from backend.services.generation.ollama import OllamaGenerator

            return OllamaGenerator()

        # Should never reach here due to validation above
        raise ValueError(f"Unknown backend: {backend}")


# Module-level factory instance (used via dependency injection)
generator_factory = GeneratorFactory()
