"""FastAPI application factory — no business logic here.

Run with::

    uvicorn backend.main:app --reload
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import documents, query, system
from backend.core.config import settings
from backend.core.exceptions import register_exception_handlers
from backend.core.logging import setup_logging

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan — startup / shutdown hooks."""
    # Ensure NLTK data is available
    import nltk

    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    # ------------------------------------------------------------------
    # Migration guard: clear stale index files from the pre-v0.2.0 schema.
    # The old codebase stored flat text arrays without per-chunk metadata;
    # the new schema expects {"text", "metadata"} dicts. Loading old files
    # would cause KeyErrors at query time, so we proactively remove them.
    # ------------------------------------------------------------------
    _cleanup_stale_index()

    log.info(
        "application_started",
        backend=settings.summarizer_backend,
        embedding_model=settings.embedding_model,
        host=settings.api_host,
        port=settings.api_port,
    )
    yield
    log.info("application_shutdown")


def _cleanup_stale_index() -> None:
    """Remove pre-v0.2.0 index files whose metadata schema is incompatible."""
    import json
    from pathlib import Path

    meta_path = Path("doc_meta.json")
    index_path = Path("faiss_index.bin")

    if not meta_path.exists():
        return  # Nothing to migrate

    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(raw, list) and raw:
            first = raw[0]
            # New schema entries have "filename" and "chunk_idx" keys.
            # Old schema entries are just bare strings or lack these keys.
            if not isinstance(first, dict) or "filename" not in first:
                log.warning(
                    "stale_index_detected",
                    detail=(
                        "Removing pre-v0.2.0 index files (incompatible metadata schema). "
                        "Re-upload your documents to rebuild the index."
                    ),
                )
                meta_path.unlink(missing_ok=True)
                index_path.unlink(missing_ok=True)
    except Exception:
        # If we can't parse it, it's definitely stale — remove it.
        log.warning("corrupt_meta_file", detail="Removing unreadable doc_meta.json")
        meta_path.unlink(missing_ok=True)
        index_path.unlink(missing_ok=True)


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    setup_logging()

    application = FastAPI(
        title="Deep Researcher — RAG Document Q&A",
        description="Retrieval-Augmented Generation system with multiple AI backends.",
        version="0.2.0",
        lifespan=lifespan,
    )

    # -- Middleware --
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Exception handlers --
    register_exception_handlers(application)

    # -- Routers --
    application.include_router(documents.router)
    application.include_router(query.router)
    application.include_router(system.router)

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
