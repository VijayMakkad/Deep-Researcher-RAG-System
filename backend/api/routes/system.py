"""System / health routes — liveness, readiness, backend management."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends

from backend.api.deps import (
    BackendState,
    get_backend_state,
    get_generator_factory,
    get_retriever,
)
from backend.core.config import settings
from backend.models.requests import SetBackendRequest
from backend.models.responses import BackendInfoResponse, HealthResponse, ReadyResponse
from backend.services.generation import AVAILABLE_BACKENDS, GeneratorFactory
from backend.services.retriever import Retriever

log = structlog.get_logger()

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe — always returns 200."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
async def ready(
    retriever: Retriever = Depends(get_retriever),
) -> ReadyResponse:
    """Readiness probe — checks whether the FAISS index is loaded."""
    return ReadyResponse(
        ready=retriever.is_ready,
        index_loaded=retriever.is_ready,
        chunk_count=len(retriever.chunks),
    )


@router.get("/system/backend", response_model=BackendInfoResponse)
async def get_backend(
    state: BackendState = Depends(get_backend_state),
) -> BackendInfoResponse:
    """Return the currently active backend and its model info."""
    backend = state.backend
    model = settings.ollama_model if backend == "ollama" else None
    return BackendInfoResponse(
        backend=backend,
        model=model,
        available_backends=list(AVAILABLE_BACKENDS),
    )


@router.post("/system/backend", response_model=BackendInfoResponse)
async def set_backend(
    body: SetBackendRequest,
    state: BackendState = Depends(get_backend_state),
    factory: GeneratorFactory = Depends(get_generator_factory),
) -> BackendInfoResponse:
    """Switch the active generation backend."""
    # Pre-create the generator to validate it works
    factory.get(body.backend)
    state.backend = body.backend

    model = settings.ollama_model if body.backend == "ollama" else None
    log.info("backend_set_via_api", backend=body.backend)
    return BackendInfoResponse(
        backend=body.backend,
        model=model,
        available_backends=list(AVAILABLE_BACKENDS),
    )


@router.get("/system/backends")
async def list_backends() -> dict:
    """List all available backend names."""
    return {"backends": list(AVAILABLE_BACKENDS)}


# ---------------------------------------------------------------------------
# Legacy compatibility routes (redirect-style)
# ---------------------------------------------------------------------------


@router.get("/backend", response_model=BackendInfoResponse, include_in_schema=False)
async def legacy_get_backend(
    state: BackendState = Depends(get_backend_state),
) -> BackendInfoResponse:
    """Legacy endpoint — forwards to ``/system/backend``."""
    return await get_backend(state)


@router.post("/set_backend", include_in_schema=False)
async def legacy_set_backend(
    backend: str,
    state: BackendState = Depends(get_backend_state),
    factory: GeneratorFactory = Depends(get_generator_factory),
) -> BackendInfoResponse:
    """Legacy endpoint — forwards to ``POST /system/backend``."""
    body = SetBackendRequest(backend=backend)  # type: ignore[arg-type]
    return await set_backend(body, state, factory)
