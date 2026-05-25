"""Query routes — standard and streaming."""

from __future__ import annotations

import json
import time

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.api.deps import (
    BackendState,
    get_backend_state,
    get_generator_factory,
    get_retriever,
)
from backend.core.exceptions import DocumentNotIndexedError
from backend.models.requests import QueryRequest
from backend.models.responses import QueryResponse, SourceChunk
from backend.services.generation import GeneratorFactory
from backend.services.retriever import Retriever

log = structlog.get_logger()

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def run_query(
    body: QueryRequest,
    retriever: Retriever = Depends(get_retriever),
    factory: GeneratorFactory = Depends(get_generator_factory),
    state: BackendState = Depends(get_backend_state),
) -> QueryResponse:
    """Run a retrieval-augmented query and return the generated answer."""
    if not retriever.is_ready:
        raise DocumentNotIndexedError()

    start = time.perf_counter()

    # Determine which backend to use
    backend_name = body.backend or state.backend
    generator = factory.get(backend_name)

    # Retrieve relevant chunks
    hits = retriever.query(
        body.query,
        top_k=body.top_k,
        diverse=body.diverse,
    )

    contexts = [h["text"] for h in hits]
    sources = [
        SourceChunk(
            filename=h["metadata"]["filename"],
            page=h["metadata"].get("page"),
            chunk_idx=h["metadata"]["chunk_idx"],
            score=h["score"],
            snippet=h["text"][:300],
        )
        for h in hits
    ]

    # Generate answer
    kwargs: dict = {}
    if body.max_words is not None:
        kwargs["max_words"] = body.max_words
    kwargs["diverse"] = body.diverse

    answer = generator.generate(body.query, contexts, **kwargs)

    elapsed_ms = (time.perf_counter() - start) * 1000
    log.info(
        "query_completed",
        backend=backend_name,
        top_k=body.top_k,
        diverse=body.diverse,
        processing_time_ms=round(elapsed_ms, 1),
    )

    return QueryResponse(
        answer=answer,
        sources=sources,
        backend_used=backend_name,
        processing_time_ms=round(elapsed_ms, 1),
    )


@router.post("/stream")
def stream_query(
    body: QueryRequest,
    retriever: Retriever = Depends(get_retriever),
    factory: GeneratorFactory = Depends(get_generator_factory),
    state: BackendState = Depends(get_backend_state),
) -> StreamingResponse:
    """Stream the answer token-by-token via Server-Sent Events (Ollama only)."""
    if not retriever.is_ready:
        raise DocumentNotIndexedError()

    backend_name = body.backend or state.backend

    # Streaming is only supported for Ollama
    if backend_name != "ollama":
        # Fall back to non-streaming for other backends
        result = run_query(body, retriever, factory, state)
        return StreamingResponse(
            _single_event(result.model_dump()),
            media_type="text/event-stream",
        )

    generator = factory.get("ollama")
    hits = retriever.query(body.query, top_k=body.top_k, diverse=body.diverse)
    contexts = [h["text"] for h in hits]

    sources = [
        {
            "filename": h["metadata"]["filename"],
            "page": h["metadata"].get("page"),
            "chunk_idx": h["metadata"]["chunk_idx"],
            "score": h["score"],
            "snippet": h["text"][:300],
        }
        for h in hits
    ]

    kwargs: dict = {}
    if body.max_words is not None:
        kwargs["max_words"] = body.max_words

    def event_generator():
        # Send sources first
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        # Stream tokens
        from backend.services.generation.ollama import OllamaGenerator

        if isinstance(generator, OllamaGenerator):
            for token in generator.generate_stream(body.query, contexts, **kwargs):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def _single_event(data: dict):
    """Yield a single SSE event for non-streaming backends."""
    yield f"data: {json.dumps(data)}\n\n"
