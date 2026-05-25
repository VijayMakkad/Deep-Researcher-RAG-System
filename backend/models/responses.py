"""Pydantic response schemas for API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    """A single retrieved source chunk."""

    filename: str
    page: int | None = None
    chunk_idx: int
    score: float
    snippet: str


class QueryResponse(BaseModel):
    """Response body for ``POST /query``."""

    answer: str
    sources: list[SourceChunk]
    backend_used: str
    processing_time_ms: float


class UploadResponse(BaseModel):
    """Response body for ``POST /documents/upload``."""

    message: str
    chunk_count: int
    file_count: int


class DocumentInfo(BaseModel):
    """Metadata for an indexed document."""

    filename: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    """Response body for ``GET /documents``."""

    documents: list[DocumentInfo]
    total_chunks: int


class BackendInfoResponse(BaseModel):
    """Response body for ``GET /system/backend``."""

    backend: str
    model: str | None = None
    available_backends: list[str] = Field(
        default_factory=lambda: ["lexrank", "distilbart", "ollama"]
    )


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: str = "ok"


class ReadyResponse(BaseModel):
    """Response body for ``GET /ready``."""

    ready: bool
    index_loaded: bool
    chunk_count: int = 0


class ErrorResponse(BaseModel):
    """Standard error response envelope."""

    detail: str
    request_id: str | None = None
