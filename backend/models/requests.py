"""Pydantic request schemas for API endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Body for ``POST /query``."""

    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    max_words: int | None = Field(None, ge=10, le=2000)
    backend: Literal["lexrank", "distilbart", "ollama"] | None = None
    diverse: bool = Field(False, description="Enable MMR diversity re-ranking")


class SetBackendRequest(BaseModel):
    """Body for ``POST /system/backend``."""

    backend: Literal["lexrank", "distilbart", "ollama"]
