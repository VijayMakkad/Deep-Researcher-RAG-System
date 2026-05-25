"""Custom exception classes and FastAPI exception handlers.

Register handlers on the FastAPI app via :func:`register_exception_handlers`.
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class DocumentNotIndexedError(Exception):
    """Raised when a query is attempted but no documents have been indexed."""


class BackendUnavailableError(Exception):
    """Raised when the requested generation backend cannot be reached."""

    def __init__(self, backend: str, detail: str = ""):
        self.backend = backend
        self.detail = detail
        super().__init__(f"Backend '{backend}' unavailable: {detail}")


class InvalidFileTypeError(Exception):
    """Raised when an uploaded file has an unsupported type."""

    def __init__(self, filename: str, allowed: list[str] | None = None):
        self.filename = filename
        self.allowed = allowed or [".pdf", ".txt", ".md"]
        super().__init__(
            f"File '{filename}' has an unsupported type. Allowed: {', '.join(self.allowed)}"
        )


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


async def _handle_document_not_indexed(
    request: Request,
    exc: DocumentNotIndexedError,
) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={"detail": "No documents have been indexed yet. Upload documents first."},
    )


async def _handle_backend_unavailable(
    request: Request,
    exc: BackendUnavailableError,
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "detail": f"Backend '{exc.backend}' is unavailable.",
            "error": exc.detail,
        },
    )


async def _handle_invalid_file_type(
    request: Request,
    exc: InvalidFileTypeError,
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": f"Unsupported file type for '{exc.filename}'.",
            "allowed_types": exc.allowed,
        },
    )


async def _handle_unhandled_exception(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    request_id = str(uuid.uuid4())[:8]
    log.error(
        "unhandled_exception",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        error=str(exc),
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred.",
            "request_id": request_id,
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all custom exception handlers to the FastAPI application."""
    app.add_exception_handler(DocumentNotIndexedError, _handle_document_not_indexed)  # type: ignore[arg-type]
    app.add_exception_handler(BackendUnavailableError, _handle_backend_unavailable)  # type: ignore[arg-type]
    app.add_exception_handler(InvalidFileTypeError, _handle_invalid_file_type)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _handle_unhandled_exception)  # type: ignore[arg-type]
