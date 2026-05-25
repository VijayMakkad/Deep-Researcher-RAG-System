"""Unified document ingestion service.

Handles file parsing (PDF, TXT, Markdown) and sliding-window chunking with
configurable size and overlap. Replaces both the inline chunking in the old
``main.py`` and the standalone ``ingest_and_index.py`` script.
"""

from __future__ import annotations

from pathlib import PurePath

import structlog

from backend.core.config import settings
from backend.core.exceptions import InvalidFileTypeError

log = structlog.get_logger()

ALLOWED_EXTENSIONS: set[str] = {".pdf", ".txt", ".md"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_file(file_bytes: bytes, filename: str) -> list[dict]:
    """Parse a file and return a list of text chunks with metadata.

    Each item in the returned list has the shape::

        {"text": str, "metadata": {"filename": str, "page": int | None, "chunk_idx": int}}

    Raises:
        InvalidFileTypeError: If the file extension is not supported.
    """
    ext = PurePath(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise InvalidFileTypeError(filename, list(ALLOWED_EXTENSIONS))

    log.info("ingesting_file", filename=filename, extension=ext)

    if ext == ".pdf":
        pages = _parse_pdf(file_bytes)
    else:
        # .txt and .md are treated as plain text with a single "page"
        text = file_bytes.decode("utf-8", errors="ignore")
        pages = [(0, text)]

    chunks: list[dict] = []
    global_chunk_idx = 0
    for page_num, page_text in pages:
        page_chunks = _sliding_window_chunk(page_text)
        for chunk_text in page_chunks:
            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "filename": filename,
                        "page": page_num if ext == ".pdf" else None,
                        "chunk_idx": global_chunk_idx,
                    },
                }
            )
            global_chunk_idx += 1

    log.info("ingestion_complete", filename=filename, chunk_count=len(chunks))
    return chunks


def validate_file_extension(filename: str) -> None:
    """Raise :class:`InvalidFileTypeError` if the extension is unsupported."""
    ext = PurePath(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise InvalidFileTypeError(filename, list(ALLOWED_EXTENSIONS))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_pdf(file_bytes: bytes) -> list[tuple[int, str]]:
    """Extract text from a PDF, returning ``[(page_number, text), ...]``."""
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    pages: list[tuple[int, str]] = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((idx + 1, text))  # 1-indexed page numbers
    return pages


def _sliding_window_chunk(text: str) -> list[str]:
    """Split *text* into overlapping chunks using settings from config."""
    chunk_size = settings.chunk_size
    stride = chunk_size - settings.chunk_overlap
    if stride <= 0:
        stride = chunk_size  # fallback to non-overlapping

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += stride
    return chunks
