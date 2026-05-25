"""Document management routes — upload, list, clear."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, File, UploadFile

from backend.api.deps import get_retriever
from backend.core.config import settings
from backend.core.exceptions import InvalidFileTypeError
from backend.models.responses import DocumentInfo, DocumentListResponse, UploadResponse
from backend.services.ingestion import ingest_file, validate_file_extension
from backend.services.retriever import Retriever

log = structlog.get_logger()

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    retriever: Retriever = Depends(get_retriever),
) -> UploadResponse:
    """Upload PDF/TXT/MD files, extract text, chunk, and add to the FAISS index."""
    all_chunks: list[dict] = []

    for upload_file in files:
        filename = upload_file.filename or "unknown"

        # Validate extension before reading the full file
        validate_file_extension(filename)

        # Validate content type for PDFs
        if filename.lower().endswith(".pdf"):
            content_type = upload_file.content_type or ""
            if content_type and content_type not in (
                "application/pdf",
                "application/octet-stream",
            ):
                raise InvalidFileTypeError(filename)

        file_bytes = await upload_file.read()

        # Validate file size
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > settings.max_upload_size_mb:
            raise InvalidFileTypeError(
                filename,
                [f"Max size: {settings.max_upload_size_mb}MB, got {size_mb:.1f}MB"],
            )

        chunks = ingest_file(file_bytes, filename)
        all_chunks.extend(chunks)

    if not all_chunks:
        return UploadResponse(
            message="No text could be extracted from the uploaded files.",
            chunk_count=0,
            file_count=len(files),
        )

    retriever.add_to_index(all_chunks)

    log.info("documents_uploaded", file_count=len(files), chunk_count=len(all_chunks))
    return UploadResponse(
        message=f"Uploaded and indexed {len(all_chunks)} chunks from {len(files)} file(s).",
        chunk_count=len(all_chunks),
        file_count=len(files),
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    retriever: Retriever = Depends(get_retriever),
) -> DocumentListResponse:
    """List all indexed documents with their chunk counts."""
    stats = retriever.get_document_stats()
    return DocumentListResponse(
        documents=[DocumentInfo(**s) for s in stats],
        total_chunks=len(retriever.chunks),
    )


@router.delete("")
async def clear_documents(
    retriever: Retriever = Depends(get_retriever),
) -> dict:
    """Clear the entire FAISS index."""
    retriever.clear_index()
    log.info("index_cleared_via_api")
    return {"message": "Index cleared successfully."}
