#!/usr/bin/env python3
"""Standalone index builder — pre-builds a FAISS index from local documents.

Usage::

    python scripts/build_index.py [--docs-dir sample_docs] [--output-dir .]

This uses the same ingestion and retriever services as the backend API,
ensuring consistent chunking and embedding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss
import structlog

from backend.core.config import settings
from backend.core.logging import setup_logging
from backend.services.ingestion import ALLOWED_EXTENSIONS, ingest_file
from backend.services.retriever import Retriever

setup_logging()
log = structlog.get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from local documents")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("sample_docs"),
        help="Directory containing documents to index (default: sample_docs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to write faiss_index.bin and doc_meta.json (default: .)",
    )
    args = parser.parse_args()

    docs_dir: Path = args.docs_dir
    output_dir: Path = args.output_dir

    if not docs_dir.exists():
        log.error("docs_dir_not_found", path=str(docs_dir))
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all supported files
    all_chunks: list[dict] = []
    file_count = 0
    for filepath in sorted(docs_dir.iterdir()):
        if filepath.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        log.info("processing_file", file=filepath.name)
        file_bytes = filepath.read_bytes()
        chunks = ingest_file(file_bytes, filepath.name)
        all_chunks.extend(chunks)
        file_count += 1

    if not all_chunks:
        log.error("no_chunks_created", docs_dir=str(docs_dir))
        sys.exit(1)

    log.info("total_chunks", count=len(all_chunks), files=file_count)

    # Build the retriever and save artifacts
    retriever = Retriever(model_name=settings.embedding_model)
    retriever.build_index(all_chunks)

    index_path = output_dir / "faiss_index.bin"
    meta_path = output_dir / "doc_meta.json"

    faiss.write_index(retriever.index, str(index_path))

    metadata = [c["metadata"] for c in all_chunks]
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log.info(
        "index_saved",
        index_path=str(index_path),
        meta_path=str(meta_path),
        chunk_count=len(all_chunks),
    )


if __name__ == "__main__":
    main()
