"""API endpoint tests."""

from __future__ import annotations

import io

import pytest


@pytest.mark.asyncio
async def test_health_returns_200(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_ready_before_upload(client):
    resp = await client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ready"] is False
    assert data["index_loaded"] is False


@pytest.mark.asyncio
async def test_upload_invalid_file_type_returns_422(client):
    fake_file = io.BytesIO(b"not a pdf")
    resp = await client.post(
        "/documents/upload",
        files=[("files", ("test.exe", fake_file, "application/octet-stream"))],
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_query_before_upload_returns_409(client):
    resp = await client.post("/query", json={"query": "What is AI?"})
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_query_returns_answer_and_sources(app_client):
    """Upload mock data then query it."""
    import numpy as np
    from httpx import ASGITransport, AsyncClient

    app, test_retriever, _, _ = app_client

    # Simulate an indexed retriever
    test_retriever.chunks = [
        {
            "text": "AI is artificial intelligence.",
            "metadata": {"filename": "test.pdf", "page": 1, "chunk_idx": 0},
        }
    ]
    # Create a minimal mock for the model and index
    test_retriever.model.encode.return_value = np.array([[0.1] * 384])

    import faiss

    index = faiss.IndexFlatL2(384)
    index.add(np.array([[0.1] * 384], dtype=np.float32))
    test_retriever.index = index
    test_retriever._embeddings = np.array([[0.1] * 384], dtype=np.float32)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/query", json={"query": "What is AI?", "top_k": 1})

    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "sources" in data
    assert "backend_used" in data
    assert "processing_time_ms" in data


@pytest.mark.asyncio
async def test_set_backend_invalid_returns_422(client):
    resp = await client.post(
        "/system/backend",
        json={"backend": "invalid_backend"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_set_backend_valid(client):
    resp = await client.post(
        "/system/backend",
        json={"backend": "lexrank"},
    )
    assert resp.status_code == 200
    assert resp.json()["backend"] == "lexrank"


@pytest.mark.asyncio
async def test_get_backend(client):
    resp = await client.get("/system/backend")
    assert resp.status_code == 200
    data = resp.json()
    assert "backend" in data
    assert "available_backends" in data


@pytest.mark.asyncio
async def test_list_backends(client):
    resp = await client.get("/system/backends")
    assert resp.status_code == 200
    assert "backends" in resp.json()


@pytest.mark.asyncio
async def test_list_documents_empty(client):
    resp = await client.get("/documents")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_chunks"] == 0
    assert data["documents"] == []


@pytest.mark.asyncio
async def test_delete_documents(client):
    resp = await client.delete("/documents")
    assert resp.status_code == 200
