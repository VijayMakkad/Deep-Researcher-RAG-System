"""Typed API client — all HTTP calls in one place.

Reads ``API_BASE_URL`` from environment or ``.env`` file (no hardcoded URLs).
Streamlit does not load ``.env`` natively, so we use ``python-dotenv``
(already installed as a transitive dependency of ``pydantic-settings``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

# Load .env so the frontend process sees API_BASE_URL even without manual export
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


@dataclass
class BackendStatus:
    display: str
    is_connected: bool
    error: str | None = None
    backend_key: str = "lexrank"


BACKEND_LABELS: dict[str, str] = {
    "lexrank": "📝 LexRank (Extractive, lightweight)",
    "distilbart": "🤖 DistilBART (Abstractive, HuggingFace)",
    "ollama": "🦙 Ollama LLM (LLaMA/Mistral, quantized)",
}


def get_backend_status() -> BackendStatus:
    """Fetch the current backend status from the API."""
    try:
        resp = httpx.get(f"{API_BASE_URL}/system/backend", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            backend_key = data.get("backend", "lexrank").lower()
            model_name = data.get("model", "")
            display = BACKEND_LABELS.get(backend_key, "❓ Unknown backend")

            if backend_key == "ollama" and model_name:
                display += f" — Model: {model_name}"

            return BackendStatus(
                display=display,
                is_connected=True,
                backend_key=backend_key,
            )
        return BackendStatus(
            display="❌ API error",
            is_connected=False,
            error=f"HTTP {resp.status_code}",
        )
    except Exception as e:
        return BackendStatus(
            display="❌ Connection failed",
            is_connected=False,
            error=str(e)[:80],
        )


def set_backend(backend_key: str) -> tuple[bool, str]:
    """Set the backend via API. Returns (success, message)."""
    try:
        resp = httpx.post(
            f"{API_BASE_URL}/system/backend",
            json={"backend": backend_key},
            timeout=10,
        )
        if resp.status_code == 200:
            return True, resp.json().get("backend", "Updated")
        return False, f"Failed: HTTP {resp.status_code}"
    except Exception as e:
        return False, f"Error: {str(e)[:50]}"


def upload_files(files: list) -> tuple[bool, str, dict | None]:
    """Upload files to the backend. Returns (success, message, response_data)."""
    try:
        multipart = [("files", (f.name, f, "application/pdf")) for f in files]
        resp = httpx.post(
            f"{API_BASE_URL}/documents/upload",
            files=multipart,
            timeout=120,
        )
        if resp.status_code == 200:
            data = resp.json()
            return True, data.get("message", "Uploaded successfully"), data
        return False, f"Upload failed: {resp.text}", None
    except Exception as e:
        return False, f"Upload error: {e}", None


def query_documents(
    query: str,
    top_k: int = 5,
    max_words: int | None = None,
    backend: str | None = None,
    diverse: bool = False,
    timeout: int = 120,
) -> tuple[bool, dict | None, str | None]:
    """Send a query to the backend. Returns (success, data, error)."""
    payload: dict = {
        "query": query.strip(),
        "top_k": max(1, min(top_k, 20)),
        "diverse": diverse,
    }
    if max_words and max_words > 0:
        payload["max_words"] = max(10, min(max_words, 2000))
    if backend:
        payload["backend"] = backend

    try:
        resp = httpx.post(
            f"{API_BASE_URL}/query",
            json=payload,
            timeout=timeout,
        )
        if resp.status_code == 200:
            return True, resp.json(), None
        return False, None, f"API Error: {resp.status_code} — {resp.text}"
    except httpx.TimeoutException:
        return False, None, f"Request timed out after {timeout} seconds."
    except httpx.ConnectError:
        return False, None, "Cannot connect to the API server."
    except Exception as e:
        return False, None, f"Unexpected error: {e}"


def health_check() -> bool:
    """Quick liveness check."""
    try:
        resp = httpx.get(f"{API_BASE_URL}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False
