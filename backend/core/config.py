"""Centralised application configuration via Pydantic Settings.

All environment variables are defined here. No other module should call
``os.getenv()`` directly — import ``settings`` from this module instead.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment / ``.env`` file."""

    # -- Backend selection --
    summarizer_backend: str = "lexrank"

    # -- Ollama --
    ollama_model: str = "llama3"
    ollama_host: str = "localhost:11434"

    # -- Embeddings --
    embedding_model: str = "all-MiniLM-L6-v2"

    # -- Chunking --
    chunk_size: int = 500
    chunk_overlap: int = 100

    # -- Upload limits --
    max_upload_size_mb: int = 50

    # -- Server --
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # -- Logging --
    log_level: str = "INFO"

    # -- Frontend --
    api_base_url: str = "http://127.0.0.1:8000"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


settings = Settings()
