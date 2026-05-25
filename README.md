# 🔬 Deep Researcher — RAG Document Q&A System

A production-grade Retrieval-Augmented Generation (RAG) system for intelligent document querying. Upload documents, ask questions, and get comprehensive answers powered by multiple AI backends.

## Architecture

```
┌───────────────────┐        ┌──────────────────────┐       ┌───────────────────┐
│  Streamlit UI     │  HTTP  │  FastAPI Backend      │       │  AI Backends      │
│  (frontend/)      │───────▶│  (backend/)           │──────▶│  • LexRank        │
│  Port 8501        │        │  Port 8000            │       │  • DistilBART     │
└───────────────────┘        └──────────┬───────────┘       │  • Ollama (LLM)   │
                                        │                    └───────────────────┘
                             ┌──────────▼───────────┐
                             │  FAISS Vector Index   │
                             │  + SentenceTransformer│
                             └──────────────────────┘
```

## Features

- **📄 Multi-format ingestion**: PDF, TXT, Markdown with configurable sliding-window chunking
- **🤖 Three AI backends**:
  - **LexRank** — extractive summarisation (fast, lightweight)
  - **DistilBART** — abstractive summarisation via HuggingFace (optional, requires `torch`)
  - **Ollama** — local LLM inference (LLaMA, Mistral, etc.)
- **🔍 FAISS vector search** with sentence-transformer embeddings
- **🔥 Diverse Mode** — MMR (Maximal Marginal Relevance) re-ranking for broader topic coverage
- **📡 Streaming** — SSE endpoint for token-by-token Ollama responses
- **🐳 Docker** — separate backend/frontend containers with health checks
- **🧪 Tested** — pytest suite with async API tests, retriever tests, and generation tests

## Project Structure

```
deep-researcher/
├── backend/
│   ├── main.py                    # FastAPI app factory
│   ├── api/
│   │   ├── deps.py                # Dependency injection
│   │   └── routes/
│   │       ├── documents.py       # /documents/upload, GET, DELETE
│   │       ├── query.py           # /query, /query/stream
│   │       └── system.py          # /health, /ready, /system/backend
│   ├── core/
│   │   ├── config.py              # Pydantic Settings (all env vars)
│   │   ├── logging.py             # structlog setup
│   │   └── exceptions.py          # Custom exceptions + handlers
│   ├── services/
│   │   ├── ingestion.py           # PDF/text parsing & chunking
│   │   ├── retriever.py           # FAISS retrieval + MMR
│   │   └── generation/
│   │       ├── base.py            # Abstract Generator base
│   │       ├── lexrank.py         # LexRank implementation
│   │       ├── distilbart.py      # DistilBART implementation
│   │       └── ollama.py          # Ollama implementation
│   └── models/
│       ├── requests.py            # Pydantic request schemas
│       └── responses.py           # Pydantic response schemas
├── frontend/
│   ├── app.py                     # Streamlit entry point
│   ├── api_client.py              # Typed HTTP client (httpx)
│   ├── style.css                  # Custom CSS
│   ├── state.py                   # Session state key constants
│   └── components/
│       ├── sidebar.py             # Sidebar UI
│       ├── results.py             # Answer + sources display
│       └── charts.py              # Relevance chart
├── tests/
│   ├── conftest.py                # Fixtures
│   ├── test_api.py
│   ├── test_retriever.py
│   └── test_generation.py
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── scripts/
│   └── build_index.py             # Standalone index builder
├── .env.example
├── .pre-commit-config.yaml
├── pyproject.toml
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) (optional, for LLM backend)

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd deep-researcher

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -e .

# Or with DistilBART support (downloads ~2GB of PyTorch + model weights)
pip install -e ".[distilbart]"

# Or with dev tools
pip install -e ".[dev]"

# Or with plain requirements.txt (no extras)
pip install -r requirements.txt
```

> **⚠️ Upgrading from v0.1.x?** The metadata schema for the FAISS index
> changed in v0.2.0. On first startup, the backend automatically detects
> and removes stale `faiss_index.bin` / `doc_meta.json` files from the old
> schema. You will need to **re-upload your documents** to rebuild the index.

### 2. Configure

```bash
cp .env.example .env
# Edit .env to adjust settings (defaults work out of the box for LexRank)
```

### 3. Run

```bash
# Terminal 1 — Backend
uvicorn backend.main:app --reload

# Terminal 2 — Frontend
streamlit run frontend/app.py
```

Open http://localhost:8501 in your browser.

### 4. (Optional) Ollama Setup

```bash
ollama serve &
ollama pull llama3

# Set in .env:
# SUMMARIZER_BACKEND=ollama
# OLLAMA_MODEL=llama3
```

## Docker Deployment

```bash
cp .env.example .env   # adjust if needed

docker compose -f docker/docker-compose.yml up --build
```

This starts two containers:
- **Backend** on port 8000 (with health checks)
- **Frontend** on port 8501 (waits for backend to be healthy)

## Environment Variables

All configuration is centralised in `backend/core/config.py` and read from `.env`:

| Variable | Default | Description |
|---|---|---|
| `SUMMARIZER_BACKEND` | `lexrank` | Default backend: `lexrank`, `distilbart`, `ollama` |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `OLLAMA_HOST` | `localhost:11434` | Ollama server address |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `MAX_UPLOAD_SIZE_MB` | `50` | Max upload file size |
| `API_HOST` | `0.0.0.0` | Backend bind address |
| `API_PORT` | `8000` | Backend port |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG` for dev) |
| `API_BASE_URL` | `http://127.0.0.1:8000` | Frontend → backend URL |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/documents/upload` | Upload and index PDF/TXT/MD files |
| `GET` | `/documents` | List indexed documents with chunk counts |
| `DELETE` | `/documents` | Clear the FAISS index |
| `POST` | `/query` | Query documents → full response |
| `POST` | `/query/stream` | Query with SSE streaming (Ollama) |
| `GET` | `/health` | Liveness check |
| `GET` | `/ready` | Readiness check (index loaded?) |
| `GET` | `/system/backend` | Current backend info |
| `POST` | `/system/backend` | Switch backend |
| `GET` | `/system/backends` | List available backends |

Interactive API docs: http://localhost:8000/docs

## Testing

```bash
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Linting
ruff check .
ruff format --check .

# Type checking
mypy backend/ --ignore-missing-imports
```

## Contributing

1. Fork the repository
2. Install dev dependencies: `pip install -e ".[dev]"`
3. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```
4. Create a feature branch and make your changes
5. Ensure all checks pass:
   ```bash
   ruff check .
   ruff format .
   mypy backend/ --ignore-missing-imports
   pytest tests/ -v
   ```
6. Open a Pull Request

## AI Backends

### LexRank
- **Type**: Extractive summarisation
- **Use case**: Quick document insights, key sentence extraction
- **Pros**: Fast, no GPU needed, preserves original text
- **Install**: Included in core dependencies

### DistilBART
- **Type**: Abstractive summarisation
- **Use case**: Coherent summaries, text generation
- **Pros**: Natural language generation, contextual understanding
- **Install**: `pip install -e ".[distilbart]"` (requires ~2GB for PyTorch)

### Ollama (LLaMA3, Mistral, etc.)
- **Type**: Large Language Model
- **Use case**: Complex reasoning, detailed analysis
- **Pros**: Comprehensive responses, advanced reasoning
- **Install**: [Install Ollama](https://ollama.ai/) separately

## License

This project is licensed under the MIT License.