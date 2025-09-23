# 🔬 Deep Researcher - RAG Document Analysis System

A powerful Retrieval-Augmented Generation (RAG) system that enables intelligent document querying with multiple AI backends. Upload documents, ask questions, and get comprehensive answers powered by state-of-the-art language models.

## 🌟 Features

- **📄 Multi-format Document Support**: PDF processing and text extraction
- **🤖 Multiple AI Backends**: 
  - **LexRank**: Extractive summarization for quick insights
  - **DistilBART**: Abstractive summarization for coherent responses
  - **Ollama (LLaMA3)**: Large language model for comprehensive analysis
- **🔍 Vector Search**: FAISS-powered semantic search with sentence transformers
- **🌐 Web Interface**: Clean Streamlit UI with real-time backend switching
- **🚀 REST API**: FastAPI backend with comprehensive documentation
- **🐳 Docker Support**: Containerized deployment with docker-compose
- **☁️ Cloud Access**: Cloudflare tunnel integration for public deployment

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │───▶│   FastAPI Backend │───▶│  AI Backends    │
│   (Port 8501)   │    │   (Port 8000)    │    │  - LexRank      │
└─────────────────┘    └──────────────────┘    │  - DistilBART   │
                                │               │  - Ollama       │
                                ▼               └─────────────────┘
                       ┌──────────────────┐
                       │  FAISS Vector DB │
                       │  + Document Store │
                       └──────────────────┘
```

## 🚀 Live Demo

**🌐 Main Application**: [https://deepresearcher.page](https://deepresearcher.page)  
**📖 API Documentation**: [https://api.deepresearcher.page/docs](https://api.deepresearcher.page/docs)

## 📋 Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) (for LLM backend)
- Docker & Docker Compose (optional)
- [Cloudflare Account](https://cloudflare.com/) (for public deployment)

## 🛠️ Installation

### Option 1: Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd deep-researcher
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv310
   source venv310/bin/activate  # On Windows: venv310\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
   ```

5. **Install and start Ollama**
   ```bash
   # Install Ollama from https://ollama.ai/
   ollama serve &
   ollama pull llama3
   ```

### Option 2: Docker Deployment

1. **Start with Docker Compose**
   ```bash
   docker-compose up -d --build
   ```

2. **Ensure Ollama is running on host**
   ```bash
   ollama serve &
   ollama pull llama3
   ```

## 🎯 Usage

### Starting the Application

**Local Development:**
```bash
# Terminal 1: Start FastAPI backend
source venv310/bin/activate
cd app
python main.py

# Terminal 2: Start Streamlit UI
source venv310/bin/activate
cd app
streamlit run streamlit_app.py --server.port 8501

# Terminal 3: Start Ollama (if not running)
ollama serve
```

**Docker:**
```bash
docker-compose up -d
```

### Using the Interface

1. **Upload Documents**: Drag and drop PDF files into the upload area
2. **Select AI Backend**: Choose between LexRank, DistilBART, or Ollama
3. **Ask Questions**: Type your questions about the uploaded documents
4. **Get Answers**: Receive contextual responses based on document content

### API Endpoints

- `POST /upload` - Upload and process documents
- `POST /query` - Query documents with specific backend
- `GET /backend` - Get current active backend
- `POST /set_backend` - Switch between AI backends
- `GET /docs` - Interactive API documentation

## 🔧 Configuration

### Environment Variables

```bash
# Backend Configuration
SUMMARIZER_BACKEND=ollama          # Default backend: lexrank, distilbart, ollama
OLLAMA_MODEL=llama3               # Ollama model to use
OLLAMA_HOST=localhost:11434       # Ollama server address

# Docker Configuration
OLLAMA_HOST=host.docker.internal:11434  # For Docker deployments
```

### Cloudflare Tunnel Setup

For public deployment with custom domain:

1. **Install cloudflared**
   ```bash
   # macOS
   brew install cloudflare/cloudflare/cloudflared
   
   # Other platforms: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/
   ```

2. **Authenticate with Cloudflare**
   ```bash
   cloudflared tunnel login
   ```

3. **Create tunnel**
   ```bash
   cloudflared tunnel create deepresearcher
   ```

4. **Configure DNS routes**
   ```bash
   cloudflared tunnel route dns deepresearcher deepresearcher.yourdomain.com
   cloudflared tunnel route dns deepresearcher api.deepresearcher.yourdomain.com
   ```

5. **Install as system service**
   ```bash
   sudo cloudflared service install
   ```

## 🏗️ Project Structure

```
deep-researcher/
├── app/
│   ├── main.py              # FastAPI backend server
│   ├── streamlit_app.py     # Streamlit web interface
│   ├── retreiver.py         # Document retrieval and vector search
│   ├── summarizer.py        # AI backend implementations
│   ├── ingest_and_index.py  # Document processing utilities
│   ├── sample_docs/         # Example documents
│   └── uploaded_docs/       # User uploaded documents
├── venv310/                 # Python virtual environment
├── .cloudflared/           # Cloudflare tunnel configuration
├── docker-compose.yml      # Docker deployment configuration
├── Dockerfile             # Container build instructions
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🧠 AI Backends

### LexRank
- **Type**: Extractive summarization
- **Use Case**: Quick document insights, key sentence extraction
- **Pros**: Fast, preserves original text
- **Cons**: Limited creativity, may miss context

### DistilBART
- **Type**: Abstractive summarization
- **Use Case**: Coherent summaries, text generation
- **Pros**: Natural language generation, contextual understanding
- **Cons**: Moderate resource usage

### Ollama (LLaMA3)
- **Type**: Large Language Model
- **Use Case**: Complex reasoning, detailed analysis
- **Pros**: Comprehensive responses, advanced reasoning
- **Cons**: Requires more computational resources

## 🔍 Technical Details

### Vector Search
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Vector Database**: FAISS with flat index
- **Similarity**: Cosine similarity for document retrieval
- **Chunk Size**: Optimized for document context

### Document Processing
- **Supported Formats**: PDF (via PyPDF2)
- **Text Extraction**: Automatic preprocessing and cleaning
- **Storage**: Local file system with metadata tracking

## 🚀 Deployment Options

### Local Development
Perfect for testing and development with full control over all components.

### Docker Container
Simplified deployment with consistent environment across platforms.

### Cloud Deployment
Public access via Cloudflare tunnels with custom domain support.

## 🛡️ Security Considerations

- Document uploads are stored locally
- No data persistence beyond session for uploaded content
- API endpoints are rate-limited
- Cloudflare provides DDoS protection and SSL termination

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [FastAPI](https://fastapi.tiangolo.com/) for the robust API framework
- [Ollama](https://ollama.ai/) for local LLM serving
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search
- [Sentence Transformers](https://www.sbert.net/) for text embeddings
- [Cloudflare](https://cloudflare.com/) for tunnel infrastructure

## 📞 Support

For questions, issues, or feature requests, please open an issue in the GitHub repository.

---

**🔬 Happy Researching! 🔬**