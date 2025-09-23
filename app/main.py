# main.py
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from retreiver import Retriever
from summarizer import make_generator, generate_answer, BACKEND
import os
import uvicorn
import PyPDF2

app = FastAPI(title="Deep Researcher - RAG Demo")

retriever = Retriever()
generator = make_generator()

class QueryReq(BaseModel):
    query: str
    top_k: int = 5
    max_words: int | None = None
    backend: str | None = None

@app.post("/upload")
async def upload_docs(files: list[UploadFile] = File(...)):
    """Upload PDFs, extract text, and build FAISS index dynamically."""
    texts = []
    for file in files:
        pdf = PyPDF2.PdfReader(file.file)
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                # simple chunking by ~500 chars
                for i in range(0, len(text), 500):
                    chunk = text[i:i+500]
                    if chunk.strip():
                        texts.append(chunk)

    if not texts:
        return {"error": "No text extracted from PDFs"}

    retriever.build_index(texts)
    return {"message": f"✅ Uploaded and indexed {len(texts)} chunks from {len(files)} PDFs"}

@app.post("/query")
def run_query(body: QueryReq):
    global generator
    if retriever.index is None:
        return {"answer": "⚠️ No documents uploaded yet.", "sources": [], "error": None}

    # If a specific backend is requested, create appropriate generator
    current_generator = generator
    if body.backend and body.backend.lower() in ["lexrank", "t5", "ollama"]:
        current_generator = make_generator(body.backend.lower())

    q = body.query
    hits = retriever.query(q, top_k=body.top_k)
    contexts = [h["meta"]["text"] for h in hits]
    sources = [{
        "doc_id": h["meta"]["doc_id"],
        "chunk_id": int(h["meta"]["chunk_id"]),  # Convert numpy.int64 to Python int
        "score": float(h["score"]),  # Also ensure score is a Python float
        "snippet": h["meta"]["text"][:300]
    } for h in hits]

    answer = generate_answer(q, contexts, current_generator,
                             max_words=body.max_words,
                             backend=body.backend)
    return {"answer": answer, "sources": sources}

@app.get("/backend")
def get_backend():
    return {
        "backend": BACKEND,
        "model": os.getenv("OLLAMA_MODEL", None)
    }

@app.post("/set_backend")
def set_backend(backend: str):
    """Set the global backend for queries."""
    global generator
    backend = backend.lower()
    if backend not in ["lexrank", "t5", "ollama"]:
        return {"error": f"Invalid backend: {backend}"}
    
    # Update generator for new backend
    generator = make_generator(backend)
    
    return {
        "message": f"✅ Backend set to {backend}", 
        "backend": backend,
        "model": os.getenv("OLLAMA_MODEL", None) if backend == "ollama" else None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
