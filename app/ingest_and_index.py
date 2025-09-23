# ingest_and_index.py
import os
import json
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader

DATA_DIR = Path("sample_docs")
INDEX_PATH = Path("faiss_index.bin")
META_PATH = Path("doc_meta.json")
EMB_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 400
CHUNK_STRIDE = 200

def read_text_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def read_pdf(p: Path) -> str:
    text = []
    reader = PdfReader(str(p))
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

def load_documents(dir_path: Path) -> List[dict]:
    docs = []
    for f in dir_path.iterdir():
        if f.suffix.lower() in [".txt", ".md"]:
            text = read_text_file(f)
        elif f.suffix.lower() == ".pdf":
            text = read_pdf(f)
        else:
            continue
        docs.append({"id": f.name, "text": text})
    return docs

def chunk_text(text: str, chunk_size=CHUNK_SIZE, stride=CHUNK_STRIDE):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += stride
    return chunks

def build_index():
    print("Loading documents...")
    docs = load_documents(DATA_DIR)
    print(f"Found {len(docs)} docs.")
    model = SentenceTransformer(EMB_MODEL)
    all_embeddings = []
    metadata = []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for idx, ch in enumerate(chunks):
            metadata.append({
                "doc_id": doc["id"],
                "chunk_id": idx,
                "text": ch[:1000]
            })
    if not metadata:
        raise RuntimeError("No chunks created. Add files to sample_docs/")
    texts = [m["text"] for m in metadata]
    print(f"Creating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    print("Building FAISS index (FlatL2)...")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    print("✅ Index built and saved.")

if __name__ == "__main__":
    build_index()
