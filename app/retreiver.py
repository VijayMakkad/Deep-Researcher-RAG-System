# retriever.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []

    def build_index(self, texts: list[str]):
        """Build a FAISS index from a list of text chunks."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.texts = texts
        print(f"✅ Built FAISS index with {len(texts)} chunks")

    def query(self, q: str, top_k=5):
        """Search top_k chunks relevant to query."""
        if self.index is None:
            return []
        q_emb = self.model.encode([q], convert_to_numpy=True)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for rank, (i, score) in enumerate(zip(I[0], D[0])):
            if 0 <= i < len(self.texts):
                results.append({
                    "score": float(np.exp(-score)),  # convert distance → similarity-like
                    "meta": {
                        "doc_id": f"doc_{i//10}",  # simple doc grouping
                        "chunk_id": i,
                        "text": self.texts[i][:500]
                    }
                })
        return results
