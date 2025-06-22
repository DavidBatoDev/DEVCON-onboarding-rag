# backend/src/app/services/vector_store.py
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimpleVectorStore:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.docs_path = os.path.join(dir_path, "docs.json")
        self.emb_path = os.path.join(dir_path, "embeddings.json")
        self._load()

    def _load(self):
        if os.path.exists(self.docs_path) and os.path.exists(self.emb_path):
            with open(self.docs_path, "r", encoding="utf-8") as f:
                self.docs = json.load(f)
            with open(self.emb_path, "r", encoding="utf-8") as f:
                self.embeddings = json.load(f)
        else:
            self.docs = []
            self.embeddings = []

    def similarity_search(self, query_embedding, top_k=3):
        # print(f"Performing similarity search with query embedding: {query_embedding[:10]}")
        if not self.embeddings:
            return []
        sims = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [self.docs[i] for i in top_indices]

def save_to_store(chunks, embeddings, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    docs_path = os.path.join(output_dir, "docs.json")
    emb_path = os.path.join(output_dir, "embeddings.json")

    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    with open(emb_path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)

def load_from_store():
    from app.core.config import settings
    return SimpleVectorStore(settings.RAG_INDEX_DIR)


