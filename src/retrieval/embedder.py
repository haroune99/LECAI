import os
import numpy as np
from typing import Optional


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]
