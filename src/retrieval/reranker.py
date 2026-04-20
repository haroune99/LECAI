import numpy as np
from typing import Optional


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, max_length=512)
        self.model_name = model_name

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        reranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        output = []
        for rank, (score, candidate) in enumerate(reranked[:top_k], 1):
            candidate["reranker_score"] = float(score)
            candidate["rank"] = rank
            output.append(candidate)
        return output
