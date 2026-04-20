import os
from typing import Optional


INDEX_DIR = "data/indexes"
os.makedirs(INDEX_DIR, exist_ok=True)


class BM25Index:
    def __init__(self, index_dir: str = INDEX_DIR):
        self.index_dir = index_dir
        self.index = None
        self.doc_ids = []
        self.texts = []

    def build(self, chunks: list[dict]):
        from rank_bm25 import BM25Okapi
        self.texts = [c["text"] for c in chunks]
        self.doc_ids = [c["metadata"].get("chunk_id", str(i)) for i, c in enumerate(chunks)]
        if not self.texts:
            return
        tokenized = [text.lower().split() for text in self.texts]
        self.index = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if not self.index:
            return []
        scores = self.index.get_scores(query.lower().split())
        ranked = sorted(zip(scores, self.texts, self.doc_ids), key=lambda x: x[0], reverse=True)
        results = []
        for score, text, doc_id in ranked[:top_k]:
            if score > 0:
                results.append({"text": text, "doc_id": doc_id, "bm25_score": float(score)})
        return results

    def save(self, name: str = "bm25"):
        if not self.index:
            return
        import pickle
        path = os.path.join(self.index_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump({"index": self.index, "texts": self.texts, "doc_ids": self.doc_ids}, f)

    def load(self, name: str = "bm25"):
        import pickle
        path = os.path.join(self.index_dir, f"{name}.pkl")
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.index = data["index"]
        self.texts = data["texts"]
        self.doc_ids = data["doc_ids"]
