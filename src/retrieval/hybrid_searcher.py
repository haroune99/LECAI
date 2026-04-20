from typing import Optional
from src.retrieval.embedder import Embedder
from src.retrieval.bm25_index import BM25Index
from src.retrieval.vector_store import VectorStore
from src.retrieval.reranker import Reranker


class HybridSearcher:
    def __init__(self):
        self.embedder = Embedder()
        self.bm25_index = BM25Index()
        self.bm25_index.load(name="bm25_main")
        self.vector_store = VectorStore()
        self.reranker = Reranker()

    def search(
        self,
        query: str,
        filters: Optional[dict] = None,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.5,
        reranker_weight: float = 0.2,
        top_k: int = 5,
    ) -> list[dict]:
        bm25_results = self.bm25_index.search(query, top_k=top_k * 3)
        query_embedding = self.embedder.embed_query(query)
        semantic_results = self.vector_store.search(
            query_embedding=query_embedding.tolist(),
            filters=filters,
            top_k=top_k * 3,
        )

        fused = self._reciprocal_rank_fusion(
            [bm25_results, semantic_results],
            weights=[bm25_weight, semantic_weight],
        )

        if not fused:
            return []

        reranked = self.reranker.rerank(query=query, candidates=fused, top_k=top_k)

        results = []
        for r in reranked:
            bm25_score = r.get("bm25_score", 0)
            semantic_score = r.get("semantic_score", 0)
            reranker_score = r.get("reranker_score", 0)
            final_score = (bm25_weight * bm25_score) + (semantic_weight * semantic_score) + (reranker_weight * reranker_score)

            results.append({
                "text": r["text"],
                "chunk_id": r.get("chunk_id", ""),
                "metadata": r.get("metadata", {}),
                "scores": {
                    "bm25": bm25_score,
                    "semantic": semantic_score,
                    "reranker": reranker_score,
                    "final": final_score,
                },
            })

        return results

    def _reciprocal_rank_fusion(self, result_lists: list[list[dict]], weights: list[float], k: int = 60) -> list[dict]:
        scores = {}
        for results, weight in zip(result_lists, weights):
            for rank, result in enumerate(results, 1):
                doc_id = result.get("doc_id") or result.get("chunk_id", result.get("text", "")[:32])
                if doc_id not in scores:
                    scores[doc_id] = {"doc_id": doc_id, "text": result.get("text", ""), "metadata": result.get("metadata", {})}
                scores[doc_id]["bm25_score"] = result.get("bm25_score", 0)
                scores[doc_id]["semantic_score"] = result.get("semantic_score", 0)
                scores[doc_id]["rrf_score"] = scores[doc_id].get("rrf_score", 0) + (weight / (k + rank))

        ranked = sorted(scores.values(), key=lambda x: x.get("rrf_score", 0), reverse=True)
        return ranked
