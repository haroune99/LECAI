import os
import time
import json
from typing import Optional
from dataclasses import dataclass

from src.retrieval.hybrid_searcher import HybridSearcher
from src.tools.base import ToolResult


INDEX_DIR = "data/indexes"
os.makedirs(INDEX_DIR, exist_ok=True)


SYSTEM_CONTEXT = """
You are a document intelligence assistant for London Export Corporation (LEC).
LEC has four subsidiaries:
- LEC Beverages: UK importer and exclusive distributor of Tsingtao beer
- LEC Robotics: Service automation and robotics solutions
- LEC Industries: AI solutions in industry, infrastructure, healthcare
- LEC Global Capital: Fund management in tech, healthcare, life sciences, renewables

Use ONLY the provided document chunks to answer the query. Do not make up information.
If the documents do not contain enough information, say so explicitly.
"""

MODEL_FOR_SYNTHESIS = "MiniMax-M2.7"


@dataclass
class SynthesisResult:
    answer: str
    sources: list[dict]
    chunks_used: int
    synthesis_tokens_used: int


def document_intelligence(
    query: str,
    filters: Optional[dict] = None,
    top_k: int = 5,
    return_scores: bool = True,
) -> ToolResult:
    start = time.time()

    try:
        searcher = HybridSearcher()
        results = searcher.search(
            query=query,
            filters=filters,
            top_k=top_k,
            bm25_weight=0.3,
            semantic_weight=0.5,
            reranker_weight=0.2,
        )

        if not results:
            return ToolResult(
                call_id="local",
                tool_name="document_intelligence",
                status="success",
                content={"answer": "No relevant documents found for this query.", "chunks": [], "scores": {}},
                latency_ms=int((time.time() - start) * 1000),
            )

        chunks_for_context = []
        scores_for_context = []
        for r in results:
            chunks_for_context.append(r["text"])
            scores_for_context.append(r["scores"])

        context_text = "\n\n---\n\n".join(
            f"[Source: {r['metadata'].get('source', 'unknown')} | Score: {r['scores']['final']:.3f}]\n{r['text'][:500]}"
            for r in results
        )

        return ToolResult(
            call_id="local",
            tool_name="document_intelligence",
            status="success",
            content={
                "query": query,
                "chunks": [r["text"][:300] for r in results],
                "sources": [
                    {"source": r["metadata"].get("source", ""), "chunk_id": r.get("chunk_id", "")}
                    for r in results
                ],
                "scores": scores_for_context if return_scores else {},
                "answer": f"[Document search returned {len(results)} relevant chunks. Use these to synthesise an answer.]",
                "chunks_for_context": context_text,
            },
            latency_ms=int((time.time() - start) * 1000),
        )

    except Exception as e:
        return ToolResult(
            call_id="local",
            tool_name="document_intelligence",
            status="error",
            content={},
            latency_ms=int((time.time() - start) * 1000),
            error_message=str(e),
        )
