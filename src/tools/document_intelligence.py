import os
import time
import json
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from src.retrieval.hybrid_searcher import HybridSearcher
from src.tools.base import ToolResult

load_dotenv(override=True)

INDEX_DIR = "data/indexes"
os.makedirs(INDEX_DIR, exist_ok=True)


SYSTEM_CONTEXT = """You are a document intelligence assistant for London Export Corporation (LEC).
LEC has four subsidiaries:
- LEC Beverages: UK importer and exclusive distributor of Tsingtao beer
- LEC Robotics: Service automation and robotics solutions
- LEC Industries: AI solutions in industry, infrastructure, healthcare
- LEC Global Capital: Fund management in tech, healthcare, life sciences, renewables

Use ONLY the provided document chunks to answer the query. Do not make up information.
If the documents do not contain enough information, say so explicitly.
"""


def _synthesize_answer(query: str, context_text: str) -> tuple[str, int, int]:
    from openai import OpenAI
    api_key = os.getenv("MINIMAX_API_KEY", "")
    client = OpenAI(api_key=api_key, base_url="https://api.minimax.io/v1").chat.completions

    messages = [
        {"role": "system", "content": SYSTEM_CONTEXT},
        {"role": "user", "content": f"Query: {query}\n\nDocument excerpts:\n{context_text}"},
    ]

    response = client.create(
        model="MiniMax-M2.7",
        messages=messages,
        temperature=0.3,
    )

    answer = response.choices[0].message.content or ""
    input_tokens = response.usage.prompt_tokens if hasattr(response, "usage") else 0
    output_tokens = response.usage.completion_tokens if hasattr(response, "usage") else 0

    return answer, input_tokens, output_tokens


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

        context_text = "\n\n---\n\n".join(
            f"[Source: {r['metadata'].get('source', 'unknown')} | Score: {r['scores']['final']:.3f}]\n{r['text'][:500]}"
            for r in results
        )

        synthesis_answer, in_tokens, out_tokens = _synthesize_answer(query, context_text)
        total_tokens = in_tokens + out_tokens

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
                "scores": [r["scores"] for r in results] if return_scores else {},
                "answer": synthesis_answer,
            },
            latency_ms=int((time.time() - start) * 1000),
            tokens_used=total_tokens,
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
