import os
import re
import time
import json
import httpx
from typing import Optional
from src.tools.base import ToolResult


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_BASE_URL = "https://api.tavily.io/v1"


def market_intelligence_search(
    query: str,
    domain_filter: str = "general",
    recency_days: int = 180,
) -> ToolResult:
    start = time.time()

    enriched_query = query
    if domain_filter != "general":
        enriched_query = f"[{domain_filter.upper()}] {query}"

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{TAVILY_BASE_URL}/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": enriched_query,
                    "search_depth": "basic",
                    "max_results": 5,
                    "include_answer": True,
                    "include_raw_content": False,
                },
            )
            data = response.json()

        if response.status_code != 200:
            return ToolResult(
                call_id="local",
                tool_name="market_intelligence_search",
                status="error",
                content={},
                latency_ms=int((time.time() - start) * 1000),
                error_message=f"Tavily API error: {response.status_code}",
            )

        results = data.get("results", [])
        answer = data.get("answer", "")

        formatted_results = []
        for r in results:
            formatted_results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", "")[:300],
                "score": r.get("score", 0),
            })

        return ToolResult(
            call_id="local",
            tool_name="market_intelligence_search",
            status="success",
            content={
                "query": query,
                "enriched_query": enriched_query,
                "answer": answer,
                "results": formatted_results,
                "domain_filter": domain_filter,
            },
            latency_ms=int((time.time() - start) * 1000),
        )

    except Exception as e:
        return ToolResult(
            call_id="local",
            tool_name="market_intelligence_search",
            status="error",
            content={},
            latency_ms=int((time.time() - start) * 1000),
            error_message=str(e),
        )
