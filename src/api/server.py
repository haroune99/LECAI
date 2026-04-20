import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio

app = FastAPI(title="LEC Trade Intelligence Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    filters: Optional[dict] = None
    top_k: int = 5
    bm25_weight: float = 0.3
    semantic_weight: float = 0.5


class AgentRequest(BaseModel):
    query: str
    session_id: str = "default"
    prompt_version: str = "v1"
    max_iterations: int = 8
    budget_cap_usd: float = 0.50


@app.get("/")
async def root():
    return {"status": "ok", "service": "LEC Trade Intelligence Agent"}


@app.post("/search")
async def search(request: SearchRequest):
    from src.retrieval.hybrid_searcher import HybridSearcher

    try:
        searcher = HybridSearcher()
        results = searcher.search(
            query=request.query,
            filters=request.filters,
            top_k=request.top_k,
            bm25_weight=request.bm25_weight,
            semantic_weight=request.semantic_weight,
        )
        return {"query": request.query, "results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/run")
async def run_agent(request: AgentRequest):
    from src.agent.graph import build_graph
    from src.agent.state import default_state

    try:
        graph = build_graph()
        state = default_state()
        state["user_query"] = request.query
        state["session_id"] = request.session_id
        state["prompt_version"] = request.prompt_version
        state["max_iterations"] = request.max_iterations
        state["budget_cap_usd"] = request.budget_cap_usd

        config = {"configurable": {"thread_id": request.session_id}}
        result = await graph.ainvoke(state, config=config)

        return {
            "query": request.query,
            "final_answer": result.get("final_answer", ""),
            "sources_cited": result.get("sources_cited", []),
            "tool_results": result.get("tool_results", []),
            "run_status": result.get("run_status", "unknown"),
            "tokens_input": result.get("tokens_input", 0),
            "tokens_output": result.get("tokens_output", 0),
            "cost_usd": result.get("cost_usd", 0.0),
            "reasoning_trace": result.get("reasoning_trace", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
