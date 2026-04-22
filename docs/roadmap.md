# What I'd Ship Next — Concrete Roadmap

One week of additional development time. These are ordered by impact-to-effort ratio.

## 1. Replace Regex Plan Parsing with Structured Outputs
**Estimated: 4 hours**

The current `parse_plan_steps` uses hand-rolled regex against the planner's plain-text output. If the model produces "Step 1 —" instead of "Step 1:", the parser silently returns `[]` and the executor has nothing to dispatch. The right fix is JSON-mode prompting.

**What the structured planner output looks like:**
```json
{
  "steps": [
    {
      "step_id": "s1",
      "action": "Look up commodity code 2203 duty rate",
      "tool": "trade_regulations_lookup",
      "tool_input": {"query_type": "tariff", "commodity_code": "2203000000"},
      "depends_on": []
    },
    {
      "step_id": "s2",
      "action": "Calculate landed cost for 10,000 cases",
      "tool": "trade_calculator",
      "tool_input": {"operation": "landed_cost", "params": {"...": "..."}},
      "depends_on": ["s1"]
    }
  ],
  "parallel_groups": [["s1"], ["s2"]],
  "reasoning": "Need duty rate before landed cost calculation.",
  "confidence": "high"
}
```

This also unlocks real parallel execution — `parallel_groups` is machine-readable, not parsed from prose.

---

## 2. Parallel Tool Execution
**Estimated: 3 hours (depends on Item 1)**

Implement `asyncio.gather()` dispatch of independent tool calls. Groups execute sequentially; calls within a group execute in parallel.

Query 9 (Longi: profile + sanctions + ROI) has two independent first steps. Sequential execution adds ~300ms of unnecessary latency per query.

```python
# executor.py — dispatch groups in parallel:
async def dispatch_parallel_group(tool_calls: list[ToolCall]) -> list[ToolResult]:
    tasks = [execute_single_tool(tc) for tc in tool_calls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [handle_result(tc, r) for tc, r in zip(tool_calls, results)]
```

---

## 3. Semantic Query Cache
**Estimated: 4 hours**

A cache that returns a stored answer for queries semantically similar to a previous query — not just exact key matches.

"Why is the UK import duty on Tsingtao beer?" and "What's the current UK duty rate for beer from China?" are different strings but the same query. A key-value cache misses the second one. A semantic cache embeds the incoming query, finds the nearest stored query by cosine similarity, and returns the cached answer if similarity exceeds 0.92.

Use the existing `all-MiniLM-L6-v2` embedder — no new dependencies. Add TTL per cache entry (90 days for regulatory data, 1 day for web search results).

---

## 4. LangSmith Observability Integration
**Estimated: 30 minutes**

Wire LangGraph's native LangSmith integration. Set three env vars and you get full traces — LLM calls, token counts, tool dispatch, routing decisions — in a visual dashboard.

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=lec-trade-agent
```

Free tier is sufficient for evaluation volumes. In production, you need to know which queries are slow, which tools fail most, and when a planner produces a bad plan. Right now you'd be toggling debug logs manually.

---

## 5. Adaptive Retrieval Weights Based on Query Type
**Estimated: 1 day including eval delta measurement**

BM25 outperforms semantic search on exact-match queries ("what is commodity code 220300"). Semantic search outperforms BM25 on conceptual queries ("what are LEC's China trade principles"). Fixed weights (BM25=0.3, semantic=0.5, reranker=0.2) are a compromise suboptimal for both.

```python
class AdaptiveWeightSelector:
    PROFILES = {
        "exact_lookup":  {"bm25": 0.6, "semantic": 0.3, "reranker": 0.1},
        "conceptual":   {"bm25": 0.1, "semantic": 0.7, "reranker": 0.2},
        "mixed":        {"bm25": 0.3, "semantic": 0.5, "reranker": 0.2},
    }

    def classify_and_select(self, query: str) -> dict:
        if re.search(r'\d{4,}', query) or any(w in query.lower() for w in ["code", "rate", "number"]):
            return self.PROFILES["exact_lookup"]
        if any(w in query.lower() for w in ["strategy", "principles", "history"]):
            return self.PROFILES["conceptual"]
        return self.PROFILES["mixed"]
```

Measure the delta in `precision@5` across eval queries. This becomes a second ablation to report — genuinely interesting to document.

---

## Priority Order

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 1 | JSON plan parsing | 4h | High — fixes silent failure mode |
| 2 | Parallel execution | 3h | Medium — latency on complex queries |
| 3 | Semantic cache | 4h | High — reduces cost/latency on repeated queries |
| 4 | LangSmith | 30min | High — production observability |
| 5 | Adaptive weights | 1d | Medium — research signal for report |