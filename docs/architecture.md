# Architecture Decisions — LEC Trade Intelligence Agent

## Stack Choices

| Component | Choice | Rationale | Rejected Alternatives |
|---|---|---|---|
| Orchestration | LangGraph | State machine maps 1:1 to plan→execute→reflect→answer loop; checkpointing enables mid-run inspection; conditional edges handle retry logic without custom code | Raw Python+asyncio (reinvents too much); LangChain (obscures decisions) |
| LLM | MiniMax-M2.7 | $0.30/M input / $1.20/M output; 204,800 token context; native tool calling via OpenAI-compatible API; agentic reasoning built in | Claude Sonnet (more expensive, used as eval judge fallback only); GPT-4o (not model-agnostic) |
| Vector store | ChromaDB local | No server, persists to disk, metadata filtering, free | Qdrant (server required); Pinecone (external dependency) |
| BM25 | rank_bm25 (BM25Okapi) | Pure Python, no server, fast | Elasticsearch (overkill) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | 384 dims, CPU-friendly, runs locally | OpenAI embeddings (external API, cost) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Local CPU, 6-v2 is fast | Cohere (external API) |
| Web search | Tavily API | Purpose-built for LLM agents, returns pre-extracted content | SerpAPI (raw HTML, more parsing) |
| KB | SQLite | Zero-config, structured lookups for tariffs/sanctions, works offline | Postgres (overkill for this scope) |
| API server | FastAPI | Clean REST interface for `/search` and `/agent/run` | Flask (less modern) |
| Demo UI | Streamlit | Password-protected, reasoning trace panel, budget meter, fast to build | Gradio (looks like ML demo) |
| Eval | Custom Python + LLM-as-judge | Real scores against 10 queries with rubric, not vibes | Human scoring only (not scalable) |

## State Machine Diagram

```
┌──────────────────────────────────────┐
│              START                   │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│      PLANNER NODE                     │
│  - Reads user_query                  │
│  - Outputs plan_text + plan_steps     │
│  - Enumerates tool calls w/ deps      │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│      EXECUTOR NODE                    │
│  - Dispatches ready tool calls        │
│  - Records ToolResults                │
│  - Increments iteration counter      │
│  - Checks budget_exceeded            │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│      ITERATION GUARD                  │
│  if iteration >= max_iterations      │
│  → force-terminate → ANSWERER         │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│      REFLECTOR NODE                   │
│  - Reviews ToolResults               │
│  - Decides: sufficient / insufficient │
│    / tool_failed                     │
└──────┬──────────────┬────────────────┘
       │              │                 │
   sufficient      insufficient      tool_failed
       │              │                 │
       ▼              ▼                 ▼
  ┌─────────┐  ┌───────────┐  ┌─────────────┐
  │ANSWERER │  │ EXECUTOR  │  │ EXECUTOR    │
  │         │  │ (new calls)│  │ (retry)     │
  └────┬────┘  └─────┬─────┘  └──────┬──────┘
       │             │                │
       └─────────────┴────────────────┘
                     │
                     ▼
              ┌──────────┐
              │   END     │
              └──────────┘
```

## Trade-offs

**MiniMax vs Claude Sonnet:** MiniMax is 4x cheaper ($0.30 vs $1.20/M output tokens), has a larger context window (204,800 vs 200,000), and was the natural choice given this is a MiniMax-affiliated role. The risk is less battle-testing for agentic tool-calling workflows versus Claude. Mitigated by using Claude as the eval judge and fallback.

**ChromaDB local vs server mode:** Local mode means no concurrency support — only one process can write at a time. For a demo that's fine. For 100 concurrent users, ChromaDB server mode or migrating to Qdrant is the fix.

**SQLite KB vs Postgres:** SQLite is fine for a single-user demo. Under concurrent writes it will queue. The KB is read-heavy (every query hits it) so it's mostly fine as-is. Postgres would be the production choice.

**Hand-rolled agent vs LangChain:** The original plan used raw Python+asyncio to maximise transparency. LangGraph was chosen instead because the conditional edge routing and state checkpointing handle the retry logic and session persistence that would otherwise require significant custom code. The tradeoff is a framework dependency — mitigated by the fact that all decisions are visible and attributable.

## Cost Model

- MiniMax-M2.7: $0.30/M input tokens, $1.20/M output tokens
- Typical 4-step query: ~3,500 input + 600 output tokens = ~$0.00177/query
- Projected cost per 1,000 queries: ~$1.77
- Hard budget cap: $0.50 per run

## Concurrency Analysis (What Breaks at 100 Users)

1. **MiniMax API rate limits:** Not publicly documented, but per-key TPM limits apply. Fix: token bucket rate limiter middleware.
2. **SQLite write contention:** Under 100 concurrent checkpoint writes, SQLite queues. Fix: PostgresSaver.
3. **Per-user budget isolation:** Current $0.50 cap is per-run, not per-user. Fix: per-user budget pools tracked in DB.
4. **Tavily free tier exhaustion:** 1,000 searches/month. 100 users × 10 queries = 1,000. Fix: paid tier or Redis cache.
5. **ChromaDB local write serialization:** Concurrent reads fine, writes serialize. Fix: server mode or Qdrant.
