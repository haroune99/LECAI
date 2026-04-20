# What I'd Ship Next — Concrete Roadmap

## 1. Clarification Loop Before Planning

For ambiguous multi-intent queries (detected by low planner confidence score), add a node that asks the user one clarifying question before executing.

**Why it matters:** Query 9 (Longi Green Energy) can be interpreted as a sanctions check, a company profile, or an ROI calculation. Without clarification, the agent commits early and may answer the wrong interpretation.

**What to build:** In the planner node, detect when `CONFIDENCE: low` — route to a clarification sub-state that returns a question to the user. The main loop pauses until the user answers.

**Estimated effort:** 1 day.

---

## 2. Per-User Session Memory with LangGraph Checkpointing

With `SqliteSaver` (already in the graph), persist conversation history per user session. Enable follow-up queries like "do the same calculation for Shanghai" without re-establishing context.

**Why it matters:** The current build is stateless — each query starts fresh. LEC staff would naturally want to iterate on a query. Without session memory, they re-type context every time.

**What to build:** Add `thread_id` as a session identifier. Use it for checkpointing. Store session summaries in SQLite after 10+ messages to compress history.

**Estimated effort:** 2 hours — the infrastructure is already wired.

---

## 3. Automated KB Refresh Pipeline

A scheduled job (cron or Airflow) that re-ingests documents with expired TTL, re-fetches HMRC tariff data from gov.uk CSV APIs, updates BM25 and vector indexes incrementally.

**Why it matters:** UK Global Tariff codes update quarterly. OFSI sanctions are updated continuously. A stale KB is a compliance risk — the agent might return an outdated duty rate or miss a new sanctions listing.

**What to build:** Per-entry TTL metadata in SQLite. A refresh script that checks `last_updated < now - 90 days` and triggers re-fetch. Integration with gov.uk open data APIs for automatic tariff CSV download.

**Estimated effort:** 1 day.

---

## 4. Streaming Responses in Streamlit

With LangGraph's streaming API + `st.write_stream`, users see tokens appear as the agent reasons — plan being generated, tool calls firing, reflection happening.

**Why it matters:** Perceived latency is a product decision. A 4-second wait for a final answer feels longer than a 4-second stream of partial responses. For executive demos, streaming is the difference between "that feels slow" and "wow, look at it thinking."

**What to build:** Replace `graph.ainvoke` with `graph.astream` in Streamlit. Yield intermediate states via `st.write_stream`. Show the reasoning trace incrementally.

**Estimated effort:** 3 hours.

---

## 5. Tool Result Caching with Redis

Many trade regulation queries are idempotent — "what's the duty rate for 2203.00?" returns the same answer every time. A Redis cache with a TTL matching the KB refresh cycle (90 days) cuts latency and API cost significantly.

**Why it matters:** At 1,000 queries/day, the trade_calculator and trade_regulations_lookup tools fire ~2,000 times. If 40% are repeat queries, that's 800 cached responses × ~50ms saved = 40 seconds of user wait time eliminated per day. More importantly, Tavily search calls are cached and don't count against the free tier.

**What to build:** Redis client with `@cache(ttl=86400 * 90)` decorator on tool functions. Cache key = hash of (tool_name, kwargs). Invalidate on KB refresh.

**Estimated effort:** 4 hours.
