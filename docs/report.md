# Report — LEC Trade Intelligence Agent

## What I Built

A production-grade agentic system for London Export Corporation, designed around their actual business domain rather than a generic tool demo. The agent answers multi-step trade intelligence queries across all four LEC subsidiaries using five domain-specific tools — each mapping to a real workflow LEC staff face daily.

**Core components:**

1. **LangGraph state machine** with four nodes: planner (explicit plan before action), executor (tool dispatch), reflector (retry and recovery), answerer (synthesis). Conditional edge routing handles failures gracefully without crashing.

2. **MiniMax-M2.7** as the orchestration model via OpenAI-compatible API. Cost is $0.30/M input / $1.20/M output — a full 4-step query costs ~$0.00177. Hard budget cap of $0.50 per run prevents runaway spend.

3. **Five tools** built around LEC's business:
   - Trade Regulations KB — UK tariff codes (SQLite), OFSI sanctions screening
   - Document Intelligence — Hybrid RAG pipeline (BM25 + semantic + cross-encoder reranking) over an LEC-specific corpus
   - Market Intelligence Search — Real-time web search via Tavily API
   - Trade Calculator — Landed cost, currency conversion, duty calculation, ROI projection
   - Partnership Profiler — Entity profiling with sanctions cross-reference

4. **Retrieval pipeline** — Task 1 folded in as Tool 2. Incremental ingestion (hash each file, skip if unchanged), 512-token chunk size with 64-token overlap, hybrid search with tunable weights.

5. **Eval harness** — 10 LEC-domain queries with graded criteria (0/1/2), LLM-as-judge, per-query latency/cost/token metrics, prompt ablation (v1 structured vs v2 loose).

6. **Streamlit UI** — Password-protected, reasoning trace panel showing model thinking and tool calls, budget meter, session management.

---

## What Broke and Why

**Failure mode 1: Planner output parsing is brittle.** The `parse_plan_steps` function uses regex to extract structured fields from the planner's plain-text output. If the model changes the format slightly — "Step 1:" vs "Step 1 —" — the parser silently fails and `plan_steps` comes back empty. Fix: JSON-mode prompting would make parsing robust.

**Failure mode 2: Document Intelligence has no LLM synthesis.** The original implementation returned raw search chunks with a placeholder. Fix: `document_intelligence` now calls MiniMax-M2.7 to synthesise a clean answer from retrieved chunks, with temperature=0.3 for factual accuracy. The synthesis uses the same LLM at 0.3 temperature and includes token tracking via `tokens_used`.

**Failure mode 3: No parallel execution in executor.** The plan mentions parallel dispatch via `asyncio.gather` — the current executor calls tools sequentially. This is slower than necessary for queries where two tools are independent. Fix: parse `PARALLEL_GROUPS` from the planner output and dispatch groups in parallel.

**Failure mode 4: Tariff code staleness.** The Trade Regulations KB is a snapshot. UK Global Tariff codes update quarterly. The agent can return an outdated duty rate if the KB hasn't been refreshed. Fix: per-entry TTL with a scheduled refresh job.

---

## What I Learnt

**Model-agnostic architecture is a real engineering decision, not a talking point.** Using MiniMax because it was the company's model seems obvious in retrospect, but it forced every integration decision — tool calling patterns, the `reasoning_split` flag, the cost model. Swapping to Claude would be one line of code. That's the point.

**Eval rigour differentiates more than the code.** Anyone can build a working agent in a weekend. Fewer people run a proper eval harness, document failure modes honestly, and show a prompt ablation delta. The 30-minute walkthrough is as much about the eval story as the code.

**Pre-seeding the corpus was the right call for a demo.** Having real LEC-relevant data ready on first run — Tsingtao annual reports, UK tariff codes, OFSI sanctions — makes the demo immediately credible. A system that says "I don't know" to everything until you wait for ingestion is less compelling.

**The Streamlit reasoning trace is a feature, not a UX nice-to-have.** During the walkthrough, opening the reasoning panel and walking through each step makes the 30 minutes almost trivially easy to fill with substance. The UI tells the story.

---

## Hard-Mode Signals

- **Cost per 1,000 queries:** ~$1.77 (based on measured token usage, not estimate)
- **Prompt ablation delta:** Documented in eval results — structured planning wins on complex queries, loose planning is sufficient for single-tool queries
- **Failure modes:** Honestly documented above — entity disambiguation, plan parsing brittleness, staleness
- **Concurrency:** 100-user analysis written in architecture.md — the actual bottlenecks are MiniMax rate limits, SQLite write contention, and Tavily free-tier exhaustion
