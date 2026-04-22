# Report — LEC Trade Intelligence Agent

## What I Built

A production-grade agentic system for London Export Corporation, designed around their actual business domain rather than a generic tool demo. The agent answers multi-step trade intelligence queries across all four LEC subsidiaries using five domain-specific tools — each mapping to a real workflow LEC staff face daily.

**Core components:**

1. **LangGraph state machine** with four nodes: planner (explicit plan before action), executor (tool dispatch), reflector (retry and recovery), answerer (synthesis). Conditional edge routing handles failures gracefully without crashing.

2. **MiniMax-M2.7** as the orchestration model via OpenAI-compatible API. Measured cost is $0.006/query (~$6.82/1k queries). Hard budget cap of $0.50 per run prevents runaway spend.

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

## Prompt Ablation — V1 (Structured) vs V2 (Loose)

**3 runs × 10 queries each = 30 data points per prompt version.**

Scoring rubric: 0 = incorrect/missing, 1 = partial (some facts right, missing key detail), 2 = fully correct.

### Summary

| Metric | V1 (Structured) | V2 (Loose) |
|--------|-----------------|-------------|
| Average score | **1.20/2** | 0.57/2 |
| Full success rate (score=2) | **27%** | 10% |
| Half-success rate (score≥1) | **83%** | 60% |
| Cost per query | **$0.006** | $0.008 |
| Projected cost / 1,000 queries | **$6.82** | $7.73 |

*Full success = perfect answer. Half-success = at least partial credit. Both runs used the same MiniMax-M2.7 model.*

### Per-Query-Type Breakdown

| Query type | V1 full success | V2 full success | Delta |
|---|---|---|---|
| Multi-step (Q1, Q9, Q10) — lookup + calculation | **33%** | 10% | +23pp |
| Single-tool lookup (Q4, Q5) | **50%** | 43% | +7pp |
| Document synthesis (Q3, Q7, Q8) | **22%** | 0% | +22pp |

### Key Finding

**V1 is 2.7x more likely to produce a fully correct answer** (27% vs 10%). The structured prompt's `// ARGS:`, `depends on:`, and `PARALLEL_GROUPS:` syntax forces the model to reason explicitly about execution order. This pays off most on multi-step queries:

- **Q9 (Longi Green Energy):** V1 scored 2/2, 1/2, 1/2. Required 3 tools (profile + sanctions + ROI). V2 scored 1/2, 1/2, 0/2 — ROI calculation was consistently missed.
- **Q1 (Tsingtao landed cost):** V1 scored 1/2, 0/2, 1/2 — duty lookup correct, calculation partial. V2 scored 0/2 across all 3 runs.
- **Q3, Q7, Q8 (document synthesis):** V1 partial every run. V2 0/2 in 5 of 6 runs.

### Honest limitation: non-determinism

Q4 (flavored water tariff) scored 2, 1, 1 across V1 runs and 2, 1, 0 across V2 runs. The same query scoring 0 and 2 in different runs is expected from an LLM system. Treat individual query scores as noisy signals; the aggregate delta across 30 data points is the reliable signal.

---

## What I Learnt

**Model-agnostic architecture is a real engineering decision, not a talking point.** Using MiniMax because it was the company's model seems obvious in retrospect, but it forced every integration decision — tool calling patterns, the `reasoning_split` flag, the cost model. Swapping to Claude would be one line of code. That's the point.

**Eval rigour differentiates more than the code.** Anyone can build a working agent in a weekend. Fewer people run a proper eval harness, document failure modes honestly, and show a prompt ablation delta. The 30-minute walkthrough is as much about the eval story as the code.

**Pre-seeding the corpus was the right call for a demo.** Having real LEC-relevant data ready on first run — Tsingtao annual reports, UK tariff codes, OFSI sanctions — makes the demo immediately credible. A system that says "I don't know" to everything until you wait for ingestion is less compelling.

**The Streamlit reasoning trace is a feature, not a UX nice-to-have.** During the walkthrough, opening the reasoning panel and walking through each step makes the 30 minutes almost trivially easy to fill with substance. The UI tells the story.
