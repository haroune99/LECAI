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

---

## Data Pipeline — Ingestion, Parsing, and Chunking

### Overview

Every source document goes through the same pipeline:

```
Raw File → Parser (format-specific) → Chunker → Embedder → Vector Store + BM25
```

**Key design decisions:**
- **Incremental ingestion:** Each file is hashed (SHA256). If the hash is already in `data/indexes/processed_hashes.txt`, the file is skipped. Only changed files are re-ingested.
- **Chunk IDs are unique per source file:** Format is `{source}_{index}` where index is `chunk_index` for recursive splits, `r{row}` for CSVs, `p{page}` for PDFs — preventing cross-file ID collisions.
- **English-only for bilingual PDFs:** The Tsingtao Annual Report is Chinese|English side-by-side. A regex heuristic (`latin_chars / total_chars > 0.5`) extracts only the English half per page before chunking.

---

### File-by-File Pipeline

#### UK Tariff CSV (`uk-tariff-2021-01-01--v4.0.1477--measures-as-defined.csv`)

**Source:** gov.uk UK Global Tariff — official commodity codes and duty rates.

**Pre-processing (clean_corpus.py):**
- Renamed double-underscore columns to simple names:
  - `commodity__code` → `commodity_code`
  - `commodity__description` → `description`
  - `measure__duty_expression` → `duty_rate`
  - `measure__type__description` → `measure_type`
  - `measure__geographical_area__description` → `origin_rules`
- Skipped 37 metadata columns (Flags, Footnotes, Conditions, etc.) not needed for basic tariff lookup.
- Rows with non-numeric duty rates (e.g. `"10 + 77 GBP / 100 kg"`) set to `0.0` via `try/except float()` parse — no regex.

**Parser:** `parse_csv()` — returns raw CSV text.
```python
# First 3 columns of cleaned CSV:
commodity_code,description,measure_type,duty_rate,origin_rules
100000000,LIVE ANIMALS,Third country duty,0%,Channel Islands
2203000000,Beer made from malt,Tariff preference,0%,Developing Countries Trading Scheme (DCTS)
```

**Chunking strategy:** Recursive character split, 512 tokens, 64 token overlap.
- Each CSV row is long enough that one row exceeds 512 tokens, so each row becomes 1–2 chunks.
- Chunk ID: `{source}_{chunk_index}` (e.g. `uk-tariff...csv_0`, `uk-tariff...csv_1`)

**Output:** 82,874 tariff codes ingested into `tariff_codes` SQLite table (one row per commodity code — duplicates from multiple measure types merged via `INSERT OR IGNORE`).

**Note:** `2203000000` (beer made from malt) shows `0%` duty under the DCTS "Tariff preference" measure type. The actual UK duty on beer is a compound formula based on alcohol-by-volume. The 0% reflects only the preferential rate — not the standard rate. A production system would need the full DCTS duty calculation logic.

---

#### UK Sanctions List CSV (`UK-Sanctions-List.csv`)

**Source:** gov.uk Office of Financial Sanctions Implementation (OFSI) — publicly available CSV.

**Pre-processing (clean_corpus.py):**
- Deleted the metadata header row (row 0 was `"Report Date: 16-Apr-2026,..."` — not a data row).
- Column `Name 6` used as `entity_name` (the primary name field; `Name 1`–`Name 5` are aliases).
- `Name type`, `Address Country`, `Date Designated`, `UK Statement of Reasons` mapped to the SQLite schema.

**Parser:** `parse_csv()` — returns raw CSV text.

**Chunking strategy:** Recursive character split, 512 tokens, 64 token overlap.
- One row = one entity. Most entries are <512 tokens, so one chunk per entity.
- Chunk ID: `{source}_{chunk_index}`

**Output:** 56,748 sanctions entities ingested into `sanctions_entities` SQLite table.

---

#### HMRC Alcohol Duty Notice PDF (`Force_of_law_guidance_for_Alcohol_Duty.pdf`)

**Source:** gov.uk — Excise Notice 363: Alcohol Duty (Force of Law provisions).

**Parser:** `parse_pdf(extract_english_only=False)` — 9 pages, text extracted directly via pypdf. No bilingual splitting needed.

**Chunking strategy:** Recursive character split, 512 tokens, 64 token overlap.
- Pages split into ~42 chunks total (averaging ~4–5 chunks per page at 512 tokens/page).
- Chunk ID: `{source}_p{page_index}_{chunk_index}` (e.g. `Force_of_law...pdf_p0_0`)

**Output:** 42 chunks indexed in ChromaDB + BM25. Also ingested into `regulatory_requirements` SQLite table with `regulatory_body=HMRC` and `applies_to=beverages`.

---

#### LEC About Page CSV (`londonexportcorp (1).csv`)

**Source:** Browser-saved HTML from `londonexportcorp.com/about`, CMS export to CSV. Each cell contains an HTML fragment from the website's content management system.

**Pre-processing (clean_corpus.py):**
- BeautifulSoup parsed each CSV cell to extract clean text (removing `<script>`, `<style>`, nav, header, footer tags).
- Empty cells and HTML boilerplate (column headers like `heading richy-rich...`) were filtered out.
- 7 meaningful text sections extracted from the raw HTML-fragment CSV:
  1. Header row (skipped)
  2. "1953: A HISTORIC PARTNERSHIP TAKES SHAPE"
  3. "In July 1953, there was a pivotal moment in history. Jack Perry Senior led a delegation to China..."
  4. "ICE BREAKING"
  5. "Over 70 years have passed since Jack Perry Senior travelled to the International Economic Conference..."
  6. "RECENT TIMES"
  7. "In 1971 LEC was asked to pioneer the first trade deals between the USA and China..."

**Parser:** `parse_csv()` — returns concatenated section text joined by `\n\n`.
```python
# What parse_csv returns for this file:
"In July 1953, there was a pivotal moment in history. Jack Perry Senior led a delegation to China...\n\n"
"Over 70 years have passed since Jack Perry Senior travelled to the International Economic Conference..."
```

**Chunking strategy:** Recursive split, 512 tokens, 64 overlap. The 7 sections produce 7 clean chunks. Chunk ID: `{source}_r{row_index}`.

**Output:** 7 chunks indexed. Critical for Query 8 ("summarise LEC's 1953 founding deal").

---

#### LEC Past Works CSV (`londonexportcorp.csv`)

**Source:** Browser-saved HTML from `londonexportcorp.com/past-works`, CMS export to CSV.

**Pre-processing:** Same as the About page — HTML-fragment-in-CSV, cleaned via BeautifulSoup.

**Parser:** `parse_csv()`

**Chunking strategy:** 5 meaningful sections extracted:
1. LEC-Celanese partnership (tobacco manufacturing, 256 trillion cigarettes/year)
2. Cultural bridges (musicals, sports — Arsenal, West Bromwich Albion)
3. St. Louis Midwest hub (US-China investment)
4. [continuation of cultural bridges]
5. [continuation of St. Louis]

**Output:** 5 chunks indexed. Covers the major historical deals referenced in LEC's public history.

---

#### Tsingtao Annual Report PDF (`青岛啤酒2024年年报-20250423.pdf`)

**Source:** Tsingtao Brewery 2024 Annual Report (English/Chinese bilingual, 259 pages, 34 MB).

**Parser:** `parse_pdf(extract_english_only=True)` — per page:
```python
def split_pdf_by_language(text: str) -> str:
    """Keep only lines where latin_chars / total_chars > 0.5."""
    lines = text.split("\n")
    english_lines = []
    for line in lines:
        latin = len(re.findall(r"[a-zA-Z]", line))
        total = len(re.findall(r"[a-zA-Z\u4e00-\u9fff]", line))
        if total > 0 and latin / total > 0.5:
            english_lines.append(line)
    return "\n".join(english_lines)
```

**Chunking strategy:** After English extraction, pages are concatenated and recursively split at 512 tokens with 64-token overlap. 259 pages → 1,203 chunks.
- Chunk ID: `{source}_p{page_index}_{chunk_index}` (e.g. `青岛啤酒...pdf_p0_0`, `青岛啤酒...pdf_p0_1`)

**Output:** 1,203 chunks indexed. Contains Tsingtao's financial statements, distribution strategy, UK market data, and operational highlights.

**Limitation:** The bilingual PDF contains Chinese text labels interspersed with English financial figures. The regex heuristic works for paragraph text but may miss some English on pages with dense mixed scripts (tables with Chinese currency labels alongside English numbers).

---

### SQLite Knowledge Base Schema

```sql
CREATE TABLE tariff_codes (
    commodity_code TEXT PRIMARY KEY,   -- e.g. "2203000000"
    description TEXT,                  -- e.g. "Beer made from malt"
    category TEXT,                     -- e.g. "Tariff preference"
    uk_duty_rate REAL,               -- e.g. 0.0 (percentage)
    vat_rate REAL,                    -- e.g. 20.0
    origin_rules TEXT,                 -- e.g. "Developing Countries Trading Scheme..."
    restrictions TEXT,
    last_updated DATE
);

CREATE TABLE sanctions_entities (
    entity_name TEXT,                -- Primary name from Name 6 column
    entity_type TEXT,                -- e.g. "Primary Name"
    jurisdiction TEXT,               -- e.g. "United Arab Emirates"
    listing_date DATE,              -- e.g. "29/06/2012"
    reason TEXT,                    -- OFSI listing reason text
    source TEXT                      -- Always "OFSI"
);

CREATE TABLE regulatory_requirements (
    category TEXT,                    -- "import"
    requirement TEXT,                -- Text excerpt from HMRC PDF
    regulatory_body TEXT,            -- "HMRC"
    applies_to TEXT,                 -- "beverages"
    notes TEXT                      -- Source document reference
);
```

### ChromaDB Index Details

- **Collection:** `lec_documents`
- **Persisted at:** `data/indexes/chroma/`
- **BM25 index:** `data/indexes/bm25_main.pkl`
- **Processed hashes:** `data/indexes/processed_hashes.txt` (SHA256 of each ingested file)
- **Unique chunk ID format:** `{source_filename}_{index_type}{index}` — prevents cross-file collisions
- **Metadata per chunk:** `source`, `file_type`, `chunk_index`, `row_index` (CSVs), `page_index` (PDFs), `document_hash`
- **Hybrid search weights:** BM25=0.3, Semantic=0.5, Reranker=0.2

### Corpus Statistics

| File | Format | Chunks | SQLite Table |
|---|---|---|---|
| UK Global Tariff | CSV | 82,874 (rows) | tariff_codes |
| UK OFSI Sanctions | CSV | 56,748 (rows) | sanctions_entities |
| HMRC Alcohol Duty Notice | PDF (9 pages) | 42 | regulatory_requirements |
| LEC About page | CSV | 7 | — (ChromaDB only) |
| LEC Past Works | CSV | 5 | — (ChromaDB only) |
| Tsingtao Annual Report | PDF (259 pages) | 1,203 | — (ChromaDB only) |
| **Total** | | **1,257 + 82,874 + 56,748** | |
