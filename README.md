# LEC Trade Intelligence Agent

A production-grade agentic system for London Export Corporation that answers multi-step trade intelligence queries across all four LEC subsidiaries.

## What It Does

Ask questions about UK-China trade in natural language. The agent plans the approach, calls the right tools, reflects on the results, and synthesises a final answer — with full visibility into its reasoning.

**Example queries:**
- "What is the current UK import duty for Tsingtao beer, and what would be the total landed cost of 10,000 cases shipped from Qingdao to Liverpool?"
- "Profile Meituan Dianping as a potential LEC Global Capital investment — sanctions check included"
- "Summarise LEC's history with China trade since the 1953 founding deal"

## Architecture

- **Framework:** LangGraph state machine (planner → executor → reflector → answerer)
- **Model:** MiniMax-M2.7 via OpenAI-compatible API ($0.30/M input, $1.20/M output)
- **Retrieval:** Hybrid BM25 + semantic search + cross-encoder reranking (ChromaDB + sentence-transformers)
- **Tools:** Trade Regulations KB (SQLite), Document Intelligence (RAG), Market Search (Tavily), Trade Calculator, Partnership Profiler
- **UI:** Streamlit with password auth and reasoning trace panel
- **API:** FastAPI (`POST /search`, `POST /agent/run`)

## Setup

**Expected project structure** — download the `data/` folder and place it at the project root so it sits alongside `src/`, `eval/`, etc.:

```
LEC/                          ← project root
├── data/                    ← download + extract this from provided archive
│   ├── raw/                 # Cleaned source documents
│   ├── processed/           # SQLite KBs (tariff_codes, sanctions_entities)
│   └── indexes/             # ChromaDB + BM25 indexes
├── src/                    # From git clone
├── eval/                   # From git clone
├── scripts/                # From git clone
├── .env                    # Create from .env.example
└── pyproject.toml
```

### Step 1 — Clone and install dependencies

```bash
git clone <repo-url> LEC
cd LEC
pip install -e .
```

### Step 2 — Add your data folder

1. Download the provided `data/` archive
2. Extract it so `data/` sits at `LEC/` root — **not** inside `src/` or anywhere else

The three subdirectories must be:
- `data/raw/` — cleaned source documents (CSVs, PDFs, HTML)
- `data/processed/` — SQLite KBs (already populated with 82,874 tariff codes + 56,748 sanctions entities)
- `data/indexes/` — ChromaDB + BM25 indexes (already built)

### Step 3 — Configure environment

```bash
cp .env.example .env
# Edit .env:
#   MINIMAX_API_KEY=your_key
#   TAVILY_API_KEY=your_key
```

### Step 4 — Run

```bash
# API server
PYTHONPATH=. python -m src.api.server

# Streamlit (separate terminal)
PYTHONPATH=. streamlit run src/app/streamlit_app.py
```

Eval harness:
```bash
PYTHONPATH=. python eval/harness.py
```
Results → `eval/results/eval_results.json`.

### If you replace or add documents

```bash
# Re-clean (if starting from raw gov.uk downloads)
python scripts/clean_corpus.py

# Re-ingest everything
python -c "
from src.retrieval.ingestor import DocumentIngestor
ingestor = DocumentIngestor()
count = ingestor.ingest_directory('data/raw/')
print(f'Ingested {count} chunks')
"
```

## Running Tests

```bash
pytest tests/ -v
```

All 12 tests pass.

## Deploying to Streamlit Community Cloud

1. Push to GitHub (public repo)
2. Go to [share.streamlit.io](https://share.streamlit.io) → "New app"
3. Connect your repo → set entry point: `src/app/streamlit_app.py`
4. In "Secrets", add:
   ```
   MINIMAX_API_KEY = your_key
   TAVILY_API_KEY = your_key
   APP_PASSWORD = lec2026
   ```
5. Deploy — app gets a public URL

## Project Structure

```
src/
  tools/          # 5 domain-specific tools
  agent/          # LangGraph state machine + nodes + prompts
  retrieval/      # RAG pipeline (chunking, indexing, search)
  api/            # FastAPI server
  app/            # Streamlit UI
data/
  raw/            # Source documents (downloaded + cleaned here)
  raw/backup/     # Original backups from clean_corpus.py
  processed/      # SQLite KBs (tariff_codes, sanctions_entities)
  indexes/        # ChromaDB + BM25 persisted indexes
eval/             # Eval harness + 10 queries + judge
tests/            # Unit tests (12 passing)
docs/             # Report, architecture, roadmap, AI usage note
scripts/          # clean_corpus.py (pre-ingestion), trace_query.py
```
