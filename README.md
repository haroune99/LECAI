# LEC Trade Intelligence Agent

A production-grade agentic system for London Export Corporation that answers multi-step trade intelligence queries across all four LEC subsidiaries.

## What It Does

Ask questions about UK-China trade in natural language. The agent plans the approach, calls the right tools, reflects on the results, and synthesises a final answer — with full visibility into its reasoning.

**Example queries:**
- "What is the current UK import duty for Tsingtao beer, and what would be the total landed cost of 10,000 cases shipped from Qingdao to Liverpool?"
- "Profile Meituan Dianping as a potential LEC Global Capital investment — sanctions check included"
- "What are the UKCA marking requirements for importing robotic arm components from China?"

## Architecture

- **Framework:** LangGraph state machine (planner → executor → reflector → answerer)
- **Model:** MiniMax-M2.7 via OpenAI-compatible API ($0.30/M input, $1.20/M output)
- **Retrieval:** Hybrid BM25 + semantic search + cross-encoder reranking (ChromaDB + sentence-transformers)
- **Tools:** Trade Regulations KB (SQLite), Document Intelligence (RAG), Market Search (Tavily), Trade Calculator, Partnership Profiler
- **UI:** Streamlit with password auth and reasoning trace panel
- **API:** FastAPI (`POST /search`, `POST /agent/run`)

## Setup

```bash
# 1. Clone and install dependencies
pip install -e .

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys:
#   MINIMAX_API_KEY=your_key
#   TAVILY_API_KEY=your_key

# 3. Seed the knowledge base
python -c "
from src.tools.trade_regulations import init_kb
init_kb()
print('KB initialised')
"

# 4. Ingest documents (requires files in data/raw/)
python -c "
from src.retrieval.ingestor import DocumentIngestor
ingestor = DocumentIngestor()
count = ingestor.ingest_directory('data/raw/')
print(f'Ingested {count} chunks')
"

# 5. Run the API server
python -m src.api.server

# 6. Run Streamlit (separate terminal)
streamlit run src/app/streamlit_app.py
```

## Document Download List

To enable the full document intelligence pipeline, download these files into `data/raw/`:

| File | Where to get it |
|---|---|
| `uk_global_tariff.csv` | gov.uk/guidance/uk-global-tariff → Download all commodity codes |
| `ofsi_sanctions_list.csv` | gov.uk → Financial Sanctions Targets List → CSV download |
| `hmrc_notice_363_alcohol_duty.pdf` | gov.uk → Excise Notice 363: Alcohol Duty → PDF |
| `tsingtao_annual_report_2025.pdf` | ir.tsingtao.com → Annual Report 2025 (English version) |
| `ifr_world_robotics_report_2024.pdf` | ifr.org → World Robotics Report 2024 executive summary |
| `lec_website/past-works.html` | Browser save from londonexportcorp.com/past-works |
| `lec_website/about.html` | Browser save from londonexportcorp.com/about |

## Running Tests

```bash
pytest tests/ -v
```

## Running Evaluation

```bash
python eval/harness.py
```

Results are written to `eval/results/eval_results.json`.

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
  raw/            # Source documents (you download these)
  processed/      # SQLite KBs
  indexes/        # ChromaDB + BM25 persisted indexes
eval/             # Eval harness + 10 queries + judge
tests/            # Unit tests
docs/             # Report, architecture, roadmap, AI usage note
```
