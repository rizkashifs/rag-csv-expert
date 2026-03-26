# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start the API server
uvicorn app.main:app --reload

# Run all tests
pytest -q

# Run a specific test file
pytest tests/test_csv_engine_real_world.py -v

# Install dependencies
pip install -r requirements.txt
```

API docs are available at `http://127.0.0.1:8000/docs` when the server is running.

## Configuration

Copy relevant env vars into a `.env` file. Key settings in [app/core/config.py](app/core/config.py):

| Variable | Options | Default |
|---|---|---|
| `LLM_PROVIDER` | `ollama`, `anthropic`, `bedrock` | `ollama` |
| `EMBEDDING_PROVIDER` | `huggingface`, `ollama` | `huggingface` |
| `ANTHROPIC_API_KEY` | — | — |
| `OLLAMA_MODEL` | any Ollama model | — |
| `BEDROCK_MODEL_ID` | AWS model ID | — |

## Architecture

The system is a 5-stage agentic RAG pipeline for CSV/Excel analytics:

```
Upload → Ingest → [Route → Reason → Retrieve] → Answer
                   (per /query request)
```

**1. Ingestion** ([app/services/ingestion.py](app/services/ingestion.py))
- Loads CSV/Excel with auto-encoding/delimiter detection
- Generates a semantic profile (column types, min/max, samples) stored in-memory
- Handles multi-sheet Excel workbooks

**2. Routing** ([app/agents/router.py](app/agents/router.py))
- Rule-based keyword matching only — no LLM call
- Routes to: `SQL_ENGINE`, `TEXT_TABLE_RAG`, `PROFILE_ONLY`, or `REFUSE`
- Enriches follow-up queries using chat history

**3. Reasoning** ([app/agents/reasoning.py](app/agents/reasoning.py))
- LLM converts natural language → structured JSON query plan
- Uses `temperature=0` for consistency
- Outputs: operation, columns, filters, group_by, limit

**4. Retrieval** — two engines:
- **CSVEngine** ([app/engines/csv_engine.py](app/engines/csv_engine.py)): Deterministic Pandas operations (filter, sum, avg, min, max, count, group_by, correlation, profile). Supports federated aggregation across Excel sheets.
- **TextEngine** ([app/engines/text_engine.py](app/engines/text_engine.py)): Regex/keyword semantic text matching without embeddings.
- **VectorEngine** ([app/engines/vector_engine.py](app/engines/vector_engine.py)): FAISS-backed similarity search via sentence-transformers.

**5. Answer Synthesis** ([app/agents/answer.py](app/agents/answer.py))
- Grounds LLM answer in actual retrieval results to prevent hallucination

### Engine Output Contract

Both engines return a standardized dict with reserved keys (see [app/engines/README.md](app/engines/README.md)):
- `relevant_rows[]` — result rows
- `_summary` — human-readable description
- `should_ask_user: bool` — refusal/clarification flag
- `sheet` — sheet name (Excel multi-sheet only)

### State & Storage

- File registry and chat history are **in-memory only** — cleared on restart
- Uploaded files saved to `data/<filename>`; same filename overwrites
- FAISS indexes persisted to `data/faiss_index/<filename>/`

### LLM Clients

[app/models/](app/models/) has per-provider clients (`anthropic_client.py`, `ollama_client.py`, `bedrock_client.py`) all extending `LLMClient` base. The active client is selected at startup from `LLM_PROVIDER`.

### Orchestration

[app/services/orchestration.py](app/services/orchestration.py) wires the full pipeline together per query. [app/agents/file_selector.py](app/agents/file_selector.py) handles auto-selection when `file_path` is omitted from the query request.
