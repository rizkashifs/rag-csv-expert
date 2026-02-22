# RAG CSV Expert

Agentic RAG API for CSV/Excel analytics with deterministic Pandas execution and text-table retrieval.

## What It Does
- Ingests `.csv`, `.xlsx`, and `.xls` files.
- Builds a dataset profile and LLM-generated summary on upload.
- Routes each question to one of these paths:
  - `SQL_ENGINE` for deterministic analytics (`sum`, `avg`, `count`, `min`, `max`, filtering, grouping, having, sorting).
  - `TEXT_TABLE_RAG` for text-heavy row retrieval using Pandas/regex keyword matching.
  - `PROFILE_ONLY` for schema/summary requests.
  - `REFUSE` when the query is ambiguous and needs clarification.
- Supports single-file queries or automatic file selection across multiple uploaded files.
- Supports LLM providers: `ollama`, `anthropic`, `bedrock`.
- Supports embedding providers for FAISS indexing: `huggingface` or `ollama`.

## Tech Stack
- FastAPI + Uvicorn
- Pandas
- FAISS (LangChain wrapper)
- Ollama / Anthropic / AWS Bedrock

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Create `.env`
Minimal local setup:
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi

EMBEDDING_PROVIDER=huggingface
HUGGINGFACE_MODEL=all-MiniLM-L6-v2

FAISS_INDEX_PATH=data/faiss_index
```

If you use Anthropic:
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_key
ANTHROPIC_MODEL=claude-3-haiku-20240307
```

If you use Bedrock:
```env
LLM_PROVIDER=bedrock
BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

If using Ollama embeddings instead of HuggingFace:
```env
EMBEDDING_PROVIDER=ollama
```

### 3. Ensure Ollama model exists (if using Ollama)
```powershell
ollama pull phi
```

### 4. Run the API
```bash
uvicorn app.main:app --reload
```

API root: `http://127.0.0.1:8000`
Swagger docs: `http://127.0.0.1:8000/docs`

## API

### `POST /api/upload`
Uploads and processes a file, registers it in memory, and creates a FAISS index.

Allowed extensions: `.csv`, `.xlsx`, `.xls`

Example:
```bash
curl -X POST "http://127.0.0.1:8000/api/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/netflix_titles.csv"
```

Typical response fields:
- `message`
- `file_path`
- `type`
- `row_count`
- `summary`
- `indexed_chunks`
- `text_heavy`

### `POST /api/query`
Runs the full orchestration pipeline.

Request body:
```json
{
  "query": "What is the average release_year?",
  "file_path": "data/netflix_titles.csv",
  "chat_id": "session-1"
}
```

Notes:
- `file_path` is optional. If omitted, the File Selector agent picks from uploaded files.
- `chat_id` is optional. If provided, recent turns are used as routing context.

Response shape:
```json
{
  "answer": "...",
  "context": ["..."],
  "metadata": {
    "execution_time": 0.42,
    "data_type": "csv",
    "row_count": 6234,
    "route": "SQL_ENGINE",
    "use_routing_agent": true,
    "route_schema": {}
  }
}
```

## Query Types Supported
- Numeric analytics: sums, averages, min/max, counts.
- Filtered retrieval: equality, inequality, `contains`, `in`, `between`, numeric/date comparisons.
- Grouped analytics: `group_by` + optional `having`.
- Sorted and limited outputs.
- Text-table search over long-form columns with keyword and ID scoping.
- Dataset profiling and semantic summaries.

## Architecture (Current Flow)
1. Upload flow:
   - `IngestionService` loads/cleans/profiles data.
   - `SummaryAgent` generates dataset summary.
   - `FileRegistry` stores metadata in memory.
   - `VectorEngine` creates FAISS index.
2. Query flow:
   - Optional file auto-selection (`FileSelectorAgent`).
   - `RouterAgent` selects route and structured schema.
   - `CSVRetrieverAgent` executes via:
     - `SQLEngine` for deterministic table ops.
     - `TextEngine` for text-table semantic retrieval.
   - `AnswerAgent` synthesizes final answer.

## Important Operational Notes
- File registry and chat history are in-memory only; restart clears them.
- Uploaded files are saved under `data/<filename>`.
- Uploading a file with the same name overwrites the previous file on disk.
- FAISS indexes are stored under `data/faiss_index/<filename>/`.

## Development
Run tests:
```bash
pytest -q
```

## Additional Docs
- `DETAILED_README.md`
- `API_EXAMPLES.txt`
- `app/engines/README.md`
