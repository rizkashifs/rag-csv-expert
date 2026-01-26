# RAG CSV Expert - Deep Dive 🧠📊

## 🎯 Core Problem It Solves

Traditional RAG pipelines treat CSVs like plain text, causing critical issues:

- **Hallucinations in math**: LLM guesses sums instead of computing them
- **Lost structure**: Relationships between columns disappear
- **Poor routing**: Semantic search used for what should be simple aggregations

## 🏗️ Architecture Overview

The application uses a 5-stage agentic pipeline:

```
User Query
    ↓
1. INGESTION & PROFILING (IngestionService)
    ↓
2. ROUTING (RouterAgent - rule-based)
    ↓
3. REASONING (CSVReasoningAgent - LLM-based intent planning)
    ↓
4. RETRIEVAL (CSVRetrieverAgent - executes with right engine)
    ↓
5. ANSWER SYNTHESIS (AnswerAgent - grounded response)
```

## Stage-by-Stage Breakdown

### 1️⃣ Ingestion & Profiling (IngestionService)

**What it does:**

- Automatically detects file encoding (handles international characters)
- Intelligently detects CSV delimiters using pandas
- Cleans messy headers (removes !, ?, special chars)
- Drops empty rows, fills NaNs with "N/A"
- Generates a semantic profile of the dataset

**Key code:**
```python
def read_csv(self, file_path: str) -> Dict[str, Any]:
    # 1. Encoding detection (chardet)
    # 2. Robust parsing (pandas with sep=None for auto-detection)
    # 3. Cleaning (regex on headers, dropna, fillna)
    # 4. Profiling (schema, distributions, sample values)
```

**Why this matters:** The semantic profile gives the LLM context without raw data:

- Column names and types
- Min/max for numeric columns
- Sample values for categorical columns
- Missing value counts

This allows the LLM to reason about structure without seeing all 100k rows.

### 2️⃣ Routing (RouterAgent)

**What it does:** Uses rule-based keyword matching (no LLM) to classify queries into three types:

| Query Type   | Keywords                          | Engine        |
|--------------|-----------------------------------|---------------|
| Aggregation  | sum, avg, count, max, min, total | CSV Engine    |
| Lookup       | id, find, search, where + digits | CSV Engine    |
| Semantic     | explain, what is, about, why     | Vector Engine |

**Why no LLM here?**

- Fast (no API call)
- Deterministic (no variance)
- Perfect for structured decisions

**Example:**
```python
query = "What is the total revenue for 2019?"
# Matches "total" → returns {"question_type": "aggregation", "engine": "csv_engine"}
```

### 3️⃣ Reasoning (CSVReasoningAgent)

**What it does:** Converts natural language → structured JSON query plan using LLM.

**Input:**

```json
{
    "query": "Show me all comedies released in 2019",
    "schema_context": "Dataset: 8800 rows, 12 cols\n- title (object): Samples: [...]
                      - listed_in (object): Samples: [Comedy, Drama, ...]\n
                      - release_year (int64): Range [1920 to 2021]"
}
```

**LLM Prompt:**

```
Represent the user's data question as a structured JSON query plan.

Schema Context: [CSV profile]
User Question: "Show me all comedies released in 2019"

Return JSON with:
- "operation": "filter" | "sum" | "avg" | "count" | etc.
- "columns": [relevant columns]
- "filters": {"column_name": "value"}
```

**Output:**

```json
{
    "operation": "filter",
    "filters": {
        "listed_in": "Comedies",
        "release_year": "2019"
    },
    "columns": [],
    "group_by": []
}
```

**Why this is genius:**

- LLM maps user intent to database semantics without executing
- Uses temperature=0 for consistency
- Falls back gracefully if JSON parsing fails

### 4️⃣ Retrieval (CSVRetrieverAgent)

**What it does:** Executes the query plan using the right engine based on routing decision.

**CSV Engine (CSVEngine)** Deterministic Pandas operations—guarantees accuracy:

```python
def execute(self, df: pd.DataFrame, intent: Dict) -> Dict:
    # Apply filters
    for col, val in intent.get("filters", {}).items():
        filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(val, ...)]
    
    # Execute operation
    if operation == "sum":
        return filtered_df[columns].sum().to_dict()
    elif operation == "count":
        return len(filtered_df)
    elif operation == "filter":
        return filtered_df.head(20).to_dict('records')
```

**Why deterministic?**

- No LLM hallucination
- Exact numbers guaranteed
- Fast (Pandas is optimized)

**Vector Engine (VectorEngine)** Semantic search using FAISS + Local embeddings (Ollama / Phi):

```python
def search(self, query: str, index_name: str, k: int = 5) -> List[str]:
    vector_store = FAISS.load_local(save_path, self.embeddings)
    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]
```

Used for:

- "What is the theme of this movie?"
- "Why did they make this series?"
- Text understanding questions

### 5️⃣ Answer Synthesis (AnswerAgent)

**What it does:** Converts structured data → natural language response.

**Input:**

```json
{
    "query": "Show me comedies from 2019",
    "retrieved_data": [{"title": "...", "released": 2019}, ...],
    "intent": {"operation": "filter", ...}
}
```

**LLM Prompt:**

```
Convert the following deterministic data results into human-readable explanation.

User Question: Show me comedies from 2019
Query Plan: {operation: filter, filters: {listed_in: Comedies, release_year: 2019}}
Retrieved Data: [actual result rows]

Rules:
- Never invent numbers
- Add context if data is limited
- Be concise and factual
```

**Output:**

```
"Found 35 comedies released in 2019. Top titles include: [...]"
```

## 🔄 Full Pipeline Example

**Query:** "What's the average rating of movies from 2020?"

1. **INGESTION:** Loads netflix_titles.csv, generates profile with rating stats
2. **ROUTING:** Matches "average" → CSV Engine (aggregation)
3. **REASONING:** Converts to JSON

   ```json
   {
       "operation": "avg",
       "columns": ["rating"],
       "filters": {"release_year": "2020"},
       "group_by": []
   }
   ```

4. **RETRIEVAL (CSV Engine):**

   - Filters: df[df["release_year"] == 2020]
   - Averages: df["rating"].mean() = 6.8
   - Returns: {"data": 6.8, "row_count": 342}

5. **ANSWER:** LLM synthesizes

   `"The average rating of movies from 2020 is 6.8 out of 10,
    based on 342 movies in the database."`

## Integration Points

**How It Fits Your Production RAG App**

This module is plug-and-play if you:

- Have CSV data → Use IngestionService directly
- Need routing → Use RouterAgent pattern
- Want semantic search → Integrate VectorEngine
- Have aggregation queries → Use CSVEngine

**Example integration:**

```python
from app.services.orchestration import orchestrator

result = orchestrator.run_pipeline(
    query="Find all Action movies from 2021",
    file_path="data/movies.csv",
    index_name="movies"
)

print(result["answer"])  # Natural language response
```

## Key Design Patterns to Learn

| Pattern                     | Location            | Why It Matters                                                  |
|-----------------------------|---------------------|----------------------------------------------------------------|
| Agent Pattern               | agents/             | Separation of concerns (route, reason, retrieve, answer)      |
| Singleton Factories         | Engine/Service modules | Single instance per component, easier state management        |
| Semantic Profiling          | IngestionService    | Context without full data exposure                            |
| Deterministic + Semantic Hybrid | Retrieval stage    | Best of both worlds                                            |
| Temperature=0 for Reasoning | All LLM calls       | Consistent, reproducible outputs                              |
| Fallback Strategies         | CSVReasoningAgent line 38+ | Graceful degradation                                          |

## 📊 What Makes This Production-Ready

- ✅ **Tested** — see tests for CSV cleaning and Netflix integration tests
- ✅ **Robust** — handles encoding, dirty headers, multi-line fields
- ✅ **Fast** — deterministic ops beat semantic search when possible
- ✅ **Private** — runs fully local with Ollama
- ✅ **Modular** — agents are loosely coupled, easy to replace/extend

## 💡 For Your CSV Module

Focus on these three components:

- **IngestionService** — Build this first, make it bulletproof
- **RouterAgent** — Simple rules save LLM calls
- **CSVEngine** — All your deterministic logic here

Start with deterministic CSV operations, add vector search layer later. That's the proven pattern here.






