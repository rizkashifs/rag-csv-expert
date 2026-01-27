# RAG CSV Expert

A disciplined, agentic RAG application specialized in complex CSV and Excel handling using local LLMs (Ollama) or Cloud LLMs (Anthropic, AWS Bedrock).

## Features
- **Multi-Format Support**: Robust handling of CSV and Excel (`.xlsx`, `.xls`) files, including multi-sheet workbooks.
- **Multi-File Intelligence**: Upload multiple files and let the system automatically route queries to the correct dataset using LLM-based file selection.
- **Deterministic Data Processing**: Uses Pandas for calculations (sums, averages, filters, correlations) across all sheets to eliminate LLM math hallucinations.
- **Agentic Routing**: Automatically routes questions to either a **CSV Engine** (for data/numbers) or a **Vector Engine** (for semantic/meaning based questions).
- **Industrial Strength Ingestion**: Robust handling of CSV encodings, delimiters, and automated semantic data profiling with LLM-generated summaries.
- **Hybrid Embeddings**: Choose between **HuggingFace** (extremely fast, runs in Python) or **Ollama** for vector indexing.
- **Local & Private**: Option to run entirely on your machine with Ollama.
- **Enterprise Cloud Support**: Native integration with **Anthropic Claude** and **AWS Bedrock** (Converse API) for production workloads.

## Prerequisites
1. **Ollama**: Download and install from [ollama.com](https://ollama.com).
2. **Python 3.10+**: Ensure you have Python installed.

## Setup Instructions

### 1. Configure Ollama
Ensure Ollama is running, then pull the required model:
```powershell
# If ollama is in your PATH:
ollama pull phi

# If you use the default Windows installation path:
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull phi
```

### 2. Configure Settings (Optional)
You can switch embedding providers in `app/core/config.py`:
```python
EMBEDDING_PROVIDER = "huggingface"  # Options: "ollama", "huggingface"
```

If you prefer using Claude for reasoning, add your Anthropic API key to a `.env` file:
```env
ANTHROPIC_API_KEY=your_sk_key_here
```

To use **AWS Bedrock**, configure your credentials and set the provider:
```env
LLM_PROVIDER=bedrock
BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

### 2. Install Dependencies
Clone the repository and install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

## API Usage
- **Documentation**: Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI.
- **Upload File**: `POST /api/upload` - Upload your CSV or Excel (`.xlsx`, `.xls`) file for indexing, profiling, and automatic summarization.
- **Query**: `POST /api/query` - Ask questions about your data.

### Multi-File Routing
The system supports intelligent routing across multiple uploaded files:
- **With file_path**: Target a specific file explicitly
  ```json
  {
    "query": "What is the average price?",
    "file_path": "data/sales_2023.csv"
  }
  ```
- **Without file_path**: Let the system automatically select the relevant file
  ```json
  {
    "query": "How many employees were hired last year?"
  }
  ```
  The **FileSelectorAgent** analyzes all uploaded file summaries and routes your query to the most relevant dataset.

## Architecture
This project follows a Multi-Agent architecture with intelligent file routing:
1. **File Selector Agent** (Optional): Determines which file to query based on LLM analysis of file summaries.
2. **Router Agent**: Classifies the question type (aggregation, lookup, semantic).
3. **Reasoning Agent**: Generates a valid JSON query plan with dataset context.
4. **Retriever Agent**: Executes the plan using **deterministic engines** (Pandas) or **semantic engines** (FAISS).
5. **Answer Agent**: Synthesizes the final result into a human-readable explanation with 0 temperature.
6. **Summary Agent**: Generates contextual summaries of datasets for better query understanding.

*Note: All agents use a unified LLM client that supports Ollama, Anthropic, and AWS Bedrock.*

For more details, read the DETAILED_README.md file.



Sample Query Payload
{
  "query": "What is the average release_year?",
  "file_path": "data/netflix_titles.csv"
}


Response:
{
  "answer": "The average release year of the data retrieved is 2013.36. This value is based on a sample size of 6,234 data points. Since the data provided is limited to a specific set of records, the average release year may not be representative of the entire population or dataset. Additional context or caveats should be considered when interpreting this result.",
  "context": [
    "data: {'release_year': 2013.3593198588387}",
    "row_count: 6234"
  ]
}
