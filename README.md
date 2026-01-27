# RAG CSV Expert

A disciplined, agentic RAG application specialized in complex CSV and Excel handling using local LLMs (Ollama) or Cloud LLMs (Anthropic, AWS Bedrock).

## Features
- **Multi-Format Support**: Robust handling of CSV and Excel (`.xlsx`, `.xls`) files, including multi-sheet workbooks.
- **Deterministic Data Processing**: Uses Pandas for calculations (sums, averages, filters) across all sheets to eliminate LLM math hallucinations.
- **Agentic Routing**: Automatically routes questions to either a **CSV Engine** (for data/numbers) or a **Vector Engine** (for semantic/meaning based questions).
- **Industrial Strength Ingestion**: Robust handling of CSV encodings, delimiters, and automated semantic data profiling.
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
- **Upload File**: `POST /api/upload` - Upload your CSV or Excel (`.xlsx`, `.xls`) file for indexing and profiling.
- **Query**: `POST /api/query` - Ask questions about your data (e.g., "What is the average price?", "Explain the user feedback Trends").
Example payload - 
{
  "query": "summarise the file",
  "file_path": "data/netflix_titles.csv"
}

## Architecture
This project follows a Multi-Agent architecture:
1. **Router Agent**: Classifies the question type.
2. **Reasoning Agent**: Generates a valid JSON query plan.
3. **Retriever Agent**: Executes the plan using **deterministic engines** (Pandas) or **semantic engines** (FAISS).
4. **Answer Agent**: Synthesizes the final result into a human-readable explanation with 0 temperature.

*Note: Agents can be easily switched between `ollama_client` and `anthropic_client` in the code.*


**For more details, read the DETAILED_README.md file.**




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
