# RAG CSV Expert

A disciplined, agentic RAG application specialized in complex CSV handling using local LLMs (Ollama + Llama3).

## Features
- **Deterministic Data Processing**: Uses Pandas for calculations (sums, averages, filters) to eliminate LLM math hallucinations.
- **Agentic Routing**: Automatically routes questions to either a **CSV Engine** (for data/numbers) or a **Vector Engine** (for semantic/meaning based questions).
- **Industrial Strength Ingestion**: Robust handling of CSV encodings, delimiters, and automated semantic data profiling.
- **Local & Private**: Runs entirely on your machine using Ollama.

## Prerequisites
1. **Ollama**: Download and install from [ollama.com](https://ollama.com).
2. **Python 3.10+**: Ensure you have Python installed.

## Setup Instructions

### 1. Configure Ollama
Ensure Ollama is running, then pull the required model:
```powershell
# If ollama is in your PATH:
ollama pull llama3

# If you use the default Windows installation path:
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull llama3
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
- **Upload CSV**: `POST /api/upload` - Upload your CSV file for indexing and profiling.
- **Query**: `POST /api/query` - Ask questions about your data (e.g., "What is the average price?", "Explain the user feedback Trends").

## Architecture
This project follows a Multi-Agent architecture:
1. **Router Agent**: Classifies the question type.
2. **Reasoning Agent**: Generates a valid JSON query plan.
3. **Retriever Agent**: Executes the plan using **deterministic engines** (Pandas) or **semantic engines** (FAISS).
4. **Answer Agent**: Synthesizes the final result into a human-readable explanation with 0 temperature.
