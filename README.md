# RAG CSV Expert

A disciplined, agentic RAG application specialized in complex CSV handling using local LLMs (Ollama + Phi) or Cloud LLMs (Anthropic Claude).

## Features
- **Deterministic Data Processing**: Uses Pandas for calculations (sums, averages, filters) to eliminate LLM math hallucinations.
- **Agentic Routing**: Automatically routes questions to either a **CSV Engine** (for data/numbers) or a **Vector Engine** (for semantic/meaning based questions).
- **Industrial Strength Ingestion**: Robust handling of CSV encodings, delimiters, and automated semantic data profiling.
- **Local & Private**: Option to run entirely on your machine using Ollama.
- **Cloud Support**: Built-in client for Anthropic (Claude) for heavy-duty reasoning tasks.

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

### 2. Configure Anthropic (Optional)
If you prefer using Claude, add your API key to a `.env` file:
```env
ANTHROPIC_API_KEY=your_sk_key_here
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

*Note: Agents can be easily switched between `ollama_client` and `anthropic_client` in the code.*

For more details, read the DETAILED_README.md file.
