# RAG CSV Expert 🚀

An agentic Retrieval-Augmented Generation (RAG) system specialized in handling complex CSV data. This application goes beyond simple text retrieval by combining deterministic data processing (Pandas) with semantic search (FAISS) and LLM-driven reasoning (Ollama).

## 🌟 Overview

RAG CSV Expert is designed to handle "messy" and complex CSV files that traditional RAG pipelines often struggle with—such as datasets with multi-line fields, mixed data types, and rich textual content.

### The Pipeline Flow
1. **Ingestion & Profiling**: Robustly loads CSVs using automatic encoding detection and delimiter sensing. It generates a "Semantic Data Profile" (schema, distributions, and sample data) to give the LLM context about the dataset's structure.
2. **Agentic Routing**: A Router Agent analyzes the user query to decide the best "Engine" for the job:
   - **Deterministic (CSV Engine)**: For aggregations (sums, averages, counts) and exact filters.
   - **Semantic (Vector Engine)**: For meaning-based questions and open-ended text searches.
3. **Reasoning Agent**: Converts natural language into a structured JSON "Query Plan" based on the detected schema.
4. **Data Engine Execution**: 
   - **CSV Engine**: Executes Pandas operations based on the Query Plan.
   - **Vector Engine**: Performs similarity searches using FAISS and local embeddings.
5. **Answer Synthesis**: An Answer Agent combines the retrieved data with the original query to generate a factual, grounded response.

## 🏗️ Architecture Reasoning

Traditional RAG often treats CSVs like raw text, which leads to "hallucinations" in calculations and lost structure. Our architecture solves this by:
- **Separating Data from Semantics**: Using Pandas for math/filtering ensures 100% accuracy for quantitative questions.
- **Context-Aware Reasoning**: By providing the LLM with a data profile (not just raw rows), it understands what columns represent before generating logic.
- **Local-First Processing**: Built on Ollama and FAISS to ensure data privacy and fast local iteration.

## 🚀 Benefits
- **Accuracy**: Reliable aggregations and filtering via deterministic code execution.
- **Robustness**: Handles multi-line descriptions and "dirty" headers automatically.
- **Efficiency**: Only indexes what's necessary, reducing computational overhead for large files.
- **Privacy**: Runs completely locally with Ollama.

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com/) installed and running.

### 1. Clone the Repository
```powershell
git clone <repo-url>
cd rag-csv-expert
```

### 2. Set Up Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
pip install pytest  # For testing
```

### 4. Prepare the LLM
```powershell
ollama pull llama3
```

## 🏃 How to Run

### Start the API Server
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
You can then access the Interactive API documentation at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Run Tests
To verify the CSV ingestion and engine logic:
```powershell
python -m pytest tests/
```

## 📊 Try it out with Complex Data
We've included a download utility and tests for the **Netflix Movies & TV Shows** dataset, which features multi-line descriptions and nested categories—perfect for testing the limits of RAG.
