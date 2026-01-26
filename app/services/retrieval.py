from app.services.orchestration import orchestrator
import os

def retrieve_csv(file_path: str, query: str):
    """
    Orchestrates the agentic RAG flow for CSV data.
    """
    index_name = os.path.basename(file_path)
    return orchestrator.run_pipeline(query, file_path, index_name)
