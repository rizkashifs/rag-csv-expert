from app.services.orchestration import orchestrator
import os

from typing import Optional

def retrieve_csv(query: str, file_path: Optional[str] = None):
    """
    Orchestrates the agentic RAG flow for CSV/Excel data, supporting multi-file routing.
    """
    return orchestrator.run_multi_file_pipeline(query, file_path)
