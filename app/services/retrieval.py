from typing import Optional

from app.services.orchestration import orchestrator


def retrieve_csv(query: str, file_path: Optional[str] = None, chat_id: Optional[str] = None):
    """
    Orchestrates the agentic RAG flow for CSV/Excel data, supporting multi-file routing.
    """
    return orchestrator.run_multi_file_pipeline(query, file_path, chat_id=chat_id)
