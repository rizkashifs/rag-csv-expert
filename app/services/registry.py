from typing import Dict, List, Any, Optional

from app.utils.logger import logger

class FileRegistry:
    """
    In-memory registry of processed files, including their summaries and schemas.
    """
    def __init__(self):
        self.files: Dict[str, Dict[str, Any]] = {}

    def add_file(
        self,
        file_path: str,
        summary: str,
        row_count: int,
        data_type: str,
        schema_context: str,
        semantic_summary: str,
        text_heavy: bool
    ):
        file_name = file_path.split("/")[-1]
        self.files[file_path] = {
            "file_name": file_name,
            "summary": summary,
            "row_count": row_count,
            "type": data_type,
            "schema_context": schema_context,
            "semantic_summary": semantic_summary,
            "text_heavy": text_heavy
        }
        logger.info(f"File added to registry: {file_path}")

    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        return self.files.get(file_path)

    def get_all_summaries(self) -> str:
        """Returns a formatted string of all file paths and their summaries."""
        summary_list = []
        for path, info in self.files.items():
            summary_list.append(f"File Path: {path}\nSummary: {info['summary']}\n---")
        return "\n".join(summary_list)

    def list_files(self) -> List[str]:
        return list(self.files.keys())

# Singleton
file_registry = FileRegistry()
