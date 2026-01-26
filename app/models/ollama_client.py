import ollama
from app.core.config import settings
from typing import List, Dict, Any, Optional

class LLMClient:
    """
    Handles communication with LLM backends.
    """
    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or settings.OLLAMA_MODEL
        self.base_url = base_url or settings.OLLAMA_BASE_URL

    def generate(self, messages: List[Dict[str, str]], options: Optional[Dict[str, Any]] = None) -> str:
        """
        Sends a request to the LLM and returns the text response.
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                stream=False,
                options=options
            )
            return response['message']['content']
        except Exception as e:
            # In a production app, we'd log this properly
            return f"LLM Error: {str(e)}"

# Singleton instance
llm_client = LLMClient()
