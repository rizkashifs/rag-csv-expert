import ollama
from app.core.config import settings
from typing import List, Dict, Any, Optional

from app.utils.logger import logger

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
            logger.info(f"Ollama Request: model={self.model}, messages_count={len(messages)}")
            response = ollama.chat(
                model=self.model,
                messages=messages,
                stream=False,
                options=options
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Ollama Error: {e}")
            raise ConnectionError(f"Could not connect to Ollama: {e}")

    def invoke_messages(self, messages: List[Dict[str, str]], options: Optional[Dict[str, Any]] = None) -> str:
        """
        Alias for generate to match user request.
        """
        return self.generate(messages, options)

# Singleton instance
llm_client = LLMClient()
