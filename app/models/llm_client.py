from app.core.config import settings
from app.models.ollama_client import llm_client as ollama
from app.models.anthropic_client import anthropic_client as anthropic
from app.models.bedrock_client import bedrock_client as bedrock

from app.utils.logger import logger

class UnifiedLLMClient:
    """
    Delegates to the configured LLM provider.
    """
    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        logger.info(f"Initialized UnifiedLLMClient with provider: {self.provider}")

    def generate(self, messages, options=None):
        if self.provider == "anthropic":
            return anthropic.generate(messages, options)
        elif self.provider == "bedrock":
            return bedrock.generate(messages, options)
        else:
            return ollama.generate(messages, options)
            
    def invoke_messages(self, messages, options=None):
        return self.generate(messages, options)

# Singleton
llm_client = UnifiedLLMClient()
