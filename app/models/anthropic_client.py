import anthropic
from app.core.config import settings
from typing import List, Dict, Any, Optional

class AnthropicClient:
    """
    Handles communication with Anthropic API.
    """
    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or settings.ANTHROPIC_MODEL
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, messages: List[Dict[str, str]], options: Optional[Dict[str, Any]] = None) -> str:
        """
        Sends a request to Anthropic and returns the text response.
        """
        try:
            # Convert messages to Anthropic format
            system_prompt = ""
            anthropic_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            # Handle options (max_tokens, temperature, etc.)
            temperature = 0.0
            if options and "temperature" in options:
                temperature = options["temperature"]
                
            max_tokens = 4096
            if options and "max_tokens" in options:
                max_tokens = options["max_tokens"]

            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=anthropic_messages
            )
            
            # Extract text from response blocks
            return "".join([block.text for block in response.content if hasattr(block, "text")])
            
        except Exception as e:
            return f"Anthropic LLM Error: {str(e)}"

# Singleton instance
anthropic_client = AnthropicClient()
