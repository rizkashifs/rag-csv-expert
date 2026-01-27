import boto3
import logging
from app.core.config import settings
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class BedrockClient:
    """
    Handles communication with AWS Bedrock via the Converse API.
    """
    def __init__(self, model_id: str = None, region: str = None):
        self.model_id = model_id or settings.BEDROCK_MODEL_ID
        self.region = region or settings.AWS_REGION
        # Note: Boto3 will use env vars AWS_ACCESS_KEY_ID etc. automatically if not provided
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.region
        )

    def generate(self, messages: List[Dict[str, str]], options: Optional[Dict[str, Any]] = None) -> str:
        """
        Sends a request to Bedrock and returns the text response.
        Converts standard message format to Bedrock converse format.
        """
        bedrock_messages = []
        system_prompts = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompts.append({"text": msg["content"]})
            else:
                bedrock_messages.append({
                    "role": msg["role"],
                    "content": [{"text": msg["content"]}]
                })

        # Options handling
        temp = 0.0
        if options and "temperature" in options:
            temp = options["temperature"]
            
        max_tokens = 2000
        if options and "max_tokens" in options:
            max_tokens = options["max_tokens"]

        try:
            logger.info(f"Bedrock Request: model={self.model_id}, region={self.region}")
            kwargs = {
                "modelId": self.model_id,
                "messages": bedrock_messages,
                "inferenceConfig": {
                    "temperature": temp,
                    "maxTokens": max_tokens
                }
            }
            if system_prompts:
                kwargs["system"] = system_prompts

            response = self.client.converse(**kwargs)
            return response["output"]["message"]["content"][0]["text"]
        except Exception as e:
            logger.error(f"Bedrock Error: {e}")
            return f"Bedrock LLM Error: {str(e)}"

    def invoke_messages(self, messages: List[Dict[str, str]], options: Optional[Dict[str, Any]] = None) -> str:
        """
        Matches the interface used by other clients.
        """
        return self.generate(messages, options)

# Singleton instance
bedrock_client = BedrockClient()