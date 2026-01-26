import logging
from app.agents.base import BaseAgent
from app.models.anthropic_client import anthropic_client as llm_client

logger = logging.getLogger(__name__)

class AnswerAgent(BaseAgent):
    """
    Generates final natural-language response using retrieved data and query plan.
    Ensures deterministic output.
    """
    def run(self, input_data: dict) -> str:
        """
        Input: {"query": str, "retrieved_data": any, "intent": dict}
        Output: Human-readable answer.
        """
        query = input_data.get("query")
        retrieved_data = input_data.get("retrieved_data")
        intent = input_data.get("intent")
        
        prompt = f"""
        Convert the following deterministic data results into a human-readable explanation.
        
        User Question: {query}
        Query Plan: {intent}
        Retrieved Data: {retrieved_data}
        
        Rules:
        - Never invent numbers.
        - Add context and caveats if the data is limited.
        - Be concise and factual.
        """
        
        messages = [{"role": "user", "content": prompt}]
        # Temp 0 for deterministic synthesis
        logger.info("Sending synthesis request to Anthropic...")
        response = llm_client.generate(messages, options={"temperature": 0.0})
        logger.info("Received answer from Anthropic.")
        return response
