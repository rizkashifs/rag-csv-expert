import json
import logging
from app.agents.base import BaseAgent
from app.models.llm_client import llm_client

logger = logging.getLogger(__name__)

class FileSelectorAgent(BaseAgent):
    """
    Analyzes user query and selects the most relevant file(s) from the registry.
    """
    def run(self, input_data: dict) -> list:
        """
        Input: {"query": str, "file_summaries": str}
        Output: list of file_paths
        """
        query = input_data.get("query")
        file_summaries = input_data.get("file_summaries")
        
        if not file_summaries:
            return []

        prompt = f"""
        You are an expert data router. Given a user query and a set of available data files (with their summaries), 
        identify which file(s) are necessary to answer the query.

        Available Files:
        {file_summaries}

        User Query: {query}

        Return ONLY a JSON list of the relevant File Paths. 
        Example Output: ["data/sales.csv", "data/inventory.xlsx"]
        If none are relevant, return [].
        """
        
        messages = [{"role": "user", "content": prompt}]
        logger.info(f"Selecting relevant files for query: {query[:50]}...")
        response = llm_client.generate(messages, options={"temperature": 0.0})
        
        try:
            # Basic cleanup of LLM response
            clean_response = response.strip()
            if "```json" in clean_response:
                clean_response = clean_response.split("```json")[-1].split("```")[0].strip()
            elif "```" in clean_response:
                clean_response = clean_response.split("```")[-1].split("```")[0].strip()
            
            return json.loads(clean_response)
        except Exception as e:
            logger.error(f"Error parsing FileSelector response: {e}. Raw: {response}")
            return []
