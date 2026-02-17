from app.agents.base import BaseAgent
from app.models.llm_client import llm_client

from app.utils.logger import logger

class SummaryAgent(BaseAgent):
    """
    Generates a concise summary of the dataset based on its schema and sample data.
    """
    def run(self, input_data: dict) -> str:
        """
        Input: {"schema_context": str, "sample_data": str}
        Output: A high-level summary of what the file contains.
        """
        schema_context = input_data.get("schema_context")
        sample_data = input_data.get("sample_data")
        
        prompt = f"""
        Provide a concise, professional summary of the following dataset.
        Focus on the semantic meaning you can infer from the first 5 rows and schema.

        Schema & Profiling:
        {schema_context}

        Sample Data (First 5 rows):
        {sample_data}

        Summary:
        """
        
        messages = [{"role": "user", "content": prompt}]
        logger.info("Generating dataset summary...")
        response = llm_client.generate(messages, options={"temperature": 0.0})
        return response
