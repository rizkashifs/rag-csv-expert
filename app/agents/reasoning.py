from app.agents.base import BaseAgent
from app.models.ollama_client import llm_client
import json

class CSVReasoningAgent(BaseAgent):
    """
    Converts natural language questions into structured query intent.
    Outputs a valid JSON query plan.
    """
    def run(self, input_data: dict) -> dict:
        """
        Input: {"query": str, "schema_context": str}
        Output: JSON query plan (operation, columns, filters, etc.)
        """
        query = input_data.get("query")
        schema_context = input_data.get("schema_context")
        
        prompt = f"""
        Represent the user's data question as a structured JSON query plan.
        
        Schema Context:
        {schema_context}
        
        User Question: {query}
        
        Return a JSON object with:
        - "operation": "sum" | "avg" | "count" | "max" | "min" | "filter" | "none"
        - "columns": [list of relevant column names]
        - "filters": {{ "column_name": "value" }}
        - "group_by": [list of columns]
        
        Rules:
        - Output ONLY valid JSON.
        - No markdown blocks.
        - No natural language explanations.
        - Use only columns present in the schema.
        """
        
        messages = [{"role": "user", "content": prompt}]
        # Using temperature 0 for reasoning consistency
        response = llm_client.generate(messages, options={"temperature": 0.0})
        
        try:
            return json.loads(response.strip())
        except Exception:
            # Fallback for parsing issues
            return {
                "operation": "none",
                "columns": [],
                "filters": {},
                "group_by": []
            }
