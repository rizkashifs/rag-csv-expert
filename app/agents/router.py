from app.agents.base import BaseAgent
import re

class RouterAgent(BaseAgent):
    """
    Classifies user questions and decides which data engine should be used.
    Uses rule-based logic (No LLM).
    """
    def __init__(self):
        self.aggregation_keywords = [
            "sum", "average", "avg", "total", "count", "max", "min", "mean", "median",
            "correlation", "corr", "relationship", "relate", "linked"
        ]
        self.lookup_keywords = [
            "id", "find", "search", "where", "details for"
        ]
        self.semantic_keywords = [
            "explain", "what is", "about", "context", "history", "why"
        ]

    def run(self, input_data: dict) -> dict:
        """
        Input: {"query": str}
        Output: {"question_type": str, "engine": str}
        """
        query = input_data.get("query", "").lower()
        
        # 1. Check for Aggregations
        for kw in self.aggregation_keywords:
            if re.search(r'\b' + kw + r'\b', query):
                return {"question_type": "aggregation", "engine": "csv_engine"}
        
        # 2. Check for ID lookups
        for kw in self.lookup_keywords:
            if re.search(r'\b' + kw + r'\b', query) and any(char.isdigit() for char in query):
                return {"question_type": "lookup", "engine": "csv_engine"}
        
        # 3. Check for Semantic / Meaning-based
        for kw in self.semantic_keywords:
            if re.search(r'\b' + kw + r'\b', query):
                return {"question_type": "semantic", "engine": "vector_engine"}
        
        # Default to Hybrid or Vector for ambiguous/judgment questions
        return {"question_type": "semantic", "engine": "vector_engine"}
