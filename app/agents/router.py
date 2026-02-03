import json
import logging
import re
from app.agents.base import BaseAgent
from app.models.llm_client import llm_client

class RouterAgent(BaseAgent):
    """
    Classifies user questions and decides which data engine should be used.
    Uses LLM-based classification with a lightweight rule-based fallback.
    """
    def __init__(self):
        self.routes = {"PROFILE_ONLY", "TEXT_TABLE_RAG", "PANDAS", "REFUSE"}
        self.aggregation_keywords = [
            "sum", "average", "avg", "total", "count", "max", "min", "mean", "median",
            "correlation", "corr", "relationship", "relate", "linked", "percent", "ratio"
        ]
        self.profile_keywords = [
            "schema", "columns", "fields", "profile", "overview", "summary"
        ]
        self.refuse_keywords = [
            "password", "secret", "social security", "credit card", "ssn"
        ]
        self.logger = logging.getLogger(__name__)

    def run(self, input_data: dict) -> dict:
        """
        Input: {"query": str, "dataset_profile": str, "semantic_summary": str, "text_heavy": bool}
        Output: {"route": str}
        """
        query = input_data.get("query", "").lower()
        dataset_profile = input_data.get("dataset_profile", "")
        semantic_summary = input_data.get("semantic_summary", "")
        text_heavy = input_data.get("text_heavy", False)

        for kw in self.refuse_keywords:
            if kw in query:
                return {"route": "REFUSE"}

        prompt = f"""
        You are a routing classifier for a CSV/Excel RAG system.
        Decide the best route based on the user query and dataset context.
        Routes:
        - PROFILE_ONLY: user wants schema/profile/summary info only.
        - TEXT_TABLE_RAG: user wants semantic meaning from text-heavy columns.
        - PANDAS: user needs calculations, aggregations, filters, or exact values.
        - REFUSE: query is unsafe or unrelated to the data.

        Dataset Profile:
        {dataset_profile}

        5-Row Semantic Summary:
        {semantic_summary}

        Text-Heavy: {text_heavy}

        User Query: {query}

        Respond with ONLY a JSON object like:
        {{"route": "PROFILE_ONLY"}}
        """

        try:
            messages = [{"role": "user", "content": prompt}]
            response = llm_client.generate(messages, options={"temperature": 0.0})
            parsed = json.loads(response.strip())
            route = parsed.get("route", "").upper().strip()
            if route in self.routes:
                return {"route": route}
        except Exception:
            self.logger.info("Router LLM failed, using fallback rules.")

        for kw in self.profile_keywords:
            if re.search(r'\b' + kw + r'\b', query):
                return {"route": "PROFILE_ONLY"}

        for kw in self.aggregation_keywords:
            if re.search(r'\b' + kw + r'\b', query):
                return {"route": "PANDAS"}

        return {"route": "TEXT_TABLE_RAG" if text_heavy else "PANDAS"}
