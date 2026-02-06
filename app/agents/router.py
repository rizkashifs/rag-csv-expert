import json
import logging
import re
from typing import Any, Dict, List

from app.agents.base import BaseAgent
from app.models.llm_client import llm_client
from app.utils.routing_keywords import PROFILE_KEYWORDS, SIMPLE_INTENT_PATTERNS


class RouterAgent(BaseAgent):
    """
    Classifies user questions and decides which data engine should be used.
    Uses context-aware + keyword routing, with LLM fallback.
    """

    def __init__(self):
        self.routes = {"PROFILE_ONLY", "TEXT_TABLE_RAG", "SQL_ENGINE", "KEYWORD_ENGINE", "REFUSE"}
        self.simple_intent_patterns = SIMPLE_INTENT_PATTERNS
        self.profile_keywords = PROFILE_KEYWORDS
        self.logger = logging.getLogger(__name__)

    def _extract_columns_from_query(self, query: str) -> List[str]:
        quoted_cols = re.findall(r'"([^"]+)"', query)
        if quoted_cols:
            return quoted_cols

        compare_match = re.search(r"\bcompare\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:and|with|vs)\s+([a-zA-Z_][a-zA-Z0-9_]*)", query, re.IGNORECASE)
        if compare_match:
            return [compare_match.group(1), compare_match.group(2)]

        agg_of_match = re.search(r"\b(?:sum|total|average|avg|mean|max|min|count)\s+of\s+([a-zA-Z_][a-zA-Z0-9_]*)", query, re.IGNORECASE)
        if agg_of_match:
            return [agg_of_match.group(1)]

        return re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", query)

    def _extract_filters(self, query: str) -> List[Dict[str, Any]]:
        filters: List[Dict[str, Any]] = []
        for column, op, value in re.findall(r'"([^"]+)"\s*(>=|<=|!=|=|>|<)\s*([^\s,]+)', query):
            filters.append({"column": column, "operator": op, "value": value.strip("\"'")})
        return filters

    def _keyword_intent(self, query: str) -> Dict[str, Any]:
        operation = "none"
        for op, patterns in self.simple_intent_patterns.items():
            if any(re.search(pattern, query, re.IGNORECASE) for pattern in patterns):
                operation = op
                break

        columns = self._extract_columns_from_query(query)
        filters = self._extract_filters(query)

        return {
            "operation": operation,
            "columns": columns[:3],
            "filters": filters,
            "group_by": [],
        }

    def _is_contextual_follow_up(self, query: str) -> bool:
        normalized = query.strip().lower()
        if not normalized:
            return False

        follow_up_markers = (
            "and ",
            "also ",
            "what about",
            "how about",
            "same",
            "that",
            "those",
            "them",
            "it",
            "for those",
            "for that",
            "by ",
        )
        is_short = len(normalized.split()) <= 8
        return is_short and any(normalized.startswith(marker) for marker in follow_up_markers)

    def _build_history_aware_query(self, raw_query: str, history: List[Dict[str, str]]) -> str:
        if not history or not self._is_contextual_follow_up(raw_query):
            return raw_query

        last_user_query = history[-1].get("user", "").strip()
        if not last_user_query:
            return raw_query

        return f"{last_user_query} {raw_query}".strip()

    def run(self, input_data: dict) -> dict:
        raw_query = input_data.get("query", "")
        query = raw_query.lower()
        dataset_profile = input_data.get("dataset_profile", "")
        semantic_summary = input_data.get("semantic_summary", "")
        text_heavy = input_data.get("text_heavy", False)
        history = input_data.get("history", [])

        history_text = "\n".join(
            [f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}" for turn in history[-5:]]
        )
        history_aware_query = self._build_history_aware_query(raw_query, history)

        for kw in self.profile_keywords:
            if re.search(r"\b" + kw + r"\b", query):
                return {
                    "route": "PROFILE_ONLY",
                    "use_routing_agent": True,
                    "schema": {"operation": "profile", "columns": [], "filters": [], "group_by": []},
                }

        keyword_intent = self._keyword_intent(history_aware_query)
        if keyword_intent["operation"] != "none":
            return {
                "route": "KEYWORD_ENGINE",
                "use_routing_agent": True,
                "schema": {
                    **keyword_intent,
                    "engine_mode": "simple",
                },
            }

        prompt = f"""
        You are a routing classifier for a CSV/Excel RAG system.
        Decide the best route based on user query, dataset context, and prior chat history.

        Routes:
        - PROFILE_ONLY: schema/profile/summary requests only.
        - TEXT_TABLE_RAG: semantic interpretation from text-heavy columns.
        - KEYWORD_ENGINE: simple single-intent analytical requests.
        - SQL_ENGINE: advanced analytical queries (complex filtering, exact conditions, multi-step intent).
        - REFUSE: unclear question or insufficient information.

        Dataset Profile:
        {dataset_profile}

        Dataset Summary:
        {semantic_summary}

        Conversation History:
        {history_text}

        Text-Heavy: {text_heavy}

        User Query: {raw_query}

        Return ONLY JSON in this shape:
        {{
          "route": "SQL_ENGINE",
          "schema": {{
            "operation": "sum|avg|count|max|min|correlation|filter|none",
            "columns": ["col1"],
            "filters": [{{"column": "Sales", "operator": ">", "value": "1000"}}],
            "group_by": []
          }}
        }}
        """

        try:
            messages = [{"role": "user", "content": prompt}]
            response = llm_client.generate(messages, options={"temperature": 0.0})
            parsed = json.loads(response.strip())
            route = parsed.get("route", "").upper().strip()
            schema = parsed.get("schema", {})
            if route in self.routes:
                return {"route": route, "use_routing_agent": True, "schema": schema}
        except Exception:
            self.logger.info("Router LLM failed, using fallback rules.")

        if text_heavy:
            return {
                "route": "TEXT_TABLE_RAG",
                "use_routing_agent": True,
                "schema": {"operation": "semantic", "columns": [], "filters": [], "group_by": []},
            }

        return {
            "route": "REFUSE",
            "use_routing_agent": True,
            "schema": {"operation": "none", "columns": [], "filters": [], "group_by": []},
        }
