import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.agents.base import BaseAgent
from app.models.llm_client import llm_client
from app.services.history import get_history

class RouterAgent(BaseAgent):
    """Route user queries to the best engine and return a structured route schema."""

    ROUTES = {"PROFILE_ONLY", "TEXT_TABLE_RAG", "SQL_ENGINE", "KEYWORD_ENGINE", "REFUSE"}
    OPERATIONS_REQUIRING_COLUMN = {"sum", "avg", "max", "min", "correlation"}

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _base_schema(self, operation: str = "none") -> Dict[str, Any]:
        return {
            "operation": operation,
            "columns": [],
            "filters": [],
            "filters": [],
            "group_by": [],
            "having": [],
            "aggregations": [],
            "sort": [],
            "limit": None,
            "engine_mode": "default",
            "confidence": 0.0,
        }

    def _sql_schema(self, operation: str = "none") -> Dict[str, Any]:
        schema = self._base_schema(operation)
        schema.update(
            {
                "engine_mode": "sql",
                "sql_plan": {
                    "target_columns": [],
                    "aggregations": [],
                    "filters": [],
                    "group_by": [],
                    "having": [],
                    "order_by": [],
                    "limit": None,
                },
            }
        )
        return schema

    def _semantic_schema(self) -> Dict[str, Any]:
        schema = self._base_schema("semantic")
        schema.update(
            {
                "engine_mode": "semantic",
                "semantic_plan": {
                    "query_text": "",
                    "target_text_columns": [],
                    "semantic_intent": "row_matching",
                    "top_k": 8,
                    "min_similarity": None,
                    "post_filters": [],
                },
            }
        )
        return schema



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
        return len(normalized.split()) <= 8 and any(normalized.startswith(marker) for marker in follow_up_markers)

    
    



    def _sync_sql_schema_layers(self, normalized: Dict[str, Any]) -> Dict[str, Any]:
        sql_plan = normalized.get("sql_plan") or {}

        # Pull nested plan fields up for the SQL executor, then push normalized values back down.
        if not normalized.get("columns"):
            normalized["columns"] = list(sql_plan.get("target_columns") or [])
        if not normalized.get("filters"):
            normalized["filters"] = list(sql_plan.get("filters") or [])
        if not normalized.get("group_by"):
            normalized["group_by"] = list(sql_plan.get("group_by") or [])
        if not normalized.get("aggregations"):
            normalized["aggregations"] = list(sql_plan.get("aggregations") or [])
        if not normalized.get("sort"):
            normalized["sort"] = list(sql_plan.get("order_by") or [])
        if not normalized.get("having"):
            normalized["having"] = list(sql_plan.get("having") or [])
        if normalized.get("limit") is None and sql_plan.get("limit") is not None:
            normalized["limit"] = sql_plan.get("limit")

        sql_plan["target_columns"] = normalized.get("columns", [])
        sql_plan["filters"] = normalized.get("filters", [])
        sql_plan["group_by"] = normalized.get("group_by", [])
        sql_plan["aggregations"] = normalized.get("aggregations", [])
        sql_plan["order_by"] = normalized.get("sort", [])
        sql_plan["limit"] = normalized.get("limit")
        sql_plan["having"] = normalized.get("having", [])
        normalized["sql_plan"] = sql_plan
        return normalized

    def _build_refusal_schema(self, reason: str, follow_up_questions: List[str]) -> Dict[str, Any]:
        schema = self._normalize_schema("REFUSE", {"operation": "none", "confidence": 0.35}, "")
        schema.update({"reason": reason, "follow_up_questions": follow_up_questions})
        return schema

    def _clarification_questions_for_schema(self, schema: Dict[str, Any]) -> List[str]:
        operation = (schema.get("operation") or "none").lower()
        questions = []
        if operation in self.OPERATIONS_REQUIRING_COLUMN and not schema.get("columns"):
            questions.append("Which numeric column should I use for this calculation?")
        if operation == "correlation" and len(schema.get("columns", [])) < 2:
            questions.append("Please specify two numeric columns to compute correlation.")
        return questions

    def _should_refuse(self, route: str, schema: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
        if route not in {"SQL_ENGINE", "KEYWORD_ENGINE", "TEXT_TABLE_RAG"}:
            return None

        operation = (schema.get("operation") or "none").lower()
        if route in {"SQL_ENGINE", "KEYWORD_ENGINE"}:
            questions = self._clarification_questions_for_schema(schema)
            if operation == "none":
                questions.append("Do you want a sum, average, count, min/max, or filtered rows?")

            has_filter_language = bool(re.search(r"\bwhere\b|\bfilter\b|\bgreater than\b|\bless than\b|>=|<=|!=|=|>|<", query, re.IGNORECASE))
            if has_filter_language and not schema.get("filters"):
                questions.append("What exact filter condition should I apply (column, operator, value)?")

            if questions:
                return {
                    "route": "REFUSE",
                    "use_routing_agent": True,
                    "schema": self._build_refusal_schema("insufficient_structured_intent", questions),
                }

        if route == "TEXT_TABLE_RAG":
            semantic_plan = schema.get("semantic_plan") or {}
            if not semantic_plan.get("query_text"):
                return {
                    "route": "REFUSE",
                    "use_routing_agent": True,
                    "schema": self._build_refusal_schema(
                        "missing_semantic_query",
                        ["What text meaning or theme should I search for in the rows?"],
                    ),
                }

        return None

    def _normalize_schema(self, route: str, schema: Dict[str, Any], raw_query: str) -> Dict[str, Any]:
        if route == "TEXT_TABLE_RAG":
            normalized = self._semantic_schema()
            normalized.update(schema or {})
            semantic_plan = normalized.get("semantic_plan") or {}
            semantic_plan.setdefault("query_text", raw_query)
            normalized["semantic_plan"] = semantic_plan
            return normalized

        if route in {"KEYWORD_ENGINE", "SQL_ENGINE", "PROFILE_ONLY", "REFUSE"}:
            operation = (schema or {}).get("operation", "profile" if route == "PROFILE_ONLY" else "none")
            normalized = self._sql_schema(operation)
            normalized.update(schema or {})
            return self._sync_sql_schema_layers(normalized)

        return schema or self._base_schema()

    def _build_llm_prompt(
        self,
        query: str,
        dataset_profile: str,
        semantic_summary: str,
        history_text: str,
        text_heavy: bool,
    ) -> str:
        return f"""
You are a router for a CSV/Excel analytics assistant.
Your goal is to route the user's query to the correct engine and structure the intent (columns, operations, filters).

CRITICAL:
1. Column Mapping: The user may type column names inexactly (e.g., "sales" instead of "Total Sales" or "revenue" instead of "Rev_2023"). 
   You MUST map their terms to the CLOSEST VALID COLUMN NAME from the "Dataset Profile". 
   If a column does not exist, do not invent one.
2. Typos: Fix obvious typos in column names or values.

Routes:
- PROFILE_ONLY: schema/profile/summary requests.
- TEXT_TABLE_RAG: semantic search or natural-language matching over text-heavy fields.
- KEYWORD_ENGINE: simple single-step analytics (e.g. "show rows where...", "count of...").
- SQL_ENGINE: complex analytics (aggregations + filters/grouping/sorting/comparisons).
- REFUSE: ambiguous or unanswerable query. If details are missing, return REFUSE with follow_up_questions.

Dataset Profile:
{dataset_profile}

Dataset Summary:
{semantic_summary}

Conversation History:
{history_text}

Text-Heavy Dataset: {text_heavy}

User Query: {query}

Return ONLY valid JSON using this schema:
{{
  "route": "SQL_ENGINE|KEYWORD_ENGINE|TEXT_TABLE_RAG|PROFILE_ONLY|REFUSE",
  "schema": {{
    "operation": "sum|avg|count|max|min|correlation|filter|semantic|profile|none",
    "columns": ["ExactColumnNameFromProfile"],
    "filters": [{{"column": "ExactColumnName", "operator": ">", "value": "1000"}}],
    "group_by": [{{"column": "ExactColumnName", "time_grain": "year|month|day|null"}}],
    "having": [{{"column": "ExactColumnName", "operator": ">", "value": "1000"}}],
    "aggregations": [{{"function": "sum", "column": "ExactColumnName"}}],
    "sort": [{{"column": "ExactColumnName", "direction": "desc"}}],
    "limit": 10,
    "engine_mode": "sql|simple|semantic",
    "confidence": 0.0,
    "sql_plan": {{
      "target_columns": ["ExactColumnName"],
      "aggregations": [{{"function": "sum", "column": "ExactColumnName"}}],
      "filters": [{{"column": "ExactColumnName", "operator": "=", "value": "West"}}],
      "group_by": [{{"column": "ExactColumnName", "time_grain": "year|month|day|null"}}],
      "having": [{{"column": "ExactColumnName", "operator": ">", "value": "1000"}}],
      "order_by": [{{"column": "ExactColumnName", "direction": "desc"}}],
      "limit": 10
    }},
    "semantic_plan": {{
      "query_text": "{query}",
      "target_text_columns": ["ExactColumnName"],
      "semantic_intent": "row_matching|theme_extraction|qa",
      "top_k": 8,
      "min_similarity": null,
      "post_filters": []
    }},
    "reason": "optional reason for refusal",
    "follow_up_questions": ["optional clarifying question"]
  }}
}}
""".strip()

    def run(self, input_data: dict) -> dict:
        raw_query = input_data.get("query", "")
        normalized_query = raw_query.lower()
        dataset_profile = input_data.get("dataset_profile", "")
        semantic_summary = input_data.get("semantic_summary", "")
        text_heavy = input_data.get("text_heavy", False)

        chat_id = input_data.get("chat_id")
        history = input_data.get("history")
        if history is None and chat_id:
            history = get_history(chat_id)
        history = history or []

        history_text = "\n".join(
            [f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}" for turn in history[-5:]]
        )
        history_aware_query = self._build_history_aware_query(raw_query, history)

        # Direct LLM call - No Regex Fast Path
        prompt = self._build_llm_prompt(history_aware_query, dataset_profile, semantic_summary, history_text, text_heavy)
        try:
            response = llm_client.generate([{"role": "user", "content": prompt}], options={"temperature": 0.0})
            
            # Cleanup JSON block if markdown fences exist
            clean_response = response.strip()
            if "```json" in clean_response:
                clean_response = clean_response.split("```json")[-1].split("```")[0].strip()
            elif "```" in clean_response:
                clean_response = clean_response.split("```")[-1].split("```")[0].strip()
            
            parsed = json.loads(clean_response)
            route = parsed.get("route", "").upper().strip()
            
            if route in self.ROUTES:
                normalized = self._normalize_schema(route, parsed.get("schema", {}), raw_query)
                refused = self._should_refuse(route, normalized, history_aware_query)
                return refused or {"route": route, "use_routing_agent": True, "schema": normalized}
        except Exception as e:
            self.logger.error(f"Router LLM failed: {e}") 
            # Fallback only on error
            if text_heavy:
                semantic_schema = self._semantic_schema()
                semantic_schema["confidence"] = 0.65
                semantic_schema["semantic_plan"]["query_text"] = raw_query
                return {
                    "route": "TEXT_TABLE_RAG",
                    "use_routing_agent": True,
                    "schema": self._normalize_schema("TEXT_TABLE_RAG", semantic_schema, raw_query),
                }

        return {
            "route": "REFUSE",
            "use_routing_agent": True,
            "schema": self._build_refusal_schema(
                "unable_to_route_with_confidence",
                [
                    "Could you clarify which metric/column you want?",
                    "Should I apply any filters such as region/date/value thresholds?",
                ],
            ),
        }
