import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.agents.base import BaseAgent
from app.models.llm_client import llm_client
from app.services.history import get_history
from app.utils.routing_keywords import PROFILE_KEYWORDS, SIMPLE_INTENT_PATTERNS


class RouterAgent(BaseAgent):
    """Route user queries to the best engine and return a structured route schema."""

    ROUTES = {"PROFILE_ONLY", "TEXT_TABLE_RAG", "SQL_ENGINE", "KEYWORD_ENGINE", "REFUSE"}
    OPERATIONS_REQUIRING_COLUMN = {"sum", "avg", "max", "min", "correlation"}

    def __init__(self):
        self.simple_intent_patterns = SIMPLE_INTENT_PATTERNS
        self.profile_keywords = PROFILE_KEYWORDS
        self.logger = logging.getLogger(__name__)

    def _base_schema(self, operation: str = "none") -> Dict[str, Any]:
        return {
            "operation": operation,
            "columns": [],
            "filters": [],
            "group_by": [],
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

    def _extract_columns_from_query(self, query: str) -> List[str]:
        quoted_cols = re.findall(r'"([^"]+)"', query)
        if quoted_cols:
            return quoted_cols

        compare_match = re.search(
            r"\bcompare\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:and|with|vs)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            query,
            re.IGNORECASE,
        )
        if compare_match:
            return [compare_match.group(1), compare_match.group(2)]

        agg_of_match = re.search(
            r"\b(?:sum|total|average|avg|mean|max|min|count)\s+of\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            query,
            re.IGNORECASE,
        )
        if agg_of_match:
            return [agg_of_match.group(1)]

        return re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", query)

    def _extract_filters(self, query: str) -> List[Dict[str, Any]]:
        filters: List[Dict[str, Any]] = []

        for column, op, value in re.findall(r'"([^"]+)"\s*(>=|<=|!=|=|>|<)\s*([^\s,]+)', query):
            filters.append({"column": column, "operator": op, "value": value.strip("\"'")})

        value_pattern = r"([a-zA-Z0-9_\-]+)"
        natural_language_patterns = [
            (rf"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+greater than\s+{value_pattern}", ">"),
            (rf"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+less than\s+{value_pattern}", "<"),
            (rf"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+at least\s+{value_pattern}", ">="),
            (rf"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+at most\s+{value_pattern}", "<="),
            (rf"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+equals?\s+{value_pattern}", "="),
        ]
        for pattern, operator in natural_language_patterns:
            for column, value in re.findall(pattern, query, re.IGNORECASE):
                filters.append({"column": column, "operator": operator, "value": value.strip("\"'")})

        return filters

    def _extract_group_by(self, query: str) -> List[str]:
        groups: List[str] = []
        by_match = re.search(r"\bby\s+([a-zA-Z_][a-zA-Z0-9_]*)", query, re.IGNORECASE)
        if by_match:
            groups.append(by_match.group(1))

        explicit_groups = re.findall(
            r"\bgroup(?:ed)?\s+by\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)",
            query,
            re.IGNORECASE,
        )
        for group_text in explicit_groups:
            groups.extend([name.strip() for name in group_text.split(",") if name.strip()])

        return list(dict.fromkeys(groups))[:3]

    def _extract_sort(self, query: str) -> List[Dict[str, str]]:
        sorts: List[Dict[str, str]] = []
        for column in re.findall(r"\border(?:ed)?\s+by\s+([a-zA-Z_][a-zA-Z0-9_]*)", query, re.IGNORECASE):
            direction = "desc" if re.search(r"\bdesc|highest|top\b", query, re.IGNORECASE) else "asc"
            sorts.append({"column": column, "direction": direction})
        return sorts[:2]

    def _extract_limit(self, query: str) -> Optional[int]:
        match = re.search(r"\b(?:top|first|limit)\s+(\d{1,4})\b", query, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _extract_aggregations(self, query: str, columns: List[str]) -> List[Dict[str, str]]:
        aliases = {
            "sum": ["sum", "total"],
            "avg": ["average", "avg", "mean"],
            "count": ["count", "how many", "number of"],
            "max": ["max", "highest", "largest"],
            "min": ["min", "lowest", "smallest"],
        }
        aggregations: List[Dict[str, str]] = []
        for function, words in aliases.items():
            if any(re.search(rf"\b{re.escape(word)}\b", query, re.IGNORECASE) for word in words):
                aggregations.append({"function": function, "column": columns[0] if columns else "*"})
        return aggregations[:2]

    def _looks_semantic(self, query: str) -> bool:
        patterns = [
            r"\bsimilar\b",
            r"\bclosest\b",
            r"\bsemantic\b",
            r"\bmatch(?:ing)?\b",
            r"\bcontains\b",
            r"\bdescrib(?:e|ing)\b",
            r"\babout\b",
            r"\btheme\b",
            r"\breview\b",
            r"\btext\b",
        ]
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in patterns)

    def _is_complex_sql(self, query: str, filters: List[Dict[str, Any]], group_by: List[str]) -> bool:
        complexity_patterns = [
            r"\bbetween\b",
            r"\bhaving\b",
            r"\btrend\b",
            r"\bper\b",
            r"\bcompare\b",
            r"\bdistribution\b",
            r"\bjoin\b",
            r"\brank\b",
            r"\bpercent(?:age)?\b",
            r"\band\b.*\band\b",
            r"\bor\b",
        ]

        score = 0
        if len(filters) >= 2:
            score += 1
        if group_by:
            score += 1
        if self._extract_limit(query) is not None:
            score += 1
        if any(re.search(pattern, query, re.IGNORECASE) for pattern in complexity_patterns):
            score += 1
        return score >= 2

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

    def _build_history_aware_query(self, raw_query: str, history: List[Dict[str, str]]) -> str:
        if not history or not self._is_contextual_follow_up(raw_query):
            return raw_query
        last_user_query = history[-1].get("user", "").strip()
        return f"{last_user_query} {raw_query}".strip() if last_user_query else raw_query

    def _keyword_intent(self, query: str) -> Dict[str, Any]:
        operation = "none"
        for op, patterns in self.simple_intent_patterns.items():
            if any(re.search(pattern, query, re.IGNORECASE) for pattern in patterns):
                operation = op
                break

        return {
            "operation": operation,
            "columns": self._extract_columns_from_query(query)[:5],
            "filters": self._extract_filters(query),
            "group_by": [],
        }

    def _build_keyword_schema(self, query: str, keyword_intent: Dict[str, Any]) -> Dict[str, Any]:
        columns = keyword_intent.get("columns", [])
        filters = keyword_intent.get("filters", [])
        group_by = self._extract_group_by(query)
        sort = self._extract_sort(query)
        limit = self._extract_limit(query)
        aggregations = self._extract_aggregations(query, columns)

        schema = self._sql_schema(keyword_intent.get("operation", "none"))
        schema.update(
            {
                "columns": columns,
                "filters": filters,
                "group_by": group_by,
                "aggregations": aggregations,
                "sort": sort,
                "limit": limit,
                "engine_mode": "sql" if self._is_complex_sql(query, filters, group_by) else "simple",
                "confidence": 0.78,
            }
        )
        schema["sql_plan"].update(
            {
                "target_columns": columns,
                "aggregations": aggregations,
                "filters": filters,
                "group_by": group_by,
                "order_by": sort,
                "limit": limit,
            }
        )
        return schema

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
        if normalized.get("limit") is None and sql_plan.get("limit") is not None:
            normalized["limit"] = sql_plan.get("limit")

        sql_plan["target_columns"] = normalized.get("columns", [])
        sql_plan["filters"] = normalized.get("filters", [])
        sql_plan["group_by"] = normalized.get("group_by", [])
        sql_plan["aggregations"] = normalized.get("aggregations", [])
        sql_plan["order_by"] = normalized.get("sort", [])
        sql_plan["limit"] = normalized.get("limit")
        sql_plan.setdefault("having", [])
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
Select exactly one route.

Routes:
- PROFILE_ONLY: schema/profile/summary requests.
- TEXT_TABLE_RAG: semantic search or natural-language matching over text-heavy fields.
- KEYWORD_ENGINE: simple single-step analytics.
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
    "columns": ["col"],
    "filters": [{{"column": "Sales", "operator": ">", "value": "1000"}}],
    "group_by": ["Region"],
    "aggregations": [{{"function": "sum", "column": "Sales"}}],
    "sort": [{{"column": "Sales", "direction": "desc"}}],
    "limit": 10,
    "engine_mode": "sql|simple|semantic",
    "confidence": 0.0,
    "sql_plan": {{
      "target_columns": ["Sales"],
      "aggregations": [{{"function": "sum", "column": "Sales"}}],
      "filters": [{{"column": "Region", "operator": "=", "value": "West"}}],
      "group_by": ["Region"],
      "having": [],
      "order_by": [{{"column": "Sales", "direction": "desc"}}],
      "limit": 10
    }},
    "semantic_plan": {{
      "query_text": "{query}",
      "target_text_columns": ["description"],
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

        if any(re.search(rf"\b{re.escape(keyword)}\b", normalized_query) for keyword in self.profile_keywords):
            profile_schema = self._normalize_schema("PROFILE_ONLY", {"operation": "profile", "confidence": 0.98}, raw_query)
            return {"route": "PROFILE_ONLY", "use_routing_agent": True, "schema": profile_schema}

        keyword_intent = self._keyword_intent(history_aware_query)
        if keyword_intent["operation"] != "none":
            schema = self._build_keyword_schema(history_aware_query, keyword_intent)
            route = "SQL_ENGINE" if schema["engine_mode"] == "sql" else "KEYWORD_ENGINE"
            refused = self._should_refuse(route, schema, history_aware_query)
            return refused or {"route": route, "use_routing_agent": True, "schema": self._normalize_schema(route, schema, raw_query)}

        if text_heavy and self._looks_semantic(history_aware_query):
            semantic_schema = self._semantic_schema()
            semantic_schema["confidence"] = 0.8
            semantic_schema["semantic_plan"]["query_text"] = raw_query
            route_payload = {
                "route": "TEXT_TABLE_RAG",
                "use_routing_agent": True,
                "schema": self._normalize_schema("TEXT_TABLE_RAG", semantic_schema, raw_query),
            }
            refused = self._should_refuse(route_payload["route"], route_payload["schema"], history_aware_query)
            return refused or route_payload

        prompt = self._build_llm_prompt(raw_query, dataset_profile, semantic_summary, history_text, text_heavy)
        try:
            response = llm_client.generate([{"role": "user", "content": prompt}], options={"temperature": 0.0})
            parsed = json.loads(response.strip())
            route = parsed.get("route", "").upper().strip()
            if route in self.ROUTES:
                normalized = self._normalize_schema(route, parsed.get("schema", {}), raw_query)
                refused = self._should_refuse(route, normalized, history_aware_query)
                return refused or {"route": route, "use_routing_agent": True, "schema": normalized}
        except Exception:
            self.logger.info("Router LLM failed, using fallback rules.")

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
