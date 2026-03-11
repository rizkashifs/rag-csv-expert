import json
import re
from typing import Any, Dict, List, Optional

from app.agents.base import BaseAgent
from app.models.llm_client import llm_client
from app.services.history import get_history
from app.utils.history_utils import truncate_history
from app.utils.logger import logger

class RouterAgent(BaseAgent):
    """Route user queries to the best engine and return a structured route schema."""

    ROUTES = {"PROFILE_ONLY", "TEXT_TABLE_RAG", "SQL_ENGINE", "REFUSE"}
    OPERATIONS_REQUIRING_COLUMN = {
        "sum",
        "avg",
        "max",
        "min",
        "median",
        "mode",
        "std",
        "variance",
        "quantile",
        "histogram",
        "value_counts",
        "distinct_count",
        "null_count",
        "null_pct",
        "correlation",
    }

    def __init__(self):
        self.logger = logger

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
                    "keywords": [],
                    "id_filters": [],
                    "target_text_columns": [],
                    "semantic_intent": "row_matching",
                    "top_k": 8,
                    "min_similarity": None,
                    "post_filters": [],
                },
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

    def _extract_sheet_columns_from_profile(self, dataset_profile: str) -> Dict[str, List[str]]:
        if not dataset_profile or "Workbook with" not in dataset_profile:
            return {}

        sheet_columns: Dict[str, List[str]] = {}
        for line in dataset_profile.splitlines():
            line = line.strip()
            if not line or line in {"Sheets:"} or line.startswith("Workbook with") or line.startswith("Total rows:") or line.startswith("Columns (union):"):
                continue

            match = re.match(r"([^:]+):\s*rows\s*=\s*\d+,\s*cols\s*=\s*\d+,\s*columns=\[(.*)\]", line)
            if not match:
                continue

            sheet_name = match.group(1).strip()
            columns_text = match.group(2).strip()
            columns = [col.strip() for col in columns_text.split(",") if col.strip()]
            sheet_columns[sheet_name] = columns

        return sheet_columns

    def _schema_has_sheet_filter(self, schema: Dict[str, Any]) -> bool:
        return any(
            isinstance(item, dict) and str(item.get("column", "")).strip().lower() == "sheet"
            for item in (schema.get("filters") or [])
        )

    def _collect_referenced_columns(self, schema: Dict[str, Any]) -> List[str]:
        referenced: List[str] = []
        seen = set()

        def add(value: Any) -> None:
            if value is None:
                return
            text = str(value).strip()
            if not text:
                return
            lowered = text.lower()
            if lowered == "sheet" or lowered in seen:
                return
            seen.add(lowered)
            referenced.append(text)

        for col in schema.get("columns") or []:
            add(col)

        for key in ("filters", "group_by", "having", "aggregations", "sort"):
            for item in schema.get(key) or []:
                if isinstance(item, dict):
                    add(item.get("column"))
                elif key == "group_by":
                    add(item)

        return referenced

    def _get_ambiguous_sheet_question(self, schema: Dict[str, Any], dataset_profile: str) -> Optional[str]:
        sheet_columns = self._extract_sheet_columns_from_profile(dataset_profile)
        if len(sheet_columns) < 2 or self._schema_has_sheet_filter(schema):
            return None

        referenced_columns = self._collect_referenced_columns(schema)
        if not referenced_columns:
            return None

        ambiguous_sheets = set()
        for column in referenced_columns:
            matches = [
                sheet_name
                for sheet_name, columns in sheet_columns.items()
                if any(existing.lower() == column.lower() for existing in columns)
            ]
            if len(matches) > 1:
                ambiguous_sheets.update(matches)

        if not ambiguous_sheets:
            return None

        joined = ", ".join(sorted(ambiguous_sheets))
        return f"Which sheet should I use: {joined}?"

    def _clarification_questions_for_schema(self, schema: Dict[str, Any]) -> List[str]:
        operation = (schema.get("operation") or "none").lower()
        questions = []
        if operation in self.OPERATIONS_REQUIRING_COLUMN and not schema.get("columns"):
            questions.append("Which numeric column should I use for this calculation?")
        if operation == "correlation" and len(schema.get("columns", [])) < 2:
            questions.append("Please specify two numeric columns to compute correlation.")
        return questions

    def _should_refuse(self, route: str, schema: Dict[str, Any], query: str, dataset_profile: str = "") -> Optional[Dict[str, Any]]:
        if route not in {"SQL_ENGINE", "TEXT_TABLE_RAG"}:
            return None

        operation = (schema.get("operation") or "none").lower()
        if route == "SQL_ENGINE":
            questions = self._clarification_questions_for_schema(schema)
            if operation == "none":
                questions.append("Do you want a sum, average, count, min/max, or filtered rows?")

            has_filter_language = bool(re.search(r"\bwhere\b|\bfilter\b|\bgreater than\b|\bless than\b|>=|<=|!=|=|>|<", query, re.IGNORECASE))
            if has_filter_language and not schema.get("filters"):
                questions.append("What exact filter condition should I apply (column, operator, value)?")

            ambiguous_sheet_question = self._get_ambiguous_sheet_question(schema, dataset_profile)
            if ambiguous_sheet_question:
                questions.append(ambiguous_sheet_question)

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

        if route in {"SQL_ENGINE", "PROFILE_ONLY", "REFUSE"}:
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
3. Sheet Intent: If the user explicitly mentions a worksheet/tab/sheet name such as "Sheet1", "sheet 2", or "Employees", preserve that scope as a SQL filter using column "sheet". Do not ignore explicit sheet scope when the same column exists in multiple sheets.

Routes:
- PROFILE_ONLY: schema/profile/summary requests.
- TEXT_TABLE_RAG: semantic search or natural-language matching over text-heavy fields.

- SQL_ENGINE: complex analytics (aggregations + filters/grouping/sorting/comparisons).
- REFUSE: ambiguous or unanswerable query. If details are missing, return REFUSE with follow_up_questions.

Dataset Profile:
{dataset_profile}

Dataset Summary:
{semantic_summary}

Recent User Queries (for context):
{history_text}

Text-Heavy Dataset: {text_heavy}

User Query: {query}

Return ONLY valid JSON using this schema:
{{
  "route": "SQL_ENGINE|TEXT_TABLE_RAG|PROFILE_ONLY|REFUSE",
  "schema": {{
    "operation": "sum|avg|count|max|min|median|mode|std|variance|quantile|histogram|value_counts|distinct_count|null_count|null_pct|correlation|filter|semantic|profile|none",
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
      "keywords": ["keyword1", "keyword2"],
      "id_filters": [{{"column": "ExactIdColumnName", "value": "1234"}}],
      "target_text_columns": ["ExactColumnName"],
      "semantic_intent": "row_matching|theme_extraction|qa",
      "top_k": 8,
      "min_similarity": null,
      "post_filters": [{{"column": "ExactColumnName", "operator": "=", "value": "someValue"}}]
    }},
    "reason": "optional reason for refusal",
    "follow_up_questions": ["optional clarifying question"]
  }}
}}

INSTRUCTIONS for semantic_plan fields:
- keywords: ALWAYS extract the key search terms from the user query (no stopwords like "the", "and", "show", etc).
- id_filters: if the user references a specific row by any identifier (e.g. "employee 1234", "order #99"), set column and value here.
- target_text_columns: the dataset columns that contain free-text to search within.
- post_filters: additional column filters (operator/value) to apply AFTER the text search has matched rows.

INSTRUCTIONS for SQL sheet scoping:
- If the query says "in Sheet1", "from Employees sheet", "only Sheet2", or otherwise names a worksheet/tab, add a filter like {{"column": "sheet", "operator": "=", "value": "Sheet1"}}.
- Keep the same sheet filter in both top-level schema.filters and schema.sql_plan.filters for SQL_ENGINE routes.
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

        # Truncate to last 5 user prompts only (no assistant responses)
        history_text = truncate_history(history, max_user_turns=5)

        # Direct LLM call - No Regex Fast Path
        prompt = self._build_llm_prompt(raw_query, dataset_profile, semantic_summary, history_text, text_heavy)
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
                refused = self._should_refuse(route, normalized, raw_query, dataset_profile)
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
