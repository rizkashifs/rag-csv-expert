import re
import logging
from typing import Dict, Any, List

class KeywordEngine:
    """
    Regex-based intent extraction engine.
    Used ONLY as a fallback when the LLM Router crashes.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def run(self, query: str, schema_context: str) -> Dict[str, Any]:
        """
        Attempts to extract intent (operation + columns + filters) using simple keywords/regex.
        Returns None if no intent could be confidently extracted.
        """
        self.logger.info(f"KeywordEngine fallback triggered for query: {query}")
        normalized = query.lower().strip()
        
        # 1. Operation Detection
        operation = "none"
        if re.search(r"\b(sum|total|add)\b", normalized):
            operation = "sum"
        elif re.search(r"\b(avg|average|mean)\b", normalized):
            operation = "avg"
        elif re.search(r"\b(count|number of|how many)\b", normalized):
            operation = "count"
        elif re.search(r"\b(show|list|display|get)\b", normalized):
            operation = "none" # generic select
        elif re.search(r"\b(min|minimum|lowest)\b", normalized):
            operation = "min"
        elif re.search(r"\b(max|maximum|highest)\b", normalized):
            operation = "max"
        
        # 2. Column Extraction (Simple substring match from schema)
        # We assume schema_context contains a string representation of columns.
        # A better approach (since we don't have the strict list passed in run usually)
        # is to rely on what is passed. But orchestration passes schema_context string.
        # We'll try to extract likely column names (alphanumeric/underscore) from query 
        # that ALSO appear in the schema_context string.
        
        extracted_columns = []
        # Find potential column-like words in query (2+ chars)
        # This is a heuristic.
        potential_cols = re.findall(r"\b[a-zA-Z0-9_]{2,}\b", normalized)
        
        # Check against schema_context (case-insensitive check)
        # schema_context is a big string, so this is "loose" matching.
        schema_lower = schema_context.lower()
        for word in potential_cols:
             if word in schema_lower:
                 # Logic to find the "Real" case-sensitive name would be hard without the list.
                 # For fallback, we might just pass the lowercase word and hope SQLEngine can handle fuzzy/lower
                 # OR we just explicitly assume the user typed it "close enough".
                 # Actually, SQLEngine logic has `existing_cols = [c for c in columns if c in filtered_df.columns]`
                 # So we need exact matches.
                 # Since we only get the generic schema string here, this is imperfect.
                 # IMPROVEMENT: Orchestrator should pass the actual DF columns to this engine?
                 # For now, let's just extract what we see.
                 extracted_columns.append(word)

        # 3. Simple Filters (where col = val)
        # Supports: where [col] is [val], [col] = [val], [col] > [val]
        filters = []
        # Pattern: (column) \s* (=|>|<|is) \s* (value)
        # Very basic.
        filter_pattern = r"(\w+)\s*(=|>|<|is)\s*([\w\.]+)"
        matches = re.findall(filter_pattern, normalized)
        for col, op, val in matches:
            if op == "is": op = "="
            filters.append({"column": col, "operator": op, "value": val})

        # If we found nothing useful, return None so we can refuse properly
        if operation == "none" and not extracted_columns and not filters:
            return None

        return {
            "operation": operation,
            "columns": list(set(extracted_columns)),
            "filters": filters,
            "group_by": [],
            "sort": [],
            "limit": 10, # fallback default
            "confidence": 0.5
        }

keyword_engine = KeywordEngine()
