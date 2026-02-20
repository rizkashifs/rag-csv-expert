from typing import Any

from app.agents.base import BaseAgent
from app.engines.csv_engine import sql_engine
from app.engines.text_engine import text_engine


class CSVRetrieverAgent(BaseAgent):
    """
    Executes the query plan and retrieves exact data using the SQL Engine.
    """

    def run(self, input_data: dict) -> Any:
        """
        Input: {"intent": dict, "df": pd.DataFrame, "index_name": str, "engine_type": str, "query": str}
        Output: {"relevant_rows": [...]}
        """
        intent = input_data.get("intent")
        df = input_data.get("df")
        engine_type = input_data.get("engine_type", "sql_engine")

        if engine_type in {"sql_engine", "csv_engine"} and df is not None:
            return sql_engine.execute(df, intent)

        if engine_type in {"text_engine", "TEXT_TABLE_RAG"} and df is not None:
            return text_engine.execute(df, intent)

        # Fallback (should not normally be reached)
        query = input_data.get("query", "")
        if df is not None:
            return text_engine.execute(df, intent)
        return {"relevant_rows": []}
