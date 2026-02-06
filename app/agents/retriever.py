from typing import Any

from app.agents.base import BaseAgent
from app.engines.csv_engine import sql_engine
from app.engines.vector_engine import vector_engine


class CSVRetrieverAgent(BaseAgent):
    """
    Executes the query plan and retrieves exact data using the SQL Engine.
    """

    def run(self, input_data: dict) -> Any:
        """
        Input: {"intent": dict, "df": pd.DataFrame, "index_name": str, "engine_type": str}
        Output: Exact rows, numbers, or computed metrics.
        """
        intent = input_data.get("intent")
        df = input_data.get("df")
        index_name = input_data.get("index_name")
        engine_type = input_data.get("engine_type", "sql_engine")

        if engine_type in {"sql_engine", "csv_engine"} and df is not None:
            return sql_engine.execute(df, intent)

        query = input_data.get("query")
        return vector_engine.search(query, index_name)
