from app.agents.base import BaseAgent
from app.engines.csv_engine import csv_engine
from app.engines.vector_engine import vector_engine
import pandas as pd
from typing import Any, Dict

class CSVRetrieverAgent(BaseAgent):
    """
    Executes the query plan produced by CSVReasoningAgent.
    Retrieves exact data from CSV using the deterministic CSV engine.
    """
    def run(self, input_data: dict) -> Any:
        """
        Input: {"intent": dict, "df": pd.DataFrame, "index_name": str, "engine_type": str}
        Output: Exact rows, numbers, or computed metrics.
        """
        intent = input_data.get("intent")
        df = input_data.get("df")
        index_name = input_data.get("index_name")
        engine_type = input_data.get("engine_type", "csv_engine")

        if engine_type == "csv_engine" and df is not None:
            return csv_engine.execute(df, intent)
        else:
            # Fallback to vector engine for semantic questions
            query = input_data.get("query")
            return vector_engine.search(query, index_name)
