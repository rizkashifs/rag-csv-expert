import pandas as pd
from typing import Dict, Any, List

class CSVEngine:
    """
    Deterministic data engine for executing queries on CSV data using Pandas.
    """
    def execute(self, df: pd.DataFrame, intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a query plan on the DataFrame.
        """
        operation = intent.get("operation", "none").lower()
        columns = intent.get("columns", [])
        filters = intent.get("filters", {})
        group_by = intent.get("group_by", [])

        # Apply filters
        filtered_df = df
        for col, val in filters.items():
            if col in df.columns:
                filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(str(val), case=False, na=False)]

        # Execute operation
        result = None
        if operation == "sum":
            result = filtered_df[columns].sum().to_dict() if columns else filtered_df.sum().to_dict()
        elif operation == "avg":
            result = filtered_df[columns].mean().to_dict() if columns else filtered_df.mean().to_dict()
        elif operation == "count":
            result = len(filtered_df)
        elif operation == "max":
            result = filtered_df[columns].max().to_dict() if columns else filtered_df.max().to_dict()
        elif operation == "min":
            result = filtered_df[columns].min().to_dict() if columns else filtered_df.min().to_dict()
        elif operation == "filter":
            result = filtered_df.head(20).to_dict('records') # Return a sample
        else:
            result = filtered_df.head(5).to_dict('records')

        return {
            "data": result,
            "row_count": len(filtered_df)
        }

# Singleton
csv_engine = CSVEngine()
