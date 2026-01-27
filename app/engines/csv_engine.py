import pandas as pd
from typing import Dict, Any, List

class CSVEngine:
    """
    Deterministic data engine for executing queries on CSV data using Pandas.
    """
    def execute(self, df_input: Any, intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a query plan on the DataFrame(s).
        df_input can be a single DataFrame or a dict of {sheet_name: DataFrame}.
        """
        operation = intent.get("operation", "none").lower()
        columns = intent.get("columns", [])
        filters = intent.get("filters", {})
        
        # Normalize input to a dict of DataFrames
        dataframes = df_input if isinstance(df_input, dict) else {"default": df_input}
        
        all_results = []
        total_rows_filtered = 0

        for name, df in dataframes.items():
            # Apply filters
            filtered_df = df
            for col, val in filters.items():
                if col in df.columns:
                    filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(str(val), case=False, na=False)]
            
            if len(filtered_df) == 0 and len(filters) > 0:
                continue

            # Check if columns exist in this sheet
            existing_cols = [c for c in columns if c in filtered_df.columns]
            
            # Execute operation on this sheet
            sheet_result = None
            if operation == "sum":
                sheet_result = filtered_df[existing_cols].sum().to_dict() if existing_cols else {}
            elif operation == "avg":
                sheet_result = filtered_df[existing_cols].mean().to_dict() if existing_cols else {}
            elif operation == "count":
                sheet_result = len(filtered_df)
            elif operation == "filter" or operation == "none":
                sheet_result = filtered_df.head(10).to_dict('records')
            
            if sheet_result:
                all_results.append({"sheet": name, "result": sheet_result})
                total_rows_filtered += len(filtered_df)

        return {
            "data": all_results if len(all_results) > 1 else (all_results[0]["result"] if all_results else []),
            "row_count": total_rows_filtered
        }

# Singleton
csv_engine = CSVEngine()
