import pandas as pd
from typing import Dict, Any


class SQLEngine:
    """
    Deterministic SQL-like analytics engine built on Pandas DataFrames.
    """

    def execute(self, df_input: Any, intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a query plan on the DataFrame(s).
        df_input can be a single DataFrame or a dict of {sheet_name: DataFrame}.
        """
        operation = intent.get("operation", "none").lower()
        columns = intent.get("columns", [])
        filters = intent.get("filters", [])

        # Backward compatibility: dict-style filters
        if isinstance(filters, dict):
            filters = [{"column": k, "operator": "=", "value": v} for k, v in filters.items()]

        dataframes = df_input if isinstance(df_input, dict) else {"default": df_input}
        all_results = []
        total_rows_filtered = 0

        for name, df in dataframes.items():
            filtered_df = df
            for f in filters:
                col = f.get("column")
                op = f.get("operator", "=")
                val = f.get("value")
                if col not in filtered_df.columns:
                    continue

                series = filtered_df[col]
                numeric_series = pd.to_numeric(series, errors="coerce")
                numeric_val = pd.to_numeric(str(val), errors="coerce")

                if op == "=":
                    filtered_df = filtered_df[series.astype(str).str.contains(str(val), case=False, na=False)]
                elif op == "!=":
                    filtered_df = filtered_df[~series.astype(str).str.contains(str(val), case=False, na=False)]
                elif op in {">", "<", ">=", "<="} and pd.notna(numeric_val):
                    if op == ">":
                        filtered_df = filtered_df[numeric_series > numeric_val]
                    elif op == "<":
                        filtered_df = filtered_df[numeric_series < numeric_val]
                    elif op == ">=":
                        filtered_df = filtered_df[numeric_series >= numeric_val]
                    elif op == "<=":
                        filtered_df = filtered_df[numeric_series <= numeric_val]

            if len(filtered_df) == 0 and len(filters) > 0:
                continue

            existing_cols = [c for c in columns if c in filtered_df.columns]
            sheet_result = None
            if operation == "sum":
                sheet_result = filtered_df[existing_cols].apply(pd.to_numeric, errors="coerce").sum().to_dict() if existing_cols else {}
            elif operation == "avg":
                sheet_result = filtered_df[existing_cols].apply(pd.to_numeric, errors="coerce").mean().to_dict() if existing_cols else {}
            elif operation == "count":
                sheet_result = len(filtered_df)
            elif operation in {"correlation", "corr"}:
                if len(existing_cols) >= 2:
                    numeric_df = filtered_df[existing_cols].apply(pd.to_numeric, errors='coerce')
                    sheet_result = numeric_df.corr().to_dict()
                else:
                    sheet_result = "Correlation requires at least two numeric columns."
            elif operation in {"filter", "none", "profile"}:
                sheet_result = filtered_df.head(10).to_dict('records')

            if sheet_result is not None:
                all_results.append({"sheet": name, "result": sheet_result})
                total_rows_filtered += len(filtered_df)

        return {
            "data": all_results if len(all_results) > 1 else (all_results[0]["result"] if all_results else []),
            "row_count": total_rows_filtered,
        }


sql_engine = SQLEngine()
# Backward-compatible alias
csv_engine = sql_engine
