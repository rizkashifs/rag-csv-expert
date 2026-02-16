import pandas as pd
import logging
from typing import Dict, Any

# Initialize logger
logger = logging.getLogger(__name__)


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
        group_by = intent.get("group_by", [])
        having = intent.get("having", [])
        sort = intent.get("sort", [])
        limit = intent.get("limit") or 10

        logger.info(f"Starting execution with operation: {operation}")
        logger.debug(f"Full intent: {intent}")

        # Backward compatibility: dict-style filters
        if isinstance(filters, dict):
            logger.info("Converting dict-style filters to list-style")
            filters = [{"column": k, "operator": "=", "value": v} for k, v in filters.items()]

        dataframes = df_input if isinstance(df_input, dict) else {"default": df_input}
        logger.info(f"Processing {len(dataframes)} dataframe(s)")
        
        all_results = []
        total_rows_filtered = 0

        for name, df in dataframes.items():
            logger.info(f"Processing sheet/dataframe: {name} (Initial rows: {len(df)})")
            filtered_df = df.copy()
            
            # 1. Filter
            if filters:
                logger.info(f"Applying {len(filters)} filters")
                for f in filters:
                    col = f.get("column")
                    op = f.get("operator", "=")
                    val = f.get("value")
                    if col not in filtered_df.columns:
                        logger.warning(f"Filter column '{col}' not found in dataframe")
                        continue

                    logger.debug(f"Applying filter: {col} {op} {val}")
                    series = filtered_df[col]
                    # Try numeric conversion for comparison
                    numeric_series = pd.to_numeric(series, errors="coerce")
                    numeric_val = pd.to_numeric(str(val), errors="coerce") if val is not None else None

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
                logger.info(f"Rows after filtering: {len(filtered_df)}")

            if len(filtered_df) == 0:
                 logger.info(f"No rows remaining for sheet '{name}' after filtering")
                 if len(dataframes) > 1: continue 

            # 2. Prepare for Aggregation/Selection
            valid_columns = [c for c in columns if c in filtered_df.columns]
            if columns and not valid_columns:
                logger.warning(f"None of the requested columns {columns} were found in sheet '{name}'")
            
            # Handle Group By with optional Time Grain
            # group_by can be list of strings OR list of dicts: [{"column": "Date", "time_grain": "year"}]
            final_groupers = []
            
            for g in group_by:
                col_name = None
                time_grain = None
                
                if isinstance(g, dict):
                    col_name = g.get("column")
                    time_grain = g.get("time_grain")
                else:
                    col_name = str(g)
                
                if col_name and col_name in filtered_df.columns:
                    if time_grain and time_grain.lower() not in ("none", "null"):
                        # Time Intelligence Logic
                        logger.info(f"Applying time grain '{time_grain}' to column '{col_name}'")
                        try:
                            # Convert to datetime if needed
                            if not pd.api.types.is_datetime64_any_dtype(filtered_df[col_name]):
                                filtered_df[col_name] = pd.to_datetime(filtered_df[col_name], errors='coerce')
                            
                            # Create truncated column
                            grain_map = {"year": "Y", "month": "M", "quarter": "Q", "week": "W", "day": "D"}
                            freq = grain_map.get(time_grain.lower(), "D")
                            
                            # Using to_period for easier display (e.g., 2023 for Year), but converting to str for grouping stability
                            truncated_col = f"{col_name}_{time_grain}"
                            filtered_df[truncated_col] = filtered_df[col_name].dt.to_period(freq).astype(str)
                            final_groupers.append(truncated_col)
                        except Exception as e:
                            logger.error(f"Time conversion failed for {col_name}: {str(e)}")
                            # Fallback to normal column if date conversions fail
                            final_groupers.append(col_name)
                    else:
                        final_groupers.append(col_name)
                else:
                    logger.warning(f"Group by column '{col_name}' not found in sheet '{name}'")

            sheet_result = None
            res_df = None
            is_scalar = False

            # 3. Grouping & Aggregation
            if final_groupers:
                logger.info(f"Performing aggregation: {operation} grouped by {final_groupers}")
                try:
                    grouped = filtered_df.groupby(final_groupers)
                    if operation == "sum":
                        res_df = grouped[valid_columns].sum(numeric_only=True)
                    elif operation == "avg":
                        res_df = grouped[valid_columns].mean(numeric_only=True)
                    elif operation == "max":
                        res_df = grouped[valid_columns].max(numeric_only=True)
                    elif operation == "min":
                        res_df = grouped[valid_columns].min(numeric_only=True)
                    else: # count or none/default for grouped
                        res_df = grouped.size().to_frame(name="count")
                    
                    res_df = res_df.reset_index()
                except Exception as e:
                    logger.error(f"Grouping failed: {str(e)}")
                    sheet_result = [{"error": f"Grouping failed: {str(e)}"}]
            
            elif operation in {"filter", "none", "profile"}:
                logger.info(f"Selecting columns: {valid_columns}")
                res_df = filtered_df[valid_columns] if valid_columns else filtered_df

            else:
                # Scalar Aggregations (Global)
                logger.info(f"Performing scalar aggregation: {operation}")
                is_scalar = True
                if operation == "sum":
                    sheet_result = filtered_df[valid_columns].apply(pd.to_numeric, errors="coerce").sum().to_dict() if valid_columns else {}
                elif operation == "avg":
                    sheet_result = filtered_df[valid_columns].apply(pd.to_numeric, errors="coerce").mean().to_dict() if valid_columns else {}
                elif operation == "count":
                    sheet_result = len(filtered_df)
                elif operation in {"correlation", "corr"}:
                    if len(valid_columns) >= 2:
                        numeric_df = filtered_df[valid_columns].apply(pd.to_numeric, errors='coerce')
                        sheet_result = numeric_df.corr().to_dict()
                    else:
                        sheet_result = "Correlation requires at least two numeric columns."

            # 4. HAVING (Post-Aggregation Filtering)
            if res_df is not None and not res_df.empty and having:
                logger.info(f"Applying {len(having)} HAVING filters")
                for h in having:
                    col = h.get("column")
                    op = h.get("operator", "=")
                    val = h.get("value")
                    
                    if col not in res_df.columns:
                        logger.warning(f"HAVING column '{col}' not found in results")
                        continue
                        
                    logger.debug(f"Applying HAVING filter: {col} {op} {val}")
                    series = res_df[col]
                    numeric_series = pd.to_numeric(series, errors="coerce")
                    numeric_val = pd.to_numeric(str(val), errors="coerce")
                    
                    if pd.notna(numeric_val):
                         if op == ">":
                            res_df = res_df[numeric_series > numeric_val]
                         elif op == "<":
                            res_df = res_df[numeric_series < numeric_val]
                         elif op == ">=":
                            res_df = res_df[numeric_series >= numeric_val]
                         elif op == "<=":
                            res_df = res_df[numeric_series <= numeric_val]
                         elif op == "=":
                            res_df = res_df[numeric_series == numeric_val]
                         elif op == "!=":
                             res_df = res_df[numeric_series != numeric_val]
                logger.info(f"Rows after HAVING filtering: {len(res_df)}")

            # 5. Sorting (only for DataFrame results)
            if res_df is not None and not res_df.empty and sort:
                logger.info(f"Applying sorting on columns: {[s['column'] for s in sort]}")
                sort_cols = [s["column"] for s in sort if s["column"] in res_df.columns]
                ascending = [s.get("direction", "asc") == "asc" for s in sort if s["column"] in res_df.columns]
                if sort_cols:
                    res_df = res_df.sort_values(by=sort_cols, ascending=ascending)
                    logger.debug("Sorting applied successfully")

            # 6. Limit (only for DataFrame results)
            if res_df is not None:
                if len(res_df) > limit:
                    logger.info(f"Limiting result from {len(res_df)} to {limit} rows")
                res_df = res_df.head(limit)
                sheet_result = res_df.to_dict('records')

            if sheet_result is not None:
                all_results.append({"sheet": name, "result": sheet_result})
                total_rows_filtered += len(filtered_df)
                if is_scalar:
                    logger.info(f"Scalar result for sheet '{name}': {sheet_result}")
                else:
                    logger.info(f"Found {len(sheet_result)} result rows for sheet '{name}'")

        logger.info(f"Execution complete. Total results: {len(all_results)}, Total rows processed: {total_rows_filtered}")
        return {
            "data": all_results if len(all_results) > 1 else (all_results[0]["result"] if all_results else []),
            "row_count": total_rows_filtered,
        }


sql_engine = SQLEngine()
# Backward-compatible alias
csv_engine = sql_engine
