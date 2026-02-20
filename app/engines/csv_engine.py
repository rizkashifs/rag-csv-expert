import re
import warnings
from typing import Any, Dict, List, Optional

import pandas as pd

from app.agents.refusal import RefusalAgent
from app.utils.logger import logger


class SQLEngine:
    """
    Deterministic SQL-like analytics engine built on Pandas DataFrames.
    """

    _DESC_DIRECTIONS = {"desc", "descending", "-1", "false"}

    def __init__(self) -> None:
        self.refusal_agent = RefusalAgent()

    def _normalize_column_token(self, value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]", "", str(value).strip().lower())
        logger.info(f"Normalizing column token. input={value}, normalized={normalized}")
        return normalized

    def _resolve_column(self, df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
        logger.info(f"Resolving column. requested={requested}, available_columns={list(df.columns)}")
        if not requested:
            logger.info("No requested column provided; resolved column is None")
            return None
        if requested in df.columns:
            logger.info(f"Column resolved by exact match: {requested}")
            return requested

        requested_norm = self._normalize_column_token(requested)
        for col in df.columns:
            if str(col).lower() == str(requested).lower():
                logger.info(f"Column resolved by case-insensitive match. requested={requested}, resolved={col}")
                return col
        for col in df.columns:
            if self._normalize_column_token(col) == requested_norm:
                logger.info(f"Column resolved by normalized token match. requested={requested}, resolved={col}")
                return col
        logger.info(f"Column could not be resolved. requested={requested}")
        return None

    def _resolve_columns(self, df: pd.DataFrame, requested_columns: List[str]) -> List[str]:
        logger.info(f"Resolving columns. requested_columns={requested_columns}")
        resolved = []
        for col in requested_columns or []:
            resolved_col = self._resolve_column(df, col)
            if resolved_col and resolved_col not in resolved:
                resolved.append(resolved_col)
        logger.info(f"Resolved columns result={resolved}")
        return resolved

    def _looks_like_year(self, value: Any) -> bool:
        s = str(value).strip()
        is_year = bool(re.fullmatch(r"\d{4}", s))
        logger.info(f"Checking if value looks like year. value={value}, is_year={is_year}")
        return is_year

    def _refusal_payload(
        self,
        summary: str,
        schema_context: str = "",
        follow_up_questions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        payload = self.refusal_agent.run(
            {
                "schema_context": schema_context,
                "route_schema": {"follow_up_questions": follow_up_questions or [summary]},
            }
        )
        logger.info(f"Creating refusal payload. summary={summary}, payload={payload}")
        return payload

    def _get_filter_description(self, filters: Optional[List[Dict[str, Any]]]) -> str:
        if not filters:
            return ""

        desc_parts = []
        for f in filters:
            col = f.get("column")
            op = str(f.get("operator", "=")).strip().lower()
            val = f.get("value")

            # Natural language mapping
            op_map = {
                ">=": "after or on" if "date" in str(col).lower() else "at least",
                ">": "after" if "date" in str(col).lower() else "greater than",
                "<": "before" if "date" in str(col).lower() else "less than",
                "<=": "before or on" if "date" in str(col).lower() else "at most",
                "=": "is",
                "==": "is",
                "contains": "containing",
                "like": "containing",
            }
            natural_op = op_map.get(op, op)
            desc_parts.append(f"{col} {natural_op} {val}")
        return " where " + " and ".join(desc_parts)

    def _summary_rows_from_scalar(
        self,
        operation: str,
        scalar_result: Any,
        columns: List[str],
        filters: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        logger.info(
            f"Building summary rows from scalar. operation={operation}, scalar_result={scalar_result}, columns={columns}, filters={filters}"
        )

        filter_desc = self._get_filter_description(filters)

        if operation == "profile" and isinstance(scalar_result, list):
            # If scalar_result is already a list of formatted stats rows, return them
            return scalar_result

        if operation == "count":
            value = scalar_result if isinstance(scalar_result, (int, float)) else 0
            return [{"count": f"Count: {value}", "_summary": f"Count of rows{filter_desc}: {value}"}]

        if isinstance(scalar_result, dict) and scalar_result:
            rows = []
            op_word = {
                "avg": "Average",
                "sum": "Sum",
                "min": "Minimum",
                "max": "Maximum",
            }.get(operation, operation.capitalize())
            for col, val in scalar_result.items():
                summary_text = f"{op_word} of {col}{filter_desc}: {val}"
                rows.append({col: summary_text, "_summary": summary_text})
            return rows

        if operation in {"correlation", "corr"} and isinstance(scalar_result, dict):
            return [{"correlation": scalar_result, "_summary": f"Correlation matrix computed{filter_desc}."}]

        target = ", ".join(columns) if columns else "target column"
        op_word = {
            "avg": "Average",
            "sum": "Sum",
            "min": "Minimum",
            "max": "Maximum",
            "count": "Count",
        }.get(operation, operation.capitalize())
        return [{"_summary": f"{op_word} could not be computed for {target}{filter_desc}.", "should_ask_user": True}]

    def _apply_single_filter(
        self,
        df: pd.DataFrame,
        filter_spec: Dict[str, Any],
        issues: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        logger.info(f"Applying single filter. filter_spec={filter_spec}, input_rows={len(df)}")
        requested_col = filter_spec.get("column")
        op = str(filter_spec.get("operator", "=")).strip().lower()
        value = filter_spec.get("value")

        col = self._resolve_column(df, requested_col)
        if col is None:
            logger.warning(f"Filter column '{requested_col}' not found in dataframe")
            if issues is not None:
                issues.append(f"Filter column '{requested_col}' was not found in the dataset.")
            return df

        series = df[col]
        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_value = pd.to_numeric(str(value), errors="coerce") if value is not None else None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            datetime_series = pd.to_datetime(series, errors="coerce")
            datetime_value = pd.to_datetime(value, errors="coerce") if value is not None else pd.NaT
        datetime_valid = datetime_series.notna().sum() > 0

        text_series = series.astype(str).str.strip()
        value_text = "" if value is None else str(value).strip()
        logger.info(
            f"Prepared filter context. resolved_column={col}, operator={op}, value={value}, numeric_value={numeric_value}"
        )

        # Handle null/empty checks
        if value_text.lower() in {"null", "none", "nan", "", "empty"}:
            if op in {"=", "==", "eq", "is"}:
                return df[series.isna() | (text_series == "")]
            if op in {"!=", "<>", "neq", "is not"}:
                return df[series.notna() & (text_series != "")]

        if op in {"=", "==", "eq"}:
            if pd.notna(numeric_value) and numeric_series.notna().sum() > 0:
                return df[numeric_series == numeric_value]
            if datetime_valid:
                if self._looks_like_year(value):
                    return df[datetime_series.dt.year == int(value_text)]
                if pd.notna(datetime_value):
                    return df[datetime_series == datetime_value]
            return df[text_series.str.lower() == value_text.lower()]

        if op in {"!=", "<>", "neq"}:
            if pd.notna(numeric_value) and numeric_series.notna().sum() > 0:
                return df[numeric_series != numeric_value]
            if datetime_valid:
                if self._looks_like_year(value):
                    return df[datetime_series.dt.year != int(value_text)]
                if pd.notna(datetime_value):
                    return df[datetime_series != datetime_value]
            return df[text_series.str.lower() != value_text.lower()]

        if op in {"contains", "like"}:
            return df[text_series.str.contains(value_text, case=False, na=False)]
        if op in {"not contains", "not_contains", "not like"}:
            return df[~text_series.str.contains(value_text, case=False, na=False)]

        if op in {"in", "not in", "not_in"}:
            raw_values = value if isinstance(value, list) else str(value).split(",")
            candidates = {str(v).strip().lower() for v in raw_values}
            mask = text_series.str.lower().isin(candidates)
            return df[~mask] if op in {"not in", "not_in"} else df[mask]

        if op == "between":
            bounds = value if isinstance(value, list) else str(value).split(",")
            if len(bounds) >= 2:
                low_val = str(bounds[0]).strip()
                high_val = str(bounds[1]).strip()

                # Try numeric first
                low_num = pd.to_numeric(low_val, errors="coerce")
                high_num = pd.to_numeric(high_val, errors="coerce")
                if pd.notna(low_num) and pd.notna(high_num) and numeric_series.notna().sum() > 0:
                    return df[(numeric_series >= low_num) & (numeric_series <= high_num)]

                # Try datetime second
                if datetime_valid:
                    low_dt = pd.to_datetime(low_val, errors="coerce")
                    high_dt = pd.to_datetime(high_val, errors="coerce")
                    if pd.notna(low_dt) and pd.notna(high_dt):
                        return df[(datetime_series >= low_dt) & (datetime_series <= high_dt)]

            if issues is not None:
                issues.append(
                    f"Filter '{requested_col} between {value}' could not be applied due to incompatible data types."
                )
            return df

        if op in {">", "<", ">=", "<="}:
            if pd.notna(numeric_value) and numeric_series.notna().sum() > 0:
                if op == ">":
                    return df[numeric_series > numeric_value]
                if op == "<":
                    return df[numeric_series < numeric_value]
                if op == ">=":
                    return df[numeric_series >= numeric_value]
                return df[numeric_series <= numeric_value]

            if datetime_valid:
                if self._looks_like_year(value):
                    years = datetime_series.dt.year
                    y = int(value_text)
                    if op == ">":
                        return df[years > y]
                    if op == "<":
                        return df[years < y]
                    if op == ">=":
                        return df[years >= y]
                    return df[years <= y]

                if pd.notna(datetime_value):
                    if op == ">":
                        return df[datetime_series > datetime_value]
                    if op == "<":
                        return df[datetime_series < datetime_value]
                    if op == ">=":
                        return df[datetime_series >= datetime_value]
                    return df[datetime_series <= datetime_value]

            if issues is not None:
                issues.append(
                    f"Filter '{requested_col} {op} {value}' could not be applied due to incompatible data type."
                )
            return df

        # fallback for legacy behavior
        if op == "=":
            return df[text_series.str.contains(value_text, case=False, na=False)]
        if op == "!=":
            return df[~text_series.str.contains(value_text, case=False, na=False)]

        logger.warning(f"Unsupported filter operator '{op}' for column '{col}'")
        if issues is not None:
            issues.append(f"Unsupported filter operator '{op}' for column '{requested_col}'.")
        return df

    def _sort_dataframe(self, res_df: pd.DataFrame, sort: List[Dict[str, Any]]) -> pd.DataFrame:
        logger.info(f"Sorting dataframe. input_rows={len(res_df)}, sort={sort}")
        if res_df.empty or not sort:
            logger.info("Skipping sort; dataframe is empty or sort spec missing")
            return res_df

        normalized_sort = []
        for sort_spec in sort:
            requested_col = sort_spec.get("column")
            resolved_col = self._resolve_column(res_df, requested_col)
            if not resolved_col:
                continue

            direction = str(sort_spec.get("direction", "asc")).strip().lower()
            ascending = direction not in self._DESC_DIRECTIONS
            normalized_sort.append((resolved_col, ascending))

        if not normalized_sort:
            logger.info("No valid sort columns resolved; returning unsorted dataframe")
            return res_df

        tmp_df = res_df.copy()
        sort_cols = []
        ascending_flags = []

        for idx, (col, ascending) in enumerate(normalized_sort):
            numeric_series = pd.to_numeric(tmp_df[col], errors="coerce")
            numeric_ratio = numeric_series.notna().mean()

            if numeric_ratio >= 0.7:
                sort_key = numeric_series
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    datetime_series = pd.to_datetime(tmp_df[col], errors="coerce")
                datetime_ratio = datetime_series.notna().mean()
                if datetime_ratio >= 0.7:
                    sort_key = datetime_series
                else:
                    sort_key = tmp_df[col].astype(str).str.lower()

            key_col = f"__sort_key_{idx}"
            tmp_df[key_col] = sort_key
            sort_cols.append(key_col)
            ascending_flags.append(ascending)

        sorted_df = tmp_df.sort_values(by=sort_cols, ascending=ascending_flags, kind="mergesort")
        drop_cols = [c for c in sorted_df.columns if c.startswith("__sort_key_")]
        result = sorted_df.drop(columns=drop_cols)
        logger.info(f"Sorting complete. output_rows={len(result)}, sort_columns={sort_cols}")
        return result

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
        logger.info(
            f"Execution intent payload. intent={intent}, columns={columns}, filters={filters}, "
            f"group_by={group_by}, having={having}, sort={sort}, limit={limit}"
        )

        if isinstance(filters, dict):
            logger.info("Converting dict-style filters to list-style")
            filters = [{"column": k, "operator": "=", "value": v} for k, v in filters.items()]

        dataframes = df_input if isinstance(df_input, dict) else {"default": df_input}
        logger.info(f"Processing {len(dataframes)} dataframe(s). dataframe_names={list(dataframes.keys())}")
        schema_context = "\n".join([f"{name}: {', '.join(map(str, df.columns))}" for name, df in dataframes.items()])

        all_rows: List[Dict[str, Any]] = []
        processing_issues: List[str] = []

        for name, df in dataframes.items():
            logger.info(f"Processing sheet/dataframe: {name} (Initial rows: {len(df)})")
            filtered_df = df.copy()
            logger.info(f"Created working copy of dataframe '{name}'. columns={list(filtered_df.columns)}")

            if filters:
                logger.info(f"Applying {len(filters)} filters")
                for filter_spec in filters:
                    filtered_df = self._apply_single_filter(filtered_df, filter_spec, processing_issues)
                    logger.info(f"Filter applied. filter_spec={filter_spec}, current_rows={len(filtered_df)}")
                logger.info(f"Rows after filtering: {len(filtered_df)}")

            if processing_issues:
                logger.info(f"Processing issues encountered; routing to refusal agent. issues={processing_issues}")
                return self._refusal_payload(
                    "I could not safely apply one or more filters in your request.",
                    schema_context=schema_context,
                    follow_up_questions=processing_issues,
                )

            if len(filtered_df) == 0:
                all_rows.append({"_summary": "No rows matched your filters.", "should_ask_user": True})
                if len(dataframes) > 1:
                    continue

            valid_columns = self._resolve_columns(filtered_df, columns)
            logger.info(f"Valid columns for operation on '{name}': {valid_columns}")
            if columns and not valid_columns:
                logger.warning(f"None of the requested columns {columns} were found in sheet '{name}'")

            final_groupers = []
            for g in group_by:
                col_name = g.get("column") if isinstance(g, dict) else str(g)
                time_grain = g.get("time_grain") if isinstance(g, dict) else None

                resolved_col_name = self._resolve_column(filtered_df, col_name)
                if not resolved_col_name:
                    logger.warning(f"Group by column '{col_name}' not found in sheet '{name}'")
                    continue

                if time_grain and str(time_grain).lower() not in ("none", "null"):
                    try:
                        if not pd.api.types.is_datetime64_any_dtype(filtered_df[resolved_col_name]):
                            filtered_df[resolved_col_name] = pd.to_datetime(filtered_df[resolved_col_name], errors="coerce")
                        
                        grain = str(time_grain).lower()
                        truncated_col = f"{resolved_col_name}_{grain}"
                        dt_series = filtered_df[resolved_col_name].dt
                        
                        if grain == "year":
                            filtered_df[truncated_col] = dt_series.year.astype(str)
                        elif grain == "month":
                            filtered_df[truncated_col] = dt_series.strftime("%Y-%m")
                        elif grain == "month_name":
                            filtered_df[truncated_col] = dt_series.month_name()
                        elif grain == "day":
                            filtered_df[truncated_col] = dt_series.strftime("%Y-%m-%d")
                        elif grain == "day_name":
                            filtered_df[truncated_col] = dt_series.day_name()
                        elif grain == "quarter":
                            filtered_df[truncated_col] = dt_series.to_period("Q").astype(str)
                        else:
                            grain_map = {"week": "W"}
                            freq = grain_map.get(grain, "D")
                            filtered_df[truncated_col] = dt_series.to_period(freq).astype(str)
                            
                        final_groupers.append(truncated_col)
                        logger.info(
                            f"Applied time grain grouping. source_column={resolved_col_name}, time_grain={time_grain}, "
                            f"derived_column={truncated_col}"
                        )
                    except Exception as e:
                        logger.error(f"Time conversion failed for {resolved_col_name}: {str(e)}")
                        final_groupers.append(resolved_col_name)
                else:
                    final_groupers.append(resolved_col_name)
            logger.info(f"Final groupers for '{name}': {final_groupers}")

            sheet_result_rows: List[Dict[str, Any]] = []
            res_df = None
            is_scalar = False
            scalar_result: Any = None

            if final_groupers:
                try:
                    logger.info(f"Executing grouped operation. operation={operation}, groupers={final_groupers}")
                    grouped = filtered_df.groupby(final_groupers)
                    if operation == "sum":
                        res_df = grouped[valid_columns].sum(numeric_only=True)
                    elif operation == "avg":
                        res_df = grouped[valid_columns].mean(numeric_only=True)
                    elif operation == "max":
                        res_df = grouped[valid_columns].max(numeric_only=True)
                    elif operation == "min":
                        res_df = grouped[valid_columns].min(numeric_only=True)
                    else:
                        res_df = grouped.size().to_frame(name="count")
                    res_df = res_df.reset_index()
                    logger.info(f"Grouped operation complete. result_rows={len(res_df)}")
                except Exception as e:
                    logger.error(f"Grouping failed: {str(e)}")
                    return self._refusal_payload(
                        "I could not complete the grouped calculation for this request.",
                        schema_context=schema_context,
                        follow_up_questions=[f"Grouping failed: {str(e)}"],
                    )

            elif operation in {"filter", "none"}:
                res_df = filtered_df[valid_columns] if valid_columns else filtered_df
                logger.info(f"Passthrough operation complete. result_rows={len(res_df)}")
            else:
                is_scalar = True
                logger.info(f"Executing scalar operation. operation={operation}, valid_columns={valid_columns}")
                if operation == "sum":
                    scalar_result = filtered_df[valid_columns].apply(pd.to_numeric, errors="coerce").sum().to_dict() if valid_columns else {}
                elif operation == "avg":
                    scalar_result = filtered_df[valid_columns].apply(pd.to_numeric, errors="coerce").mean().to_dict() if valid_columns else {}
                elif operation == "max":
                    scalar_result = filtered_df[valid_columns].apply(pd.to_numeric, errors="coerce").max().to_dict() if valid_columns else {}
                elif operation == "min":
                    scalar_result = filtered_df[valid_columns].apply(pd.to_numeric, errors="coerce").min().to_dict() if valid_columns else {}
                elif operation == "count":
                    scalar_result = len(filtered_df)
                elif operation == "profile":
                    logger.info(f"Profiling columns: {valid_columns}")
                    filter_desc = self._get_filter_description(filters)
                    stats_list = []
                    target_cols = valid_columns if valid_columns else list(filtered_df.columns)[:10]
                    for col in target_cols:
                        series = filtered_df[col]
                        non_null = series.count()
                        unique = series.nunique()
                        summary_parts = [f"Count: {len(series)}", f"Unique: {unique}"]
                        
                        # Try Numeric
                        num_series = pd.to_numeric(series, errors="coerce")
                        if num_series.notna().sum() > 0:
                            summary_parts.append(f"Min: {round(num_series.min(), 2)}")
                            summary_parts.append(f"Max: {round(num_series.max(), 2)}")
                            summary_parts.append(f"Avg: {round(num_series.mean(), 2)}")
                        else:
                            # Try Date
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", UserWarning)
                                dt_series = pd.to_datetime(series, errors="coerce")
                            if dt_series.notna().sum() > 0:
                                summary_parts.append(f"Min Date: {dt_series.min().strftime('%Y-%m-%d')}")
                                summary_parts.append(f"Max Date: {dt_series.max().strftime('%Y-%m-%d')}")
                        
                        stat_summary = " | ".join(summary_parts)
                        stats_list.append({
                            "column": col,
                            "stats": stat_summary,
                            "_summary": f"Summary for '{col}'{filter_desc}: {stat_summary}"
                        })
                    scalar_result = stats_list
                elif operation in {"correlation", "corr"}:
                    if len(valid_columns) >= 2:
                        numeric_df = filtered_df[valid_columns].apply(pd.to_numeric, errors="coerce")
                        scalar_result = numeric_df.corr().to_dict()
                    else:
                        scalar_result = {}
                logger.info(f"Scalar operation complete. scalar_result={scalar_result}")

            if res_df is not None and not res_df.empty and having:
                logger.info(f"Applying HAVING filters. having={having}")
                for h in having:
                    res_df = self._apply_single_filter(res_df, h, processing_issues)
                    logger.info(f"Applied HAVING filter. filter={h}, current_rows={len(res_df)}")

            if res_df is not None and not res_df.empty and sort:
                res_df = self._sort_dataframe(res_df, sort)

            if res_df is not None:
                sheet_result_rows = res_df.head(limit).to_dict("records")
                logger.info(f"Materialized tabular result rows for '{name}'. row_count={len(sheet_result_rows)}")
            elif is_scalar:
                sheet_result_rows = self._summary_rows_from_scalar(operation, scalar_result, valid_columns, filters=filters)
                logger.info(f"Materialized scalar summary rows for '{name}'. rows={sheet_result_rows}")

            if len(dataframes) > 1:
                for row in sheet_result_rows:
                    if "sheet" not in row:
                        row["sheet"] = name

            all_rows.extend(sheet_result_rows)
            logger.info(f"Accumulated result rows count={len(all_rows)} after processing '{name}'")

        if not all_rows:
            logger.info("Execution produced no rows. Returning refusal payload")
            return self._refusal_payload("I could not compute a result from the provided dataset and query plan.", schema_context=schema_context)

        payload = {"relevant_rows": all_rows}
        logger.info(f"Execution complete. Returning payload: {payload}")
        return payload


sql_engine = SQLEngine()
# Backward-compatible aliases
CSVEngine = SQLEngine
csv_engine = sql_engine
