import logging
import re
import warnings
from typing import Any, Dict, List, Optional

import pandas as pd

# Initialize logger
logger = logging.getLogger(__name__)


class SQLEngine:
    """
    Deterministic SQL-like analytics engine built on Pandas DataFrames.
    """

    _DESC_DIRECTIONS = {"desc", "descending", "-1", "false"}

    def _normalize_column_token(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())

    def _resolve_column(self, df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
        if not requested:
            return None
        if requested in df.columns:
            return requested

        requested_norm = self._normalize_column_token(requested)
        for col in df.columns:
            if str(col).lower() == str(requested).lower():
                return col
        for col in df.columns:
            if self._normalize_column_token(col) == requested_norm:
                return col
        return None

    def _resolve_columns(self, df: pd.DataFrame, requested_columns: List[str]) -> List[str]:
        resolved = []
        for col in requested_columns or []:
            resolved_col = self._resolve_column(df, col)
            if resolved_col and resolved_col not in resolved:
                resolved.append(resolved_col)
        return resolved

    def _looks_like_year(self, value: Any) -> bool:
        s = str(value).strip()
        return bool(re.fullmatch(r"\d{4}", s))

    def _refusal_payload(self, summary: str) -> Dict[str, Any]:
        return {"relevant_rows": [{"_summary": summary, "should_ask_user": True}]}

    def _summary_rows_from_scalar(self, operation: str, scalar_result: Any, columns: List[str]) -> List[Dict[str, Any]]:
        if operation == "count":
            value = scalar_result if isinstance(scalar_result, (int, float)) else 0
            return [{"count": f"Count: {value}", "_summary": f"Count of rows: {value}"}]

        if isinstance(scalar_result, dict) and scalar_result:
            rows = []
            op_word = {
                "avg": "Average",
                "sum": "Sum",
                "min": "Minimum",
                "max": "Maximum",
            }.get(operation, operation.capitalize())
            for col, val in scalar_result.items():
                rows.append({col: f"{op_word}: {val}", "_summary": f"{op_word} of {col}: {val}"})
            return rows

        if operation in {"correlation", "corr"} and isinstance(scalar_result, dict):
            return [{"correlation": scalar_result, "_summary": "Correlation matrix computed."}]

        target = ", ".join(columns) if columns else "target column"
        op_word = {
            "avg": "Average",
            "sum": "Sum",
            "min": "Minimum",
            "max": "Maximum",
            "count": "Count",
        }.get(operation, operation.capitalize())
        return [{"_summary": f"{op_word} could not be computed for {target}.", "should_ask_user": True}]

    def _apply_single_filter(self, df: pd.DataFrame, filter_spec: Dict[str, Any]) -> pd.DataFrame:
        requested_col = filter_spec.get("column")
        op = str(filter_spec.get("operator", "=")).strip().lower()
        value = filter_spec.get("value")

        col = self._resolve_column(df, requested_col)
        if col is None:
            logger.warning(f"Filter column '{requested_col}' not found in dataframe")
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
                low = pd.to_numeric(str(bounds[0]).strip(), errors="coerce")
                high = pd.to_numeric(str(bounds[1]).strip(), errors="coerce")
                if pd.notna(low) and pd.notna(high) and numeric_series.notna().sum() > 0:
                    return df[(numeric_series >= low) & (numeric_series <= high)]
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

        # fallback for legacy behavior
        if op == "=":
            return df[text_series.str.contains(value_text, case=False, na=False)]
        if op == "!=":
            return df[~text_series.str.contains(value_text, case=False, na=False)]

        logger.warning(f"Unsupported filter operator '{op}' for column '{col}'")
        return df

    def _sort_dataframe(self, res_df: pd.DataFrame, sort: List[Dict[str, Any]]) -> pd.DataFrame:
        if res_df.empty or not sort:
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
        return sorted_df.drop(columns=drop_cols)

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

        if isinstance(filters, dict):
            logger.info("Converting dict-style filters to list-style")
            filters = [{"column": k, "operator": "=", "value": v} for k, v in filters.items()]

        dataframes = df_input if isinstance(df_input, dict) else {"default": df_input}
        logger.info(f"Processing {len(dataframes)} dataframe(s)")

        all_rows: List[Dict[str, Any]] = []

        for name, df in dataframes.items():
            logger.info(f"Processing sheet/dataframe: {name} (Initial rows: {len(df)})")
            filtered_df = df.copy()

            if filters:
                logger.info(f"Applying {len(filters)} filters")
                for filter_spec in filters:
                    filtered_df = self._apply_single_filter(filtered_df, filter_spec)
                logger.info(f"Rows after filtering: {len(filtered_df)}")

            if len(filtered_df) == 0:
                all_rows.append({"_summary": "No rows matched your filters.", "should_ask_user": True})
                if len(dataframes) > 1:
                    continue

            valid_columns = self._resolve_columns(filtered_df, columns)
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
                        grain_map = {"year": "Y", "month": "M", "quarter": "Q", "week": "W", "day": "D"}
                        freq = grain_map.get(str(time_grain).lower(), "D")
                        truncated_col = f"{resolved_col_name}_{time_grain}"
                        filtered_df[truncated_col] = filtered_df[resolved_col_name].dt.to_period(freq).astype(str)
                        final_groupers.append(truncated_col)
                    except Exception as e:
                        logger.error(f"Time conversion failed for {resolved_col_name}: {str(e)}")
                        final_groupers.append(resolved_col_name)
                else:
                    final_groupers.append(resolved_col_name)

            sheet_result_rows: List[Dict[str, Any]] = []
            res_df = None
            is_scalar = False
            scalar_result: Any = None

            if final_groupers:
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
                    else:
                        res_df = grouped.size().to_frame(name="count")
                    res_df = res_df.reset_index()
                except Exception as e:
                    logger.error(f"Grouping failed: {str(e)}")
                    sheet_result_rows = [{"_summary": f"Grouping failed: {str(e)}", "should_ask_user": True}]

            elif operation in {"filter", "none", "profile"}:
                res_df = filtered_df[valid_columns] if valid_columns else filtered_df
            else:
                is_scalar = True
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
                elif operation in {"correlation", "corr"}:
                    if len(valid_columns) >= 2:
                        numeric_df = filtered_df[valid_columns].apply(pd.to_numeric, errors="coerce")
                        scalar_result = numeric_df.corr().to_dict()
                    else:
                        scalar_result = {}

            if res_df is not None and not res_df.empty and having:
                for h in having:
                    res_df = self._apply_single_filter(res_df, h)

            if res_df is not None and not res_df.empty and sort:
                res_df = self._sort_dataframe(res_df, sort)

            if res_df is not None:
                sheet_result_rows = res_df.head(limit).to_dict("records")
            elif is_scalar:
                sheet_result_rows = self._summary_rows_from_scalar(operation, scalar_result, valid_columns)

            if len(dataframes) > 1:
                for row in sheet_result_rows:
                    if "sheet" not in row:
                        row["sheet"] = name

            all_rows.extend(sheet_result_rows)

        if not all_rows:
            return self._refusal_payload("I could not compute a result from the provided dataset and query plan.")

        return {"relevant_rows": all_rows}


sql_engine = SQLEngine()
# Backward-compatible aliases
CSVEngine = SQLEngine
csv_engine = sql_engine
