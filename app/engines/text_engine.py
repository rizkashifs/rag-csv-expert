"""
TextEngine — Grep/Pandas-based semantic search engine.

Replaces the vector_engine for TEXT_TABLE_RAG queries.
No embeddings required. Uses keyword matching, normalised text search,
and id-based row scoping.  Output is identical to SQLEngine:
    {"relevant_rows": [{"col": val, ..., "_summary": str}]}
"""
import re
import warnings
from typing import Any, Dict, List, Optional

import pandas as pd

from app.agents.refusal import RefusalAgent
from app.utils.logger import logger


class TextEngine:
    """
    Deterministic text-search engine built on Pandas / regex.

    Reads a semantic_plan produced by the RouterAgent and:
      1. Resolves & pre-filters rows by id_filters (e.g. employee_id = 1234)
      2. Builds a searchable text column from target_text_columns
      3. Searches it using keywords / query_text
      4. Applies post_filters on the matched rows
      5. Returns {relevant_rows: [...]} matching the SQL engine contract
    """

    # ── column name helpers (mirrors SQLEngine) ──────────────────────────────

    def _normalize_token(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())

    def _resolve_column(self, df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
        if not requested:
            return None
        if requested in df.columns:
            return requested
        req_norm = self._normalize_token(requested)
        for col in df.columns:
            if str(col).lower() == str(requested).lower():
                return col
        for col in df.columns:
            if self._normalize_token(col) == req_norm:
                return col
        return None

    def _resolve_columns(self, df: pd.DataFrame, requested: List[str]) -> List[str]:
        resolved = []
        for r in requested or []:
            col = self._resolve_column(df, r)
            if col and col not in resolved:
                resolved.append(col)
        return resolved

    # ── date handling helpers ─────────────────────────────────────────────────

    def _robust_to_datetime(self, series: pd.Series) -> pd.Series:
        """
        Robustly convert a series to datetime handling mixed formats and YYYYMMDD integers.
        """
        if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
            sample = series.dropna().head(20)
            if not sample.empty and all(19000101 <= x <= 21001231 for x in sample if pd.notna(x)):
                str_series = series.astype(str).str.replace(r"\.0$", "", regex=True)
                return pd.to_datetime(str_series, format="%Y%m%d", errors="coerce")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")

    # ── refusal (same format as SQLEngine) ───────────────────────────────────

    def _refusal_payload(
        self,
        summary: str,
        schema_context: str = "",
        follow_up_questions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        agent = RefusalAgent()
        payload = agent.run(
            {
                "schema_context": schema_context,
                "route_schema": {"follow_up_questions": follow_up_questions or [summary]},
            }
        )
        logger.info(f"TextEngine refusal. summary={summary}")
        return payload

    # ── text normalisation ────────────────────────────────────────────────────

    @staticmethod
    def _normalise_series(series: pd.Series) -> pd.Series:
        """Lowercase, strip, collapse whitespace, fill NaN with empty string."""
        return (
            series.fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    # ── id / pre-filter application ───────────────────────────────────────────

    def _apply_id_filters(
        self,
        df: pd.DataFrame,
        id_filters: List[Dict[str, Any]],
        issues: List[str],
    ) -> pd.DataFrame:
        """
        Apply exact-match / equality id_filters BEFORE the text search.
        Format: [{"column": "employee_id", "value": "1234"}]
        """
        for f in id_filters:
            col = self._resolve_column(df, f.get("column"))
            val = f.get("value")
            if col is None:
                issues.append(f"ID filter column '{f.get('column')}' not found in dataset.")
                continue
            if val is None:
                continue

            text_series = df[col].astype(str).str.strip()
            numeric_val = pd.to_numeric(str(val), errors="coerce")
            numeric_series = pd.to_numeric(df[col], errors="coerce")

            if pd.notna(numeric_val) and numeric_series.notna().sum() > 0:
                df = df[numeric_series == numeric_val]
            else:
                df = df[text_series.str.lower() == str(val).strip().lower()]

            logger.info(
                f"ID filter applied. column={col}, value={val}, remaining_rows={len(df)}"
            )
        return df

    # ── post-filter application (mirrors SQLEngine._apply_single_filter) ──────

    def _apply_post_filter(
        self,
        df: pd.DataFrame,
        filter_spec: Dict[str, Any],
        issues: List[str],
    ) -> pd.DataFrame:
        col = self._resolve_column(df, filter_spec.get("column"))
        if col is None:
            issues.append(
                f"Post-filter column '{filter_spec.get('column')}' not found in dataset."
            )
            return df

        op = str(filter_spec.get("operator", "=")).strip().lower()
        value = filter_spec.get("value")
        value_text = "" if value is None else str(value).strip()

        series = df[col]
        text_series = series.astype(str).str.strip()
        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_value = pd.to_numeric(value_text, errors="coerce")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            datetime_series = self._robust_to_datetime(series)
            datetime_value = pd.to_datetime(value, errors="coerce", dayfirst=True) if value is not None else pd.NaT
        datetime_valid = datetime_series.notna().sum() > 0

        try:
            if op in {"=", "==", "eq"}:
                if pd.notna(numeric_value) and numeric_series.notna().sum() > 0:
                    return df[numeric_series == numeric_value]
                if datetime_valid:
                    # Year matching helper
                    if bool(re.fullmatch(r"\d{4}", value_text)):
                        return df[datetime_series.dt.year == int(value_text)]
                    
                    # Robust partial date matching
                    try:
                        p = pd.Period(value_text)
                        if p.freqstr.startswith("A"):
                            return df[datetime_series.dt.year == p.year]
                        if p.freqstr.startswith("Q"):
                            return df[(datetime_series.dt.year == p.year) & (datetime_series.dt.quarter == p.quarter)]
                        if p.freqstr.startswith("M"):
                            return df[(datetime_series.dt.year == p.year) & (datetime_series.dt.month == p.month)]
                    except Exception:
                        pass

                    if pd.notna(datetime_value):
                        return df[datetime_series == datetime_value]
                return df[text_series.str.lower() == value_text.lower()]

            if op in {"!=", "<>", "neq"}:
                if pd.notna(numeric_value) and numeric_series.notna().sum() > 0:
                    return df[numeric_series != numeric_value]
                if datetime_valid:
                    if bool(re.fullmatch(r"\d{4}", value_text)):
                        return df[datetime_series.dt.year != int(value_text)]
                    
                    try:
                        p = pd.Period(value_text)
                        if p.freqstr.startswith("A"):
                            return df[datetime_series.dt.year != p.year]
                        if p.freqstr.startswith("Q"):
                            return df[(datetime_series.dt.year != p.year) | (datetime_series.dt.quarter != p.quarter)]
                        if p.freqstr.startswith("M"):
                            return df[(datetime_series.dt.year != p.year) | (datetime_series.dt.month != p.month)]
                    except Exception:
                        pass

                    if pd.notna(datetime_value):
                        return df[datetime_series != datetime_value]
                return df[text_series.str.lower() != value_text.lower()]

            if op in {"contains", "like"}:
                return df[text_series.str.contains(value_text, case=False, na=False)]

            if op in {"not contains", "not_contains", "not like"}:
                return df[~text_series.str.contains(value_text, case=False, na=False)]

            if op in {"in", "not in", "not_in"}:
                raw = value if isinstance(value, list) else str(value).split(",")
                candidates = {str(v).strip().lower() for v in raw}
                mask = text_series.str.lower().isin(candidates)
                return df[~mask] if op in {"not in", "not_in"} else df[mask]

            if op in {">", "<", ">=", "<="}:
                if pd.notna(numeric_value) and numeric_series.notna().sum() > 0:
                    if op == ">":
                        return df[numeric_series > numeric_value]
                    if op == "<":
                        return df[numeric_series < numeric_value]
                    if op == ">=":
                        return df[numeric_series >= numeric_value]
                    return df[numeric_series <= numeric_value]

                if datetime_valid and pd.notna(datetime_value):
                    if op == ">":
                        return df[datetime_series > datetime_value]
                    if op == "<":
                        return df[datetime_series < datetime_value]
                    if op == ">=":
                        return df[datetime_series >= datetime_value]
                    return df[datetime_series <= datetime_value]

        except Exception as exc:
            issues.append(f"Post-filter on '{col}' failed: {exc}")

        logger.warning(f"Post-filter could not be applied. spec={filter_spec}")
        return df

    # ── keyword extraction ────────────────────────────────────────────────────

    @staticmethod
    def _extract_keywords(semantic_plan: Dict[str, Any]) -> List[str]:
        """
        Pull keywords from semantic_plan.keywords OR decompose query_text into tokens.
        """
        explicit = semantic_plan.get("keywords") or []
        if explicit:
            return [str(k).strip().lower() for k in explicit if str(k).strip()]

        query_text = semantic_plan.get("query_text", "")
        # Remove common stopwords and split into tokens (≥ 3 chars)
        stopwords = {
            "the", "and", "for", "with", "what", "who", "how", "are",
            "were", "was", "have", "has", "that", "this", "from", "where",
            "show", "find", "get", "list", "give", "tell", "all", "any",
            "can", "does", "did", "please", "me", "you", "of", "in", "on",
            "at", "by", "an", "a", "is", "to", "be", "do",
        }
        tokens = re.findall(r"\b[a-z0-9_-]{3,}\b", query_text.lower())
        return [t for t in tokens if t not in stopwords]

    # ── main execute ──────────────────────────────────────────────────────────

    def execute(self, df_input: Any, intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a text-based search plan on a DataFrame (or dict of DataFrames).

        intent is the full normalised route schema from RouterAgent.
        The semantic_plan sub-key drives all search behaviour.

        Returns: {"relevant_rows": [{"col": val, ..., "_summary": str}]}
        """
        # ── unpack dataframes ────────────────────────────────────────────────
        dataframes: Dict[str, pd.DataFrame] = (
            df_input if isinstance(df_input, dict) else {"default": df_input}
        )
        schema_context = "\n".join(
            [f"{name}: {', '.join(map(str, df.columns))}" for name, df in dataframes.items()]
        )

        semantic_plan: Dict[str, Any] = intent.get("semantic_plan") or {}
        query_text: str = semantic_plan.get("query_text", "")
        top_k: int = int(semantic_plan.get("top_k") or 8)
        semantic_intent: str = semantic_plan.get("semantic_intent", "row_matching")
        target_text_columns: List[str] = semantic_plan.get("target_text_columns") or []
        post_filters: List[Dict[str, Any]] = semantic_plan.get("post_filters") or []
        id_filters: List[Dict[str, Any]] = semantic_plan.get("id_filters") or []

        logger.info(
            f"TextEngine.execute. query_text={query_text!r}, "
            f"target_cols={target_text_columns}, id_filters={id_filters}, "
            f"post_filters={post_filters}, top_k={top_k}"
        )

        # ── basic validation ─────────────────────────────────────────────────
        if not query_text.strip() and not id_filters:
            return self._refusal_payload(
                "No search text or ID filters were provided.",
                schema_context=schema_context,
                follow_up_questions=[
                    "What keyword or phrase should I search for?",
                    "Or provide an ID (e.g. employee_id = 1234) to scope the search.",
                ],
            )

        keywords = self._extract_keywords(semantic_plan)
        logger.info(f"Extracted keywords: {keywords}")

        all_rows: List[Dict[str, Any]] = []
        processing_issues: List[str] = []

        for sheet_name, df in dataframes.items():
            logger.info(f"TextEngine processing sheet: {sheet_name} ({len(df)} rows)")

            # --- Virtual 'sheet' row scoping ---
            # Check both id_filters and post_filters for sheet-level scoping matches
            all_potential_sheet_filters = (id_filters or []) + (post_filters or [])
            sheet_filters = [f for f in all_potential_sheet_filters if str(f.get("column")).lower() == "sheet"]
            should_skip_sheet = False
            for sf in sheet_filters:
                val = str(sf.get("value", "")).strip().lower()
                op = str(sf.get("operator", "=")).strip().lower()
                if op in ("=", "==", "eq", "is"):
                    if val != sheet_name.lower():
                        should_skip_sheet = True
                elif op in ("!=", "<>", "neq", "is not"):
                    if val == sheet_name.lower():
                        should_skip_sheet = True
                elif op in ("contains", "like"):
                    if val not in sheet_name.lower():
                        should_skip_sheet = True
                
                if should_skip_sheet:
                    logger.info(f"TextEngine skipping sheet '{sheet_name}' due to sheet mismatch.")
                    break
            
            if should_skip_sheet:
                continue

            # Filter out virtual 'sheet' filters to avoid 'column not found' errors in resolved steps
            active_id_filters = [f for f in id_filters if str(f.get("column")).lower() != "sheet"]
            active_post_filters = [f for f in post_filters if str(f.get("column")).lower() != "sheet"]

            working_df = df.copy()

            # ── step 1: apply id_filters to scope rows ───────────────────────
            if active_id_filters:
                working_df = self._apply_id_filters(working_df, active_id_filters, processing_issues)
                logger.info(f"After id_filters: {len(working_df)} rows remain")

            if working_df.empty:
                # If we were searching for a specific ID and it's not here, it's a legitimate miss for this sheet
                continue

            # ── step 2: resolve text columns to search ───────────────────────
            resolved_text_cols = self._resolve_columns(working_df, target_text_columns)

            # Fallback: use all object/string columns if none resolved
            if not resolved_text_cols:
                resolved_text_cols = [
                    col
                    for col in working_df.columns
                    if working_df[col].dtype == object
                    or str(working_df[col].dtype) == "string"
                ]
                logger.info(
                    f"No target columns resolved; falling back to all text columns: {resolved_text_cols}"
                )

            if not resolved_text_cols:
                processing_issues.append(
                    "No searchable text columns could be found in this dataset."
                )
                continue

            # ── step 3: build a single combined searchable text column ────────
            searchable = self._normalise_series(
                working_df[resolved_text_cols]
                .fillna("")
                .apply(lambda row: " ".join(row.astype(str)), axis=1)
            )

            # ── step 4: keyword / query_text matching ─────────────────────────
            if keywords:
                # All keywords must appear somewhere in the combined text (AND logic)
                mask = pd.Series(True, index=working_df.index)
                for kw in keywords:
                    mask &= searchable.str.contains(re.escape(kw), case=False, na=False)

                matched_df = working_df[mask]

                # Fallback: ANY keyword match if strict AND returns nothing
                if matched_df.empty and len(keywords) > 1:
                    logger.info("AND keyword match empty; falling back to ANY keyword (OR) match")
                    any_mask = pd.Series(False, index=working_df.index)
                    for kw in keywords:
                        any_mask |= searchable.str.contains(re.escape(kw), case=False, na=False)
                    matched_df = working_df[any_mask]
            else:
                # No keywords extracted — return all scoped rows (id_filters already applied)
                matched_df = working_df

            logger.info(f"After text search: {len(matched_df)} rows matched")

            if matched_df.empty:
                processing_issues.append(
                    f"No rows matched the search terms {keywords!r} in sheet '{sheet_name}'."
                )
                continue

            # ── step 5: apply post_filters ────────────────────────────────────
            for pf in active_post_filters:
                matched_df = self._apply_post_filter(matched_df, pf, processing_issues)
                logger.info(f"After post-filter {pf}: {len(matched_df)} rows")

            if matched_df.empty:
                processing_issues.append(
                    "No results remained after applying post-filters."
                )
                continue

            # ── step 6: build rich row context ────────────────────────────────
            top_rows = matched_df.head(top_k)
            for _, row in top_rows.iterrows():
                row_dict = row.to_dict()

                # Build a human-readable summary sentence per row
                search_excerpts = []
                for col in resolved_text_cols:
                    val = str(row.get(col, "")).strip()
                    if val and val.lower() not in {"nan", "none", ""}:
                        search_excerpts.append(f"{col}: {val}")

                summary_text = " | ".join(search_excerpts) if search_excerpts else str(row_dict)
                row_dict["_summary"] = (
                    f"[{sheet_name}] Matched '{', '.join(keywords) or query_text}' — {summary_text}"
                )
                row_dict["_matched_columns"] = resolved_text_cols
                row_dict["_keywords"] = keywords

                if len(dataframes) > 1:
                    row_dict.setdefault("sheet", sheet_name)

                all_rows.append(row_dict)

        # ── handle complete failure ───────────────────────────────────────────
        if not all_rows:
            logger.warning(f"TextEngine produced no results. issues={processing_issues}")
            return self._refusal_payload(
                "No matching rows were found for your search.",
                schema_context=schema_context,
                follow_up_questions=processing_issues
                or [
                    f"No rows matched the keywords {keywords!r}. Try different terms?",
                    "Should I search in a specific column instead?",
                ],
            )

        logger.info(f"TextEngine returning {len(all_rows)} rows.")
        return {"relevant_rows": all_rows}


# Singleton
text_engine = TextEngine()
