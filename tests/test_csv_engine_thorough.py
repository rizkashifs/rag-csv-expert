"""
Comprehensive real-world test suite for SQLEngine (csv_engine.py).
Covers all critical operations, edge cases, and real-world CSV/Excel field types
that a world-class SQL agent must handle correctly.
"""

import pandas as pd
import pytest

from app.engines.csv_engine import sql_engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rows(result):
    return result["relevant_rows"]


def _first(result):
    return result["relevant_rows"][0]


def _is_refusal(result):
    row = result["relevant_rows"][0]
    return row.get("should_ask_user") is True


# ===========================================================================
# 1. FILTER OPERATORS
# ===========================================================================

class TestFilterOperators:

    def test_eq_numeric(self):
        df = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "id", "operator": "=", "value": 2}]})
        assert len(_rows(r)) == 1 and _rows(r)[0]["id"] == 2

    def test_eq_text_case_insensitive(self):
        df = pd.DataFrame({"region": ["North", "South", "East"]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "region", "operator": "=", "value": "north"}]})
        assert len(_rows(r)) == 1 and _rows(r)[0]["region"] == "North"

    def test_neq_operator(self):
        df = pd.DataFrame({"status": ["active", "inactive", "active"]})
        r = sql_engine.execute(df, {"operation": "count", "filters": [{"column": "status", "operator": "!=", "value": "inactive"}]})
        assert "2" in _first(r)["count"]

    def test_gt_numeric(self):
        df = pd.DataFrame({"score": [50, 70, 90, 30]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "score", "operator": ">", "value": 60}]})
        assert len(_rows(r)) == 2

    def test_gte_numeric(self):
        df = pd.DataFrame({"score": [50, 70, 90]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "score", "operator": ">=", "value": 70}]})
        assert len(_rows(r)) == 2

    def test_lt_and_lte(self):
        df = pd.DataFrame({"price": [5.0, 10.0, 15.0, 20.0]})
        r_lt = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "price", "operator": "<", "value": 15}]})
        r_lte = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "price", "operator": "<=", "value": 15}]})
        assert len(_rows(r_lt)) == 2
        assert len(_rows(r_lte)) == 3

    def test_contains_text(self):
        df = pd.DataFrame({"genre": ["Comedy, Drama", "Action", "Comedy"]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "genre", "operator": "contains", "value": "Comedy"}]})
        assert len(_rows(r)) == 2

    def test_not_contains(self):
        df = pd.DataFrame({"genre": ["Comedy, Drama", "Action", "Thriller"]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "genre", "operator": "not_contains", "value": "Comedy"}]})
        assert len(_rows(r)) == 2

    def test_like_sql_wildcard_prefix(self):
        df = pd.DataFrame({"name": ["Alice", "Albert", "Bob", "Alicia"]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "name", "operator": "like", "value": "Al%"}]})
        assert len(_rows(r)) == 3

    def test_like_sql_wildcard_suffix(self):
        # Alice, Price, Grace all end in "ce"; Bob does not
        df = pd.DataFrame({"name": ["Alice", "Price", "Grace", "Bob"]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "name", "operator": "like", "value": "%ce"}]})
        assert len(_rows(r)) == 3

    def test_not_like(self):
        # USA and UK start with U; Germany and France do not
        df = pd.DataFrame({"country": ["USA", "UK", "Germany", "France"]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "country", "operator": "not like", "value": "U%"}]})
        assert len(_rows(r)) == 2

    def test_in_operator_text(self):
        df = pd.DataFrame({"region": ["North", "South", "East", "West"]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "region", "operator": "in", "value": ["North", "West"]}]})
        assert len(_rows(r)) == 2

    def test_not_in_operator(self):
        df = pd.DataFrame({"region": ["North", "South", "East", "West"]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "region", "operator": "not in", "value": ["North", "West"]}]})
        assert len(_rows(r)) == 2

    def test_in_operator_comma_string(self):
        df = pd.DataFrame({"category": ["A", "B", "C", "D"]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "category", "operator": "in", "value": "A,C"}]})
        assert len(_rows(r)) == 2

    def test_between_numeric(self):
        df = pd.DataFrame({"salary": [30000, 50000, 70000, 90000]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "salary", "operator": "between", "value": [40000, 75000]}]})
        assert len(_rows(r)) == 2

    def test_between_dates(self):
        df = pd.DataFrame({"date": ["2023-01-01", "2023-06-15", "2024-01-01", "2024-12-31"]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "date", "operator": "between", "value": ["2023-01-01", "2023-12-31"]}]})
        assert len(_rows(r)) == 2

    def test_null_filter_is(self):
        df = pd.DataFrame({"notes": ["has notes", None, "more notes", None]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "notes", "operator": "=", "value": "null"}]})
        assert len(_rows(r)) == 2

    def test_null_filter_is_not(self):
        df = pd.DataFrame({"notes": ["has notes", None, "more notes", None]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "notes", "operator": "!=", "value": "null"}]})
        assert len(_rows(r)) == 2

    def test_chained_filters_and_logic(self):
        df = pd.DataFrame({
            "region": ["North", "North", "South", "South"],
            "sales": [100, 500, 200, 800],
        })
        r = sql_engine.execute(df, {
            "operation": "none",
            "filters": [
                {"column": "region", "operator": "=", "value": "North"},
                {"column": "sales", "operator": ">", "value": 300},
            ],
        })
        assert len(_rows(r)) == 1 and _rows(r)[0]["sales"] == 500

    def test_filter_nonexistent_column_returns_refusal(self):
        """Filter on a missing column cannot be applied — engine should return a refusal."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        r = sql_engine.execute(df, {"operation": "count", "filters": [{"column": "DOES_NOT_EXIST", "operator": "=", "value": 1}]})
        # Column not found → issue reported → refusal payload
        assert _is_refusal(r)


# ===========================================================================
# 2. DATE / TIME FILTERING
# ===========================================================================

class TestDateFilters:

    def test_year_equality_filter(self):
        df = pd.DataFrame({"date": ["2022-03-01", "2023-07-15", "2023-11-30", "2024-01-01"]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "date", "operator": "=", "value": "2023"}]})
        assert len(_rows(r)) == 2

    def test_year_gte_filter(self):
        df = pd.DataFrame({"date": ["2021-01-01", "2022-06-01", "2023-12-31", "2024-01-01"]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "date", "operator": ">=", "value": "2023"}]})
        assert len(_rows(r)) == 2

    def test_quarter_filter(self):
        # pandas Period format for quarters is "2023Q1" (not "Q1 2023")
        df = pd.DataFrame({"date": ["2023-01-15", "2023-04-20", "2023-07-01", "2023-10-10"]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "date", "operator": "=", "value": "2023Q1"}]})
        assert len(_rows(r)) == 1 and _rows(r)[0]["date"] == "2023-01-15"

    def test_month_filter(self):
        # Use day > 12 to avoid dayfirst ambiguity (e.g. "2023-03-15" is unambiguously March 15)
        df = pd.DataFrame({"date": ["2023-03-15", "2023-03-25", "2023-04-14", "2023-05-20"]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "date", "operator": "=", "value": "2023-03"}]})
        assert len(_rows(r)) == 2

    def test_yyyymmdd_integer_date(self):
        df = pd.DataFrame({
            "date_int": [20230101, 20230601, 20240101],
            "amount": [100, 200, 300],
        })
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "date_int", "operator": "=", "value": "2023"}]})
        assert len(_rows(r)) == 2

    def test_dayfirst_date_format(self):
        df = pd.DataFrame({"date": ["01/03/2023", "15/03/2023", "01/04/2023"]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "date", "operator": "=", "value": "2023-03"}]})
        assert len(_rows(r)) == 2

    def test_group_by_year_grain(self):
        df = pd.DataFrame({
            "date": ["2023-01-15", "2023-06-20", "2024-03-01"],
            "sales": [100, 200, 300],
        })
        r = sql_engine.execute(df, {
            "operation": "sum",
            "columns": ["sales"],
            "group_by": [{"column": "date", "time_grain": "year"}],
        })
        rows = _rows(r)
        assert len(rows) == 2
        year_2023 = next(row for row in rows if "2023" in str(row.values()))
        assert year_2023["sum_sales"] == 300.0

    def test_group_by_month_grain(self):
        # Use day > 12 to avoid dayfirst ambiguity in date parsing
        df = pd.DataFrame({
            "date": ["2023-01-15", "2023-01-20", "2023-02-14"],
            "sales": [50, 100, 200],
        })
        r = sql_engine.execute(df, {
            "operation": "sum",
            "columns": ["sales"],
            "group_by": [{"column": "date", "time_grain": "month"}],
        })
        rows = _rows(r)
        assert len(rows) == 2
        jan = next(row for row in rows if "2023-01" in str(row.values()))
        assert jan["sum_sales"] == 150.0

    def test_group_by_quarter_grain(self):
        # Use day > 12 to avoid dayfirst ambiguity
        df = pd.DataFrame({
            "date": ["2023-01-15", "2023-04-15", "2023-07-15", "2023-10-15"],
            "revenue": [1000, 2000, 3000, 4000],
        })
        r = sql_engine.execute(df, {
            "operation": "sum",
            "columns": ["revenue"],
            "group_by": [{"column": "date", "time_grain": "quarter"}],
        })
        assert len(_rows(r)) == 4


# ===========================================================================
# 3. AGGREGATION OPERATIONS
# ===========================================================================

class TestAggregations:

    def test_sum(self):
        df = pd.DataFrame({"amount": [100, 200, 300]})
        r = sql_engine.execute(df, {"operation": "sum", "columns": ["amount"]})
        assert "600" in _first(r)["amount"]

    def test_avg(self):
        df = pd.DataFrame({"score": [80, 90, 70]})
        r = sql_engine.execute(df, {"operation": "avg", "columns": ["score"]})
        assert "80" in _first(r)["score"]

    def test_count_no_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        r = sql_engine.execute(df, {"operation": "count"})
        assert "5" in _first(r)["count"]

    def test_max_returns_context_rows(self):
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "salary": [50000, 90000, 70000],
        })
        r = sql_engine.execute(df, {"operation": "max", "columns": ["name", "salary"]})
        rows = _rows(r)
        assert any(row.get("name") == "Bob" for row in rows)

    def test_min_returns_context_rows(self):
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "salary": [50000, 90000, 70000],
        })
        r = sql_engine.execute(df, {"operation": "min", "columns": ["name", "salary"]})
        rows = _rows(r)
        assert any(row.get("name") == "Alice" for row in rows)

    def test_median(self):
        df = pd.DataFrame({"val": [1, 3, 5, 7, 9]})
        r = sql_engine.execute(df, {"operation": "median", "columns": ["val"]})
        assert "5" in _first(r)["val"]

    def test_mode(self):
        df = pd.DataFrame({"val": [1, 2, 2, 3, 3, 3]})
        r = sql_engine.execute(df, {"operation": "mode", "columns": ["val"]})
        assert "3" in _first(r)["val"]

    def test_std(self):
        df = pd.DataFrame({"val": [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]})
        r = sql_engine.execute(df, {"operation": "std", "columns": ["val"]})
        assert "Standard Deviation" in _first(r)["val"]

    def test_variance(self):
        df = pd.DataFrame({"val": [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]})
        r = sql_engine.execute(df, {"operation": "variance", "columns": ["val"]})
        assert "Variance" in _first(r)["val"]

    def test_quantile_75th(self):
        df = pd.DataFrame({"salary": [1000, 2000, 3000, 4000]})
        r = sql_engine.execute(df, {"operation": "quantile", "columns": ["salary"], "percentile": 75})
        assert "Quantile" in _first(r)["salary"]

    def test_quantile_as_decimal(self):
        df = pd.DataFrame({"salary": [1000, 2000, 3000, 4000]})
        r = sql_engine.execute(df, {"operation": "quantile", "columns": ["salary"], "percentile": 0.5})
        assert "Quantile" in _first(r)["salary"]

    def test_distinct_count(self):
        df = pd.DataFrame({"color": ["red", "blue", "red", "green", "blue"]})
        r = sql_engine.execute(df, {"operation": "distinct_count", "columns": ["color"]})
        assert "3" in _first(r)["color"]

    def test_null_count(self):
        df = pd.DataFrame({"val": [1, None, 3, None, 5]})
        r = sql_engine.execute(df, {"operation": "null_count", "columns": ["val"]})
        assert "2" in _first(r)["val"]

    def test_null_pct(self):
        df = pd.DataFrame({"val": [1, None, 3, None]})
        r = sql_engine.execute(df, {"operation": "null_pct", "columns": ["val"]})
        assert "50" in _first(r)["val"]

    def test_correlation(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
        r = sql_engine.execute(df, {"operation": "correlation", "columns": ["x", "y"]})
        assert "correlation" in _first(r)
        corr_matrix = _first(r)["correlation"]
        assert abs(corr_matrix["x"]["y"] - 1.0) < 1e-6

    def test_correlation_single_column_fails_gracefully(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        r = sql_engine.execute(df, {"operation": "correlation", "columns": ["x"]})
        assert _is_refusal(r)

    def test_value_counts(self):
        df = pd.DataFrame({"category": ["A", "B", "A", "C", "B", "A"]})
        r = sql_engine.execute(df, {"operation": "value_counts", "columns": ["category"]})
        rows = _rows(r)
        a_row = next(row for row in rows if row.get("value") == "A")
        assert a_row["count"] == 3

    def test_value_counts_respects_top_k(self):
        df = pd.DataFrame({"cat": list("AABBBBCCCCDDDDDDEE")})
        r = sql_engine.execute(df, {"operation": "value_counts", "columns": ["cat"], "top_k": 3})
        assert len(_rows(r)) == 3

    def test_histogram_bins(self):
        df = pd.DataFrame({"val": list(range(100))})
        r = sql_engine.execute(df, {"operation": "histogram", "columns": ["val"], "bins": 5})
        assert len(_rows(r)) == 5
        assert all("bin" in row and "count" in row for row in _rows(r))

    def test_profile(self):
        df = pd.DataFrame({"salary": [50000, 60000, 70000], "name": ["Alice", "Bob", "Charlie"]})
        r = sql_engine.execute(df, {"operation": "profile", "columns": ["salary", "name"]})
        assert len(_rows(r)) == 2
        salary_row = next(row for row in _rows(r) if row["column"] == "salary")
        assert "Avg:" in salary_row["stats"]

    def test_sum_on_text_column_is_refusal(self):
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
        r = sql_engine.execute(df, {"operation": "sum", "columns": ["name"]})
        assert _is_refusal(r)

    def test_operation_alias_mean(self):
        df = pd.DataFrame({"val": [10, 20, 30]})
        r = sql_engine.execute(df, {"operation": "mean", "columns": ["val"]})
        assert "Average" in _first(r)["val"]

    def test_operation_alias_describe(self):
        df = pd.DataFrame({"val": [10, 20, 30]})
        r = sql_engine.execute(df, {"operation": "describe", "columns": ["val"]})
        assert "stats" in _first(r)

    def test_multi_operation_pipe_syntax(self):
        df = pd.DataFrame({"val": [10, 20, 30]})
        r = sql_engine.execute(df, {"operation": "avg|sum", "columns": ["val"]})
        summaries = " ".join(row.get("_summary", "") for row in _rows(r))
        assert "Average" in summaries and "Sum" in summaries


# ===========================================================================
# 4. GROUP BY + HAVING
# ===========================================================================

class TestGroupByAndHaving:

    def test_group_by_sum(self):
        df = pd.DataFrame({
            "region": ["North", "North", "South", "South"],
            "sales": [100, 200, 300, 400],
        })
        r = sql_engine.execute(df, {
            "operation": "sum",
            "columns": ["sales"],
            "group_by": ["region"],
        })
        rows = _rows(r)
        north = next(row for row in rows if row["region"] == "North")
        assert north["sum_sales"] == 300.0

    def test_group_by_avg(self):
        df = pd.DataFrame({
            "dept": ["Eng", "Eng", "HR", "HR"],
            "salary": [90000, 110000, 60000, 80000],
        })
        r = sql_engine.execute(df, {
            "operation": "avg",
            "columns": ["salary"],
            "group_by": ["dept"],
        })
        rows = _rows(r)
        eng = next(row for row in rows if row["dept"] == "Eng")
        assert eng["avg_salary"] == 100000.0

    def test_group_by_count(self):
        df = pd.DataFrame({
            "category": ["A", "A", "B", "B", "B"],
            "val": [1, 2, 3, 4, 5],
        })
        r = sql_engine.execute(df, {"operation": "count", "group_by": ["category"]})
        rows = _rows(r)
        b_row = next(row for row in rows if row["category"] == "B")
        assert b_row["count"] == 3

    def test_group_by_multi_column(self):
        df = pd.DataFrame({
            "region": ["North", "North", "South", "South"],
            "product": ["A", "B", "A", "B"],
            "sales": [100, 200, 300, 400],
        })
        r = sql_engine.execute(df, {
            "operation": "sum",
            "columns": ["sales"],
            "group_by": ["region", "product"],
        })
        assert len(_rows(r)) == 4

    def test_having_filters_grouped_results(self):
        """Critical: HAVING must work on renamed aggregated columns (e.g., sum_Sales)."""
        df = pd.DataFrame({
            "region": ["North", "North", "South", "South", "East"],
            "sales": [100, 200, 150, 400, 300],
        })
        r = sql_engine.execute(df, {
            "operation": "sum",
            "columns": ["sales"],
            "group_by": ["region"],
            "having": [{"column": "sales", "operator": ">", "value": "400"}],
        })
        # North=300, South=550, East=300 → only South > 400
        rows = _rows(r)
        assert len(rows) == 1
        assert rows[0]["region"] == "South"

    def test_having_on_count(self):
        df = pd.DataFrame({
            "category": ["A", "A", "B", "B", "B", "C"],
        })
        r = sql_engine.execute(df, {
            "operation": "count",
            "group_by": ["category"],
            "having": [{"column": "count", "operator": ">=", "value": "2"}],
        })
        rows = _rows(r)
        # A=2, B=3, C=1 → A and B pass
        assert len(rows) == 2

    def test_group_by_distinct_count(self):
        df = pd.DataFrame({
            "dept": ["Eng", "Eng", "HR", "HR"],
            "project": ["P1", "P1", "P2", "P3"],
        })
        r = sql_engine.execute(df, {
            "operation": "distinct_count",
            "columns": ["project"],
            "group_by": ["dept"],
        })
        rows = _rows(r)
        hr = next(row for row in rows if row["dept"] == "HR")
        assert hr["distinct_count_project"] == 2

    def test_group_by_multi_aggregation(self):
        df = pd.DataFrame({
            "region": ["North", "North", "South"],
            "sales": [100, 200, 300],
            "cost": [50, 80, 120],
        })
        r = sql_engine.execute(df, {
            "operation": "sum",
            "columns": ["sales"],
            "group_by": ["region"],
            "aggregations": [
                {"function": "sum", "column": "sales"},
                {"function": "avg", "column": "cost"},
            ],
        })
        rows = _rows(r)
        assert len(rows) == 2
        north = next(row for row in rows if row["region"] == "North")
        assert "sum_sales" in north
        assert "avg_cost" in north

    def test_group_by_with_filter_and_sort(self):
        df = pd.DataFrame({
            "region": ["North", "North", "South", "South", "East"],
            "product": ["X", "Y", "X", "Y", "X"],
            "sales": [100, 200, 300, 50, 400],
        })
        r = sql_engine.execute(df, {
            "operation": "sum",
            "columns": ["sales"],
            "filters": [{"column": "product", "operator": "=", "value": "X"}],
            "group_by": ["region"],
            "sort": [{"column": "sales", "direction": "desc"}],
        })
        rows = _rows(r)
        # X: North=100, South=300, East=400 → sorted desc by sales
        assert rows[0]["region"] == "East"


# ===========================================================================
# 5. SORT + LIMIT
# ===========================================================================

class TestSortAndLimit:

    def test_sort_numeric_desc(self):
        df = pd.DataFrame({"val": [3, 1, 4, 1, 5, 9, 2, 6]})
        r = sql_engine.execute(df, {"operation": "none", "sort": [{"column": "val", "direction": "desc"}], "limit": 3})
        assert _rows(r)[0]["val"] == 9

    def test_sort_numeric_asc(self):
        df = pd.DataFrame({"val": [3, 1, 4, 1, 5]})
        r = sql_engine.execute(df, {"operation": "none", "sort": [{"column": "val", "direction": "asc"}]})
        assert _rows(r)[0]["val"] == 1

    def test_sort_text_alphabetical(self):
        df = pd.DataFrame({"name": ["Charlie", "Alice", "Bob"]})
        r = sql_engine.execute(df, {"operation": "none", "sort": [{"column": "name", "direction": "asc"}]})
        assert _rows(r)[0]["name"] == "Alice"

    def test_sort_numeric_stored_as_text(self):
        df = pd.DataFrame({"revenue": ["100", "9", "75", "1000"]})
        r = sql_engine.execute(df, {"operation": "none", "sort": [{"column": "revenue", "direction": "desc"}]})
        assert int(_rows(r)[0]["revenue"]) == 1000

    def test_sort_dates(self):
        df = pd.DataFrame({"date": ["2022-12-31", "2023-01-01", "2021-06-15"]})
        r = sql_engine.execute(df, {"operation": "none", "sort": [{"column": "date", "direction": "desc"}]})
        assert _rows(r)[0]["date"] == "2023-01-01"

    def test_sort_direction_descending_synonym(self):
        df = pd.DataFrame({"val": [1, 2, 3]})
        r = sql_engine.execute(df, {"operation": "none", "sort": [{"column": "val", "direction": "DESCENDING"}]})
        assert _rows(r)[0]["val"] == 3

    def test_limit_enforced(self):
        df = pd.DataFrame({"val": list(range(100))})
        r = sql_engine.execute(df, {"operation": "none", "limit": 5})
        assert len(_rows(r)) == 5

    def test_multi_column_sort(self):
        df = pd.DataFrame({
            "region": ["North", "South", "North", "South"],
            "sales": [200, 100, 100, 300],
        })
        r = sql_engine.execute(df, {
            "operation": "none",
            "sort": [
                {"column": "region", "direction": "asc"},
                {"column": "sales", "direction": "desc"},
            ],
        })
        # North first: 200 then 100; South: 300 then 100
        assert _rows(r)[0] == {"region": "North", "sales": 200}


# ===========================================================================
# 6. COLUMN RESOLUTION
# ===========================================================================

class TestColumnResolution:

    def test_case_insensitive_column(self):
        df = pd.DataFrame({"Total Sales": [100, 200]})
        r = sql_engine.execute(df, {"operation": "sum", "columns": ["total sales"]})
        assert "300" in _first(r)["Total Sales"]

    def test_normalized_token_match(self):
        df = pd.DataFrame({"Total Sales ($)": [100, 200]})
        r = sql_engine.execute(df, {"operation": "sum", "columns": ["totalsales"]})
        assert "300" in _first(r)["Total Sales ($)"]

    def test_column_with_spaces_in_filter(self):
        df = pd.DataFrame({"Order Date": ["2023-01-01", "2024-01-01"]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "order date", "operator": "=", "value": "2024"}]})
        assert len(_rows(r)) == 1

    def test_underscore_to_space_column(self):
        df = pd.DataFrame({"first_name": ["Alice", "Bob"]})
        r = sql_engine.execute(df, {"operation": "count", "filters": [{"column": "firstname", "operator": "=", "value": "Alice"}]})
        assert "1" in _first(r)["count"]


# ===========================================================================
# 7. REAL-WORLD CSV FIELD TYPES
# ===========================================================================

class TestRealWorldFieldTypes:

    def test_currency_string_field(self):
        """$1,234.56 style values — engine should handle as text."""
        df = pd.DataFrame({
            "price": ["$100.00", "$200.50", "$50.00"],
            "item": ["A", "B", "C"],
        })
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "price", "operator": "contains", "value": "$1"}]})
        assert len(_rows(r)) == 1

    def test_percentage_string_field(self):
        df = pd.DataFrame({
            "growth": ["10%", "25%", "5%", "30%"],
        })
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "growth", "operator": "contains", "value": "2"}]})
        assert len(_rows(r)) == 1

    def test_boolean_values(self):
        df = pd.DataFrame({"active": [True, False, True, True]})
        r = sql_engine.execute(df, {"operation": "count", "filters": [{"column": "active", "operator": "=", "value": "True"}]})
        assert "3" in _first(r)["count"]

    def test_scientific_notation_values(self):
        df = pd.DataFrame({"amount": [1.5e6, 2.3e6, 0.5e6]})
        r = sql_engine.execute(df, {"operation": "sum", "columns": ["amount"]})
        assert "Sum" in _first(r)["amount"]

    def test_unicode_column_names(self):
        df = pd.DataFrame({"Société": ["Paris", "Lyon"], "Chiffre d'affaires": [1000, 2000]})
        r = sql_engine.execute(df, {"operation": "sum", "columns": ["Chiffre d'affaires"]})
        assert "3000" in str(_first(r))

    def test_unicode_values(self):
        df = pd.DataFrame({"city": ["München", "Zürich", "Paris"]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "city", "operator": "=", "value": "München"}]})
        assert len(_rows(r)) == 1

    def test_mixed_numeric_text_in_column(self):
        """Column with mostly numbers but some non-numeric entries."""
        df = pd.DataFrame({"score": ["85", "90", "N/A", "75", "N/A"]})
        r = sql_engine.execute(df, {"operation": "avg", "columns": ["score"]})
        # Should compute avg over numeric values: (85+90+75)/3 = 83.33
        assert "Average" in _first(r)["score"]

    def test_all_null_column_null_pct(self):
        df = pd.DataFrame({"val": [None, None, None]})
        r = sql_engine.execute(df, {"operation": "null_pct", "columns": ["val"]})
        assert "100" in _first(r)["val"]

    def test_single_row_dataframe(self):
        df = pd.DataFrame({"a": [42], "b": ["hello"]})
        r = sql_engine.execute(df, {"operation": "count"})
        assert "1" in _first(r)["count"]

    def test_empty_dataframe_after_filter(self):
        df = pd.DataFrame({"region": ["North", "South"]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "region", "operator": "=", "value": "East"}]})
        assert _is_refusal(r)

    def test_dict_style_filters_converted(self):
        """Old dict-style filters like {'city': 'London'} must be handled."""
        df = pd.DataFrame({"city": ["London", "Paris", "London"]})
        r = sql_engine.execute(df, {"operation": "count", "filters": {"city": "London"}})
        assert "2" in _first(r)["count"]

    def test_large_column_count_profile(self):
        """Profile should handle DataFrames with many columns without crashing."""
        cols = {f"col_{i}": list(range(10)) for i in range(30)}
        df = pd.DataFrame(cols)
        r = sql_engine.execute(df, {"operation": "profile"})
        assert len(_rows(r)) > 0

    def test_duplicate_values_in_group_by(self):
        df = pd.DataFrame({
            "tag": ["python", "python", "java", "java", "python"],
            "stars": [100, 200, 150, 250, 50],
        })
        r = sql_engine.execute(df, {"operation": "sum", "columns": ["stars"], "group_by": ["tag"]})
        python_row = next(row for row in _rows(r) if row["tag"] == "python")
        assert python_row["sum_stars"] == 350.0


# ===========================================================================
# 8. MULTI-SHEET / EXCEL SUPPORT
# ===========================================================================

class TestMultiSheetExcel:

    def _make_excel_dict(self):
        return {
            "Q1": pd.DataFrame({"region": ["North", "South"], "sales": [100, 200]}),
            "Q2": pd.DataFrame({"region": ["East", "West"], "sales": [300, 400]}),
        }

    def test_multi_sheet_sum_all(self):
        df_dict = self._make_excel_dict()
        r = sql_engine.execute(df_dict, {"operation": "sum", "columns": ["sales"]})
        rows = _rows(r)
        # Each sheet produces a sum row tagged with sheet name
        assert any(row.get("sheet") == "Q1" for row in rows)
        assert any(row.get("sheet") == "Q2" for row in rows)

    def test_multi_sheet_filter_by_sheet(self):
        df_dict = self._make_excel_dict()
        r = sql_engine.execute(df_dict, {
            "operation": "none",
            "filters": [{"column": "sheet", "operator": "=", "value": "Q1"}],
        })
        rows = _rows(r)
        assert all(row.get("sheet") == "Q1" for row in rows if "sheet" in row)

    def test_multi_sheet_count_per_sheet(self):
        df_dict = self._make_excel_dict()
        r = sql_engine.execute(df_dict, {"operation": "count"})
        rows = _rows(r)
        assert len(rows) == 2

    def test_flat_excel_sheet_column_filter(self):
        """Already-flattened Excel df with 'sheet' column as real column."""
        df = pd.DataFrame([
            {"sheet": "Sheet1", "id": 1, "val": 100},
            {"sheet": "Sheet1", "id": 2, "val": 200},
            {"sheet": "Sheet2", "id": 3, "val": 300},
        ])
        r = sql_engine.execute(df, {
            "operation": "sum",
            "columns": ["val"],
            "filters": [{"column": "sheet", "operator": "=", "value": "Sheet1"}],
        })
        assert "300" in _first(r)["val"]


# ===========================================================================
# 9. EDGE CASES & REFUSALS
# ===========================================================================

class TestEdgeCasesAndRefusals:

    def test_no_matching_rows_returns_refusal(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        r = sql_engine.execute(df, {"operation": "none", "filters": [{"column": "x", "operator": "=", "value": 99}]})
        assert _is_refusal(r)

    def test_invalid_quantile_value(self):
        df = pd.DataFrame({"val": [1, 2, 3]})
        r = sql_engine.execute(df, {"operation": "quantile", "columns": ["val"], "percentile": "not_a_number"})
        assert _is_refusal(r)

    def test_correlation_requires_two_columns(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        r = sql_engine.execute(df, {"operation": "correlation", "columns": ["x"]})
        assert _is_refusal(r)

    def test_sum_non_numeric_column_refusal(self):
        df = pd.DataFrame({"tag": ["alpha", "beta", "gamma"]})
        r = sql_engine.execute(df, {"operation": "sum", "columns": ["tag"]})
        assert _is_refusal(r)

    def test_filter_incompatible_type_refusal(self):
        """Numeric comparison against a non-numeric non-date text column."""
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        r = sql_engine.execute(df, {
            "operation": "sum",
            "columns": ["name"],
            "filters": [{"column": "name", "operator": ">", "value": "not-a-number"}],
        })
        assert _is_refusal(r)

    def test_unsupported_operation_graceful(self):
        df = pd.DataFrame({"val": [1, 2, 3]})
        # 'unknown_op' is not in SUPPORTED_OPERATIONS, falls through to 'none'
        r = sql_engine.execute(df, {"operation": "unknown_op"})
        assert "relevant_rows" in r

    def test_select_specific_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        r = sql_engine.execute(df, {"operation": "none", "columns": ["a", "c"]})
        for row in _rows(r):
            assert "a" in row and "c" in row
            assert "b" not in row

    def test_filter_then_count(self):
        df = pd.DataFrame({"dept": ["Eng", "HR", "Eng", "Finance", "Eng"]})
        r = sql_engine.execute(df, {
            "operation": "count",
            "filters": [{"column": "dept", "operator": "=", "value": "Eng"}],
        })
        assert "3" in _first(r)["count"]

    def test_summary_contains_filter_description(self):
        df = pd.DataFrame({"region": ["North", "South"], "sales": [100, 200]})
        r = sql_engine.execute(df, {
            "operation": "sum",
            "columns": ["sales"],
            "filters": [{"column": "region", "operator": "=", "value": "North"}],
        })
        assert "region" in _first(r)["_summary"]

    def test_operation_none_returns_all_rows(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        r = sql_engine.execute(df, {"operation": "none"})
        assert len(_rows(r)) == 3

    def test_operation_filter_synonym(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        r = sql_engine.execute(df, {"operation": "filter", "filters": [{"column": "a", "operator": ">", "value": 1}]})
        assert len(_rows(r)) == 2
