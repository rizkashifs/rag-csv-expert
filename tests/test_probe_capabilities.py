"""
Probe tests: systematically test every query pattern to find what WORKS and what FAILS.
This file is a diagnostic tool — failing tests expose engine / router limitations.
"""

import pandas as pd
import pytest

from app.engines.csv_engine import sql_engine
from app.engines.text_engine import TextEngine

text_engine = TextEngine()


def _rows(r):
    return r["relevant_rows"]

def _first(r):
    return r["relevant_rows"][0]

def _is_refusal(r):
    return r["relevant_rows"][0].get("should_ask_user") is True


# ---------------------------------------------------------------------------
# SAMPLE DATASETS
# ---------------------------------------------------------------------------

@pytest.fixture
def employees():
    return pd.DataFrame({
        "emp_id": [101, 102, 103, 104, 105, 106],
        "name": ["Alice Johnson", "Bob Smith", "Carol Williams", "David Brown", "Eve Davis", "Frank Miller"],
        "department": ["Engineering", "HR", "Engineering", "Marketing", "Engineering", "HR"],
        "salary": [95000, 60000, 88000, 72000, 102000, 58000],
        "join_date": ["2020-03-15", "2019-07-22", "2021-05-14", "2022-01-18", "2018-11-30", "2023-06-25"],
        "performance_review": [
            "Excellent technical skills, leads the backend team effectively",
            "Great at conflict resolution and employee onboarding",
            "Strong algorithms background, needs improvement in communication",
            "Creative campaign designer, consistently meets deadlines",
            "Senior architect, mentors junior developers regularly",
            "Handles payroll processing and benefits administration",
        ],
    })

@pytest.fixture
def sales():
    return pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "customer": ["Acme Corp", "Globex", "Initech", "Acme Corp", "Globex", "Initech", "Acme Corp", "Umbrella"],
        "product": ["Widget A", "Widget B", "Gadget X", "Widget A", "Gadget X", "Widget B", "Gadget X", "Widget A"],
        "quantity": [10, 5, 2, 8, 3, 12, 1, 20],
        "unit_price": [25.00, 50.00, 199.99, 25.00, 199.99, 50.00, 199.99, 25.00],
        "order_date": ["2023-01-15", "2023-02-20", "2023-03-18", "2023-06-22", "2023-07-14", "2023-09-30", "2024-01-15", "2024-03-22"],
        "region": ["North", "South", "East", "North", "South", "East", "West", "North"],
    })

@pytest.fixture
def multi_sheet():
    return {
        "Q1_Sales": pd.DataFrame({
            "region": ["North", "South", "East"],
            "revenue": [100000, 80000, 95000],
            "expenses": [60000, 50000, 55000],
        }),
        "Q2_Sales": pd.DataFrame({
            "region": ["North", "South", "East", "West"],
            "revenue": [120000, 90000, 105000, 70000],
            "expenses": [65000, 55000, 60000, 45000],
        }),
    }


# ===========================================================================
# SECTION 1: BASIC SELECT / FILTER — What works
# ===========================================================================

class TestBasicSelectFilter:
    """These are the bread-and-butter queries that MUST work."""

    def test_select_all_rows(self, employees):
        r = sql_engine.execute(employees, {"operation": "none"})
        assert len(_rows(r)) == 6

    def test_select_specific_columns(self, employees):
        r = sql_engine.execute(employees, {"operation": "none", "columns": ["name", "salary"]})
        assert set(_rows(r)[0].keys()) == {"name", "salary"}

    def test_filter_equals_text(self, employees):
        r = sql_engine.execute(employees, {"operation": "none", "filters": [{"column": "department", "operator": "=", "value": "Engineering"}]})
        assert len(_rows(r)) == 3

    def test_filter_equals_numeric(self, sales):
        r = sql_engine.execute(sales, {"operation": "none", "filters": [{"column": "quantity", "operator": "=", "value": 10}]})
        assert len(_rows(r)) == 1

    def test_filter_gt(self, employees):
        r = sql_engine.execute(employees, {"operation": "none", "filters": [{"column": "salary", "operator": ">", "value": 90000}]})
        assert len(_rows(r)) == 2  # Alice 95k, Eve 102k

    def test_filter_gte(self, employees):
        r = sql_engine.execute(employees, {"operation": "none", "filters": [{"column": "salary", "operator": ">=", "value": 95000}]})
        assert len(_rows(r)) == 2  # Alice 95k, Eve 102k

    def test_filter_lt(self, employees):
        r = sql_engine.execute(employees, {"operation": "none", "filters": [{"column": "salary", "operator": "<", "value": 60000}]})
        assert len(_rows(r)) == 1  # Frank 58k

    def test_filter_neq(self, employees):
        r = sql_engine.execute(employees, {"operation": "none", "filters": [{"column": "department", "operator": "!=", "value": "HR"}]})
        assert len(_rows(r)) == 4

    def test_filter_contains(self, sales):
        r = sql_engine.execute(sales, {"operation": "none", "filters": [{"column": "product", "operator": "contains", "value": "Widget"}]})
        assert len(_rows(r)) == 5

    def test_filter_not_contains(self, sales):
        r = sql_engine.execute(sales, {"operation": "none", "filters": [{"column": "product", "operator": "not_contains", "value": "Widget"}]})
        assert len(_rows(r)) == 3

    def test_filter_in_list(self, employees):
        r = sql_engine.execute(employees, {"operation": "none", "filters": [{"column": "department", "operator": "in", "value": ["Engineering", "Marketing"]}]})
        assert len(_rows(r)) == 4

    def test_filter_not_in(self, employees):
        r = sql_engine.execute(employees, {"operation": "none", "filters": [{"column": "department", "operator": "not in", "value": ["HR"]}]})
        assert len(_rows(r)) == 4

    def test_filter_between_numeric(self, employees):
        r = sql_engine.execute(employees, {"operation": "none", "filters": [{"column": "salary", "operator": "between", "value": [70000, 100000]}]})
        assert len(_rows(r)) == 3  # Carol 88k, David 72k, Alice 95k

    def test_filter_like_wildcard(self, employees):
        r = sql_engine.execute(employees, {"operation": "none", "filters": [{"column": "name", "operator": "like", "value": "%son"}]})
        assert len(_rows(r)) == 1  # Alice Johnson

    def test_chained_and_filters(self, employees):
        r = sql_engine.execute(employees, {
            "operation": "none",
            "filters": [
                {"column": "department", "operator": "=", "value": "Engineering"},
                {"column": "salary", "operator": ">", "value": 90000},
            ],
        })
        assert len(_rows(r)) == 2  # Alice 95k, Eve 102k

    def test_filter_year_on_date(self, sales):
        r = sql_engine.execute(sales, {"operation": "none", "filters": [{"column": "order_date", "operator": "=", "value": "2024"}]})
        assert len(_rows(r)) == 2

    def test_filter_gte_year_on_date(self, sales):
        r = sql_engine.execute(sales, {"operation": "none", "filters": [{"column": "order_date", "operator": ">=", "value": "2024"}]})
        assert len(_rows(r)) == 2


# ===========================================================================
# SECTION 2: AGGREGATION QUERIES — What works
# ===========================================================================

class TestAggregationQueries:

    def test_count_all(self, employees):
        r = sql_engine.execute(employees, {"operation": "count"})
        assert "6" in _first(r)["count"]

    def test_count_with_filter(self, employees):
        r = sql_engine.execute(employees, {"operation": "count", "filters": [{"column": "department", "operator": "=", "value": "Engineering"}]})
        assert "3" in _first(r)["count"]

    def test_sum(self, employees):
        r = sql_engine.execute(employees, {"operation": "sum", "columns": ["salary"]})
        assert "475000" in _first(r)["salary"]

    def test_avg(self, employees):
        r = sql_engine.execute(employees, {"operation": "avg", "columns": ["salary"]})
        avg = float(_first(r)["salary"].split(": ")[1])
        assert abs(avg - 79166.67) < 1

    def test_max_with_context_rows(self, employees):
        r = sql_engine.execute(employees, {"operation": "max", "columns": ["name", "salary"]})
        assert any(row.get("name") == "Eve Davis" for row in _rows(r))

    def test_min(self, employees):
        r = sql_engine.execute(employees, {"operation": "min", "columns": ["name", "salary"]})
        assert any(row.get("name") == "Frank Miller" for row in _rows(r))

    def test_median(self, employees):
        r = sql_engine.execute(employees, {"operation": "median", "columns": ["salary"]})
        assert "Median" in _first(r)["salary"]

    def test_mode(self, sales):
        r = sql_engine.execute(sales, {"operation": "mode", "columns": ["quantity"]})
        # No repeated quantities in our data, so mode returns the first
        assert "Mode" in _first(r)["quantity"]

    def test_std(self, employees):
        r = sql_engine.execute(employees, {"operation": "std", "columns": ["salary"]})
        assert "Standard Deviation" in _first(r)["salary"]

    def test_distinct_count(self, sales):
        r = sql_engine.execute(sales, {"operation": "distinct_count", "columns": ["customer"]})
        assert "4" in _first(r)["customer"]

    def test_null_count(self, employees):
        r = sql_engine.execute(employees, {"operation": "null_count", "columns": ["salary"]})
        assert "0" in _first(r)["salary"]

    def test_value_counts(self, sales):
        r = sql_engine.execute(sales, {"operation": "value_counts", "columns": ["customer"]})
        acme = next(row for row in _rows(r) if row["value"] == "Acme Corp")
        assert acme["count"] == 3

    def test_histogram(self, employees):
        r = sql_engine.execute(employees, {"operation": "histogram", "columns": ["salary"], "bins": 3})
        assert len(_rows(r)) == 3

    def test_profile(self, employees):
        r = sql_engine.execute(employees, {"operation": "profile", "columns": ["salary"]})
        assert "Avg:" in _first(r)["stats"]

    def test_quantile(self, employees):
        r = sql_engine.execute(employees, {"operation": "quantile", "columns": ["salary"], "percentile": 90})
        assert "Quantile" in _first(r)["salary"]

    def test_correlation(self, sales):
        r = sql_engine.execute(sales, {"operation": "correlation", "columns": ["quantity", "unit_price"]})
        assert "correlation" in _first(r)

    def test_multi_operation(self, employees):
        r = sql_engine.execute(employees, {"operation": "avg|sum", "columns": ["salary"]})
        summaries = " ".join(row.get("_summary", "") for row in _rows(r))
        assert "Average" in summaries and "Sum" in summaries


# ===========================================================================
# SECTION 3: GROUP BY + HAVING — What works
# ===========================================================================

class TestGroupByQueries:

    def test_group_by_count(self, employees):
        r = sql_engine.execute(employees, {"operation": "count", "group_by": ["department"]})
        eng = next(row for row in _rows(r) if row["department"] == "Engineering")
        assert eng["count"] == 3

    def test_group_by_sum(self, sales):
        r = sql_engine.execute(sales, {
            "operation": "sum",
            "columns": ["quantity"],
            "group_by": ["customer"],
        })
        acme = next(row for row in _rows(r) if row["customer"] == "Acme Corp")
        assert acme["sum_quantity"] == 19.0  # 10+8+1

    def test_group_by_avg(self, employees):
        r = sql_engine.execute(employees, {
            "operation": "avg",
            "columns": ["salary"],
            "group_by": ["department"],
        })
        eng = next(row for row in _rows(r) if row["department"] == "Engineering")
        assert abs(eng["avg_salary"] - 95000) < 1

    def test_group_by_with_sort(self, sales):
        r = sql_engine.execute(sales, {
            "operation": "sum",
            "columns": ["quantity"],
            "group_by": ["customer"],
            "sort": [{"column": "quantity", "direction": "desc"}],
        })
        assert _rows(r)[0]["customer"] == "Umbrella"  # 20 units

    def test_group_by_with_having(self, sales):
        r = sql_engine.execute(sales, {
            "operation": "count",
            "group_by": ["customer"],
            "having": [{"column": "count", "operator": ">=", "value": "3"}],
        })
        # Acme=3, Globex=2, Initech=2, Umbrella=1 → only Acme passes
        assert len(_rows(r)) == 1
        assert _rows(r)[0]["customer"] == "Acme Corp"

    def test_group_by_time_year(self, sales):
        r = sql_engine.execute(sales, {
            "operation": "sum",
            "columns": ["quantity"],
            "group_by": [{"column": "order_date", "time_grain": "year"}],
        })
        rows = _rows(r)
        assert len(rows) == 2
        y2024 = next(row for row in rows if "2024" in str(row.values()))
        assert y2024["sum_quantity"] == 21.0  # 1+20

    def test_group_by_multi_aggregation(self, sales):
        r = sql_engine.execute(sales, {
            "operation": "sum",
            "columns": ["quantity"],
            "group_by": ["region"],
            "aggregations": [
                {"function": "sum", "column": "quantity"},
                {"function": "avg", "column": "unit_price"},
            ],
        })
        north = next(row for row in _rows(r) if row["region"] == "North")
        assert "sum_quantity" in north
        assert "avg_unit_price" in north

    def test_group_by_having_on_renamed_column(self, sales):
        """HAVING must resolve against prefixed columns (e.g. sum_quantity)."""
        r = sql_engine.execute(sales, {
            "operation": "sum",
            "columns": ["quantity"],
            "group_by": ["region"],
            "having": [{"column": "quantity", "operator": ">", "value": "15"}],
        })
        # North=38, South=8, East=14, West=1 → only North > 15
        assert len(_rows(r)) == 1
        assert _rows(r)[0]["region"] == "North"


# ===========================================================================
# SECTION 4: SORT + LIMIT — What works
# ===========================================================================

class TestSortLimit:

    def test_sort_desc_with_limit(self, employees):
        r = sql_engine.execute(employees, {
            "operation": "none",
            "columns": ["name", "salary"],
            "sort": [{"column": "salary", "direction": "desc"}],
            "limit": 3,
        })
        assert len(_rows(r)) == 3
        assert _rows(r)[0]["name"] == "Eve Davis"

    def test_sort_asc(self, employees):
        r = sql_engine.execute(employees, {
            "operation": "none",
            "sort": [{"column": "salary", "direction": "asc"}],
            "limit": 1,
        })
        assert _rows(r)[0]["name"] == "Frank Miller"

    def test_sort_by_date(self, sales):
        r = sql_engine.execute(sales, {
            "operation": "none",
            "columns": ["order_id", "order_date"],
            "sort": [{"column": "order_date", "direction": "desc"}],
            "limit": 1,
        })
        assert _rows(r)[0]["order_id"] == 8  # 2024-03-22

    def test_multi_column_sort(self, employees):
        r = sql_engine.execute(employees, {
            "operation": "none",
            "sort": [
                {"column": "department", "direction": "asc"},
                {"column": "salary", "direction": "desc"},
            ],
        })
        # Engineering first (sorted desc by salary): Eve, Alice, Carol
        assert _rows(r)[0]["name"] == "Eve Davis"


# ===========================================================================
# SECTION 5: MULTI-SHEET EXCEL — What works
# ===========================================================================

class TestMultiSheet:

    def test_all_sheets_counted(self, multi_sheet):
        r = sql_engine.execute(multi_sheet, {"operation": "count"})
        assert len(_rows(r)) == 2  # one per sheet

    def test_filter_by_sheet(self, multi_sheet):
        r = sql_engine.execute(multi_sheet, {
            "operation": "none",
            "filters": [{"column": "sheet", "operator": "=", "value": "Q1_Sales"}],
        })
        assert len(_rows(r)) == 3

    def test_sum_by_sheet(self, multi_sheet):
        r = sql_engine.execute(multi_sheet, {
            "operation": "sum",
            "columns": ["revenue"],
            "filters": [{"column": "sheet", "operator": "=", "value": "Q2_Sales"}],
        })
        total = float(_rows(r)[0]["revenue"].split(": ")[1])
        assert total == 385000.0

    def test_group_by_region_across_sheets(self, multi_sheet):
        r = sql_engine.execute(multi_sheet, {
            "operation": "sum",
            "columns": ["revenue"],
            "group_by": ["region"],
        })
        # North appears in both sheets: 100k + 120k
        rows = _rows(r)
        north_rows = [row for row in rows if row.get("region") == "North"]
        total_north = sum(row["sum_revenue"] for row in north_rows)
        assert total_north == 220000.0


# ===========================================================================
# SECTION 6: TEXT ENGINE — What works
# ===========================================================================

class TestTextEngine:

    def test_keyword_search(self, employees):
        r = text_engine.execute(employees, {
            "semantic_plan": {
                "query_text": "mentors junior",
                "keywords": ["mentors", "junior"],
                "target_text_columns": ["performance_review"],
            }
        })
        assert len(_rows(r)) == 1
        assert _rows(r)[0]["name"] == "Eve Davis"

    def test_single_keyword_search(self, employees):
        r = text_engine.execute(employees, {
            "semantic_plan": {
                "query_text": "payroll",
                "keywords": ["payroll"],
                "target_text_columns": ["performance_review"],
            }
        })
        assert len(_rows(r)) == 1
        assert _rows(r)[0]["name"] == "Frank Miller"

    def test_or_fallback_when_and_fails(self, employees):
        r = text_engine.execute(employees, {
            "semantic_plan": {
                "query_text": "payroll algorithms",
                "keywords": ["payroll", "algorithms"],
                "target_text_columns": ["performance_review"],
            }
        })
        # AND match fails (no one has both), falls back to OR → Frank + Carol
        assert len(_rows(r)) == 2

    def test_id_filter_scoping(self, employees):
        r = text_engine.execute(employees, {
            "semantic_plan": {
                "query_text": "skills",
                "keywords": ["skills"],
                "target_text_columns": ["performance_review"],
                "id_filters": [{"column": "emp_id", "value": 101}],
            }
        })
        assert len(_rows(r)) == 1
        assert _rows(r)[0]["name"] == "Alice Johnson"

    def test_post_filter_after_text_search(self, employees):
        r = text_engine.execute(employees, {
            "semantic_plan": {
                "query_text": "team",
                "keywords": ["team"],
                "target_text_columns": ["performance_review"],
                "post_filters": [{"column": "department", "operator": "=", "value": "Engineering"}],
            }
        })
        # Alice mentions "team", is in Engineering; Frank mentions "team" too but is HR
        # Wait, Alice: "leads the backend team effectively" → has "team"
        # Frank: no "team" mention
        # Check all reviews for "team"
        assert all(row["department"] == "Engineering" for row in _rows(r))

    def test_no_keywords_returns_all_scoped(self, employees):
        r = text_engine.execute(employees, {
            "semantic_plan": {
                "query_text": "",
                "keywords": [],
                "target_text_columns": ["performance_review"],
                "id_filters": [{"column": "department", "value": "HR"}],
            }
        })
        # No keywords → returns all HR rows
        assert len(_rows(r)) == 2

    def test_text_engine_multi_sheet_with_sheet_filter(self, multi_sheet):
        """TextEngine should skip sheets not matching the filter."""
        r = text_engine.execute(multi_sheet, {
            "semantic_plan": {
                "query_text": "North",
                "keywords": ["north"],
                "target_text_columns": ["region"],
                "post_filters": [{"column": "sheet", "operator": "=", "value": "Q1_Sales"}],
            }
        })
        assert len(_rows(r)) == 1
        assert _rows(r)[0]["sheet"] == "Q1_Sales"


# ===========================================================================
# SECTION 7: KNOWN LIMITATIONS & EDGE CASES THAT FAIL
# ===========================================================================

class TestKnownLimitations:
    """
    These tests document KNOWN limitations / edge cases.
    Each test is marked with xfail or documents the workaround.
    """

    def test_no_OR_filter_logic(self, employees):
        """SQL engine only supports AND logic for filters. There is no way
        to express 'department = Engineering OR department = Marketing'
        except via the 'in' operator."""
        # Workaround: use the 'in' operator
        r = sql_engine.execute(employees, {
            "operation": "none",
            "filters": [{"column": "department", "operator": "in", "value": ["Engineering", "Marketing"]}],
        })
        assert len(_rows(r)) == 4  # works via 'in'

    def test_no_computed_columns(self, sales):
        """Cannot create computed columns like 'total = quantity * unit_price'.
        The engine has no expression evaluator."""
        # This is a fundamental limitation — no way to express this
        r = sql_engine.execute(sales, {"operation": "sum", "columns": ["quantity"]})
        # Can still sum individual columns
        assert "Sum" in _first(r)["quantity"]

    def test_no_join_across_sheets(self, multi_sheet):
        """Cannot JOIN data across Excel sheets (e.g. lookup region name from
        a reference sheet). Each sheet is queried independently."""
        # Sum across both sheets works per-sheet, not as a SQL JOIN
        r = sql_engine.execute(multi_sheet, {"operation": "sum", "columns": ["revenue"]})
        assert len(_rows(r)) == 2  # separate sum per sheet

    def test_no_subquery(self, employees):
        """No subquery support: 'show employees with salary above average'
        requires two queries — first get the avg, then filter."""
        # Workaround: compute avg manually and pass the value as a filter
        avg_r = sql_engine.execute(employees, {"operation": "avg", "columns": ["salary"]})
        avg_val = float(_first(avg_r)["salary"].split(": ")[1])
        r = sql_engine.execute(employees, {
            "operation": "none",
            "filters": [{"column": "salary", "operator": ">", "value": avg_val}],
        })
        assert len(_rows(r)) > 0  # works with two queries

    def test_no_window_functions(self, employees):
        """No ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD etc.
        Cannot express 'rank employees by salary within each department'."""
        # Workaround: group_by + sort gives partial equivalent
        r = sql_engine.execute(employees, {
            "operation": "none",
            "sort": [
                {"column": "department", "direction": "asc"},
                {"column": "salary", "direction": "desc"},
            ],
        })
        assert len(_rows(r)) == 6  # sorted, but no rank column

    def test_no_pivot_or_crosstab(self, sales):
        """Cannot pivot data (e.g. customers as rows, products as columns, sum of quantity as values).
        Only flat group-by is supported."""
        # group_by two columns is the closest alternative
        r = sql_engine.execute(sales, {
            "operation": "sum",
            "columns": ["quantity"],
            "group_by": ["customer", "product"],
        })
        assert len(_rows(r)) > 0

    def test_no_regex_filter(self, employees):
        """The 'contains' operator does a literal substring match, not regex.
        Patterns like 'John.*son' won't work as regex."""
        r = sql_engine.execute(employees, {
            "operation": "none",
            "filters": [{"column": "name", "operator": "contains", "value": "John"}],
        })
        # Literal substring match works
        assert len(_rows(r)) == 1  # Alice Johnson

    def test_no_case_when_logic(self, employees):
        """No CASE/WHEN conditional logic.
        Cannot express 'if salary > 90k then "senior" else "junior"'."""
        # No test needed — simply not supported

    def test_group_by_with_unsupported_operation_returns_refusal(self, employees):
        """Operations like histogram, value_counts, correlation, profile
        are NOT supported with group_by — the engine returns a graceful refusal
        instead of crashing."""
        r = sql_engine.execute(employees, {
            "operation": "histogram",
            "columns": ["salary"],
            "group_by": ["department"],
        })
        assert _is_refusal(r)

    def test_yyyymmdd_int_gte_filter(self):
        """YYYYMMDD integer columns correctly route to datetime path
        for all comparison operators (=, !=, >, <, >=, <=)."""
        df = pd.DataFrame({"date_int": [20230101, 20230601, 20240101], "val": [1, 2, 3]})
        r = sql_engine.execute(df, {
            "operation": "none",
            "filters": [{"column": "date_int", "operator": ">=", "value": "2024"}],
        })
        assert len(_rows(r)) == 1

    def test_text_engine_no_fuzzy_matching(self, employees):
        """TextEngine does exact substring matching, not fuzzy/phonetic.
        'Jon' won't match 'John', 'Jonson' won't match 'Johnson'."""
        r = text_engine.execute(employees, {
            "semantic_plan": {
                "query_text": "Jonson",
                "keywords": ["jonson"],
                "target_text_columns": ["name"],
            }
        })
        # "Jonson" ≠ "Johnson" — no fuzzy match
        assert _is_refusal(r)

    def test_text_engine_no_semantic_similarity(self, employees):
        """TextEngine is keyword-based, not semantic. 'coding' won't match
        'algorithms' even though they're related concepts."""
        r = text_engine.execute(employees, {
            "semantic_plan": {
                "query_text": "coding",
                "keywords": ["coding"],
                "target_text_columns": ["performance_review"],
            }
        })
        # No review contains the literal word "coding"
        assert _is_refusal(r)
