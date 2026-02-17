import pandas as pd

from app.engines.csv_engine import sql_engine


def test_sort_numeric_values_stored_as_text():
    df = pd.DataFrame(
        {
            "customer": ["A", "B", "C", "D"],
            "revenue": ["100", "9", "75", "1000"],
        }
    )

    intent = {
        "operation": "none",
        "columns": ["customer", "revenue"],
        "sort": [{"column": "revenue", "direction": "desc"}],
        "limit": 4,
    }

    result = sql_engine.execute(df, intent)
    revenues = [int(row["revenue"]) for row in result["relevant_rows"]]
    assert revenues == [1000, 100, 75, 9]


def test_average_with_year_filter_on_date_column():
    df = pd.DataFrame(
        {
            "Date": ["2023-01-10", "2023-04-20", "2024-02-01", "2024-03-01"],
            "Sales": [100, 300, 500, 700],
        }
    )

    intent = {
        "operation": "avg",
        "columns": ["Sales"],
        "filters": [{"column": "Date", "operator": "=", "value": "2024"}],
    }

    result = sql_engine.execute(df, intent)
    assert "Average: 600" in result["relevant_rows"][0]["Sales"]


def test_sort_and_filter_with_fuzzy_column_names():
    df = pd.DataFrame(
        {
            "Order Date": ["2024-01-01", "2023-12-31", "2024-04-01"],
            "Total Sales ($)": [50, 500, 150],
        }
    )

    intent = {
        "operation": "none",
        "columns": ["order_date", "total_sales"],
        "filters": [{"column": "order date", "operator": ">=", "value": "2024"}],
        "sort": [{"column": "total sales", "direction": "DESCENDING"}],
        "limit": 5,
    }

    result = sql_engine.execute(df, intent)
    assert len(result["relevant_rows"]) == 2
    assert result["relevant_rows"][0]["Total Sales ($)"] == 150


def test_refusal_structure_when_nothing_matches():
    df = pd.DataFrame({"region": ["east", "west"], "sales": [10, 20]})
    intent = {
        "operation": "none",
        "filters": [{"column": "region", "operator": "=", "value": "north"}],
    }
    result = sql_engine.execute(df, intent)

    assert "relevant_rows" in result
    assert result["relevant_rows"][0]["should_ask_user"] is True
    assert "_summary" in result["relevant_rows"][0]
