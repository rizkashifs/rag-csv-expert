
import pandas as pd
from app.engines.csv_engine import sql_engine

def test_engine():
    # Setup Data
    data = {
        "Region": ["North", "North", "South", "South", "East", "West"],
        "Product": ["A", "B", "A", "B", "A", "B"],
        "Sales": [100, 200, 150, 50, 300, 400],
        "Cost": [50, 100, 75, 25, 150, 200]
    }
    df = pd.DataFrame(data)
    print("--- Source Data ---")
    print(df)
    print("\n")

    # Test 1: Group By Sum
    intent1 = {
        "operation": "sum",
        "columns": ["Sales"],
        "group_by": ["Region"]
    }
    res1 = sql_engine.execute(df, intent1)
    print("Test 1 (Group By Region):", res1)
    assert len(res1["relevant_rows"]) == 4 # N, S, E, W

    # Test 2: Sort + Limit
    intent2 = {
        "operation": "none",
        "columns": ["Region", "Sales"],
        "sort": [{"column": "Sales", "direction": "desc"}],
        "limit": 3
    }
    res2 = sql_engine.execute(df, intent2)
    print("Test 2 (Top 3 Sales):", res2)
    assert len(res2["relevant_rows"]) == 3
    assert res2["relevant_rows"][0]["Sales"] == 400 # West

    # Test 3: Filter + Group By + Sort
    intent3 = {
        "operation": "count",
        "group_by": ["Product"],
        "filters": [{"column": "Sales", "operator": ">", "value": "100"}],
        "sort": [{"column": "count", "direction": "desc"}] 
    }
    # Rows > 100: North-B(200), South-A(150), East-A(300), West-B(400)
    # Product A: 2 (South, East)
    # Product B: 2 (North, West)
    res3 = sql_engine.execute(df, intent3)
    print("Test 3 (Filter > 100 -> Count by Product):", res3)
    
    print("\n--- All Tests Passed ---")

if __name__ == "__main__":
    test_engine()
