
import pandas as pd
from app.engines.csv_engine import sql_engine

def test_time_and_having():
    # Setup Data
    data = {
        "Date": ["2023-01-15", "2023-02-20", "2023-05-10", "2024-01-05", "2024-03-12"],
        "Region": ["North", "North", "South", "South", "East"],
        "Sales": [100, 200, 150, 400, 300]
    }
    df = pd.DataFrame(data)
    print("--- Source Data ---")
    print(df)
    print("\n")

    # Test 1: Time Intelligence (Group by Year)
    # 2023: 100+200+150 = 450
    # 2024: 400+300 = 700
    intent1 = {
        "operation": "sum",
        "columns": ["Sales"],
        "group_by": [{"column": "Date", "time_grain": "year"}]
    }
    res1 = sql_engine.execute(df, intent1)
    print("Test 1 (Trend by Year):", res1)
    data1 = res1["relevant_rows"]
    assert len(data1) == 2
    # Check if 'Date_year' or similar exists and values are correct
    item_2024 = next(item for item in data1 if "2024" in str(item.values()))
    assert item_2024["sum_Sales"] == 700

    # Test 2: HAVING Clause
    # Group by Region, Sum Sales.
    # North: 300, South: 550, East: 300
    # Having Sales > 400 -> Should only return South
    intent2 = {
        "operation": "sum",
        "columns": ["Sales"],
        "group_by": ["Region"],
        "having": [{"column": "Sales", "operator": ">", "value": "400"}]
    }
    res2 = sql_engine.execute(df, intent2)
    print("Test 2 (Having Sales > 400):", res2)
    data2 = res2["relevant_rows"]
    assert len(data2) == 1
    assert data2[0]["Region"] == "South"
    assert data2[0]["sum_Sales"] == 550

    print("\n--- All Tests Passed ---")

if __name__ == "__main__":
    test_time_and_having()
