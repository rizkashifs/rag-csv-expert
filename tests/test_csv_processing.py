import pytest
import pandas as pd
import os
import io
from app.services.ingestion import IngestionService
from app.engines.csv_engine import CSVEngine

@pytest.fixture
def ingestion_service():
    return IngestionService()

@pytest.fixture
def csv_engine():
    return CSVEngine()

def test_read_csv_basic(ingestion_service, tmp_path):
    # Create a simple CSV
    csv_content = "name,age,city\nAlice,30,New York\nBob,25,Los Angeles"
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)
    
    result = ingestion_service.read_csv(str(csv_file))
    
    assert result["row_count"] == 2
    assert "name" in result["schema"]
    assert "Alice" in result["semantic_context"]

def test_read_csv_cleaning(ingestion_service, tmp_path):
    # CSV with dirty headers and some empty rows
    csv_content = " name! , age? \nAlice,30\n\nBob,25\n\n"
    csv_file = tmp_path / "dirty.csv"
    csv_file.write_text(csv_content)
    
    result = ingestion_service.read_csv(str(csv_file))
    
    assert "name" in result["schema"]
    assert "age" in result["schema"]
    assert result["row_count"] == 2 # Empty rows dropped

def test_csv_engine_operations(csv_engine):
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [30, 25, 35],
        "city": ["New York", "Los Angeles", "New York"]
    })
    
    # Test Count
    result = csv_engine.execute(df, {"operation": "count"})
    assert result["relevant_rows"][0]["_summary"].lower().startswith("count of rows")
    
    # Test Sum
    result = csv_engine.execute(df, {"operation": "sum", "columns": ["age"]})
    assert "Sum: 90" in result["relevant_rows"][0]["age"]
    
    # Test Filter
    result = csv_engine.execute(df, {"operation": "filter", "filters": {"city": "New York"}})
    assert len(result["relevant_rows"]) == 2
    assert result["relevant_rows"][0]["name"] == "Alice"

def test_csv_engine_routes_filter_type_issues_to_refusal(csv_engine):
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [30, 25, 35],
    })

    result = csv_engine.execute(
        df,
        {
            "operation": "sum",
            "columns": ["age"],
            "filters": [{"column": "age", "operator": ">", "value": "not-a-number"}],
        },
    )

    assert "relevant_rows" in result
    assert result["relevant_rows"][0]["should_ask_user"] is True
    assert "Please clarify" in result["relevant_rows"][0]["_summary"]
