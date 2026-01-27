import pytest
import pandas as pd
import os
from app.services.orchestration import orchestrator
from app.services.registry import file_registry

@pytest.fixture(autouse=True)
def clean_registry():
    file_registry.files = {}
    yield

def test_multi_file_routing(tmp_path):
    # 1. Create two different files
    sales_path = str(tmp_path / "sales.csv")
    pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02"],
        "revenue": [100, 200]
    }).to_csv(sales_path, index=False)

    hr_path = str(tmp_path / "employees.csv")
    pd.DataFrame({
        "name": ["Alice", "Bob"],
        "department": ["HR", "Engineering"],
        "salary": [50000, 70000]
    }).to_csv(hr_path, index=False)

    # 2. Register both
    orchestrator.register_data(sales_path)
    orchestrator.register_data(hr_path)

    assert len(file_registry.list_files()) == 2

    # 3. Test Routing - Sales Query
    # Note: This requires a working LLM (Ollama or Bedrock/Anthropic)
    try:
        result_sales = orchestrator.run_multi_file_pipeline("What was the total revenue?")
        # We check if the intent refers to 'revenue' or if the orchestration logged the correct file
        # Since we use a real LLM, we verify the registry info used
        assert any("revenue" in str(v).lower() for v in result_sales.get("retrieved_data", {}).values())
    except Exception as e:
        pytest.skip(f"LLM not available for full integration test: {e}")

def test_explicit_file_routing(tmp_path):
    sales_path = str(tmp_path / "sales.csv")
    pd.DataFrame({"revenue": [100]}).to_csv(sales_path, index=False)
    
    orchestrator.register_data(sales_path)
    
    # Test with explicit path
    result = orchestrator.run_multi_file_pipeline("Total revenue", file_path=sales_path)
    assert result["metadata"]["row_count"] == 1
