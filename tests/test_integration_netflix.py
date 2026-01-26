import pytest
import os
from app.services.ingestion import ingestion_service
from app.engines.csv_engine import csv_engine

def test_netflix_dataset_ingestion():
    file_path = "c:/Users/admin/Documents/GitHub/rag-csv-expert/data/netflix_titles.csv"
    assert os.path.exists(file_path), "Netflix dataset not found. Run download first."
    
    result = ingestion_service.read_csv(file_path)
    
    assert result["row_count"] > 0
    assert "description" in result["schema"]
    assert "title" in result["schema"]
    
    # Check for multi-line handling (just visually or by checking a known record)
    # The read_csv uses pandas with engine='python' and sep=None which is robust.
    df = result["df"]
    
    # Verify we have variety in data
    assert "Movie" in df["type"].unique()
    assert "TV Show" in df["type"].unique()

def test_netflix_complex_query():
    file_path = "c:/Users/admin/Documents/GitHub/rag-csv-expert/data/netflix_titles.csv"
    ingestion_result = ingestion_service.read_csv(file_path)
    df = ingestion_result["df"]
    
    # Query: Find all Comedies released in 2019
    intent = {
        "operation": "filter",
        "filters": {
            "listed_in": "Comedies",
            "release_year": "2019"
        }
    }
    
    result = csv_engine.execute(df, intent)
    
    assert result["row_count"] > 0
    for record in result["data"]:
        assert "Comedies" in record["listed_in"]
        assert str(record["release_year"]) == "2019"
