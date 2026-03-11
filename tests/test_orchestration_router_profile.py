from unittest.mock import patch

import pandas as pd

from app.services.orchestration import orchestrator


@patch.object(orchestrator.summary_agent, "run", return_value="Workbook summary")
@patch("app.services.orchestration.ingestion_service")
@patch("app.services.orchestration.file_registry")
@patch("app.services.orchestration.get_history")
def test_excel_router_gets_compact_workbook_profile(mock_get_history, mock_registry, mock_ingestion, mock_summary_run):
    sheet1 = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    sheet2 = pd.DataFrame({"b": [7, 8], "d": [9, 10]})

    mock_ingestion.load_data.return_value = {
        "df": {"Sheet1": sheet1, "Sheet2": sheet2},
        "semantic_context": "verbose semantic context",
        "row_count": 4,
        "type": "excel",
        "text_heavy": False,
        "sample_data": "",
    }
    mock_registry.get_file_info.return_value = None
    mock_get_history.return_value = []

    captured = {}

    def fake_router_run(payload):
        captured["dataset_profile"] = payload["dataset_profile"]
        return {
            "route": "PROFILE_ONLY",
            "use_routing_agent": True,
            "schema": {"operation": "profile"},
        }

    with patch.object(orchestrator.router, "run", side_effect=fake_router_run):
        result = orchestrator.run_pipeline("profile this workbook", "dummy.xlsx", "dummy_index")

    assert result["question_type"] == "profile_only"
    assert captured["dataset_profile"] == "\n".join(
        [
            "Workbook with 2 sheets",
            "Total rows: 4, Max columns :3",
            "Sheets:",
            "Sheet1: rows = 2, cols=3, columns=[a,b,c]",
            "Sheet2: rows = 2, cols=2, columns=[b,d]",
            "Columns (union): a,b,c,d",
        ]
    )


@patch.object(orchestrator.summary_agent, "run", return_value="CSV summary")
@patch("app.services.orchestration.ingestion_service")
@patch("app.services.orchestration.file_registry")
@patch("app.services.orchestration.get_history")
def test_csv_router_gets_compact_dataset_profile(mock_get_history, mock_registry, mock_ingestion, mock_summary_run):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8], "e": [9, 10]})

    mock_ingestion.load_data.return_value = {
        "df": df,
        "semantic_context": "verbose csv context",
        "row_count": 2,
        "type": "csv",
        "text_heavy": False,
        "sample_data": "",
    }
    mock_registry.get_file_info.return_value = None
    mock_get_history.return_value = []

    captured = {}

    def fake_router_run(payload):
        captured["dataset_profile"] = payload["dataset_profile"]
        return {
            "route": "PROFILE_ONLY",
            "use_routing_agent": True,
            "schema": {"operation": "profile"},
        }

    with patch.object(orchestrator.router, "run", side_effect=fake_router_run):
        result = orchestrator.run_pipeline("profile this csv", "dummy.csv", "dummy_index")

    assert result["question_type"] == "profile_only"
    assert captured["dataset_profile"] == "\n".join(
        [
            "Rows: 2, Columns: 5",
            "Columns: a,b,c,d,e",
        ]
    )


@patch.object(orchestrator.answer, "run", return_value="ok")
@patch.object(orchestrator.retriever, "run", return_value={"relevant_rows": []})
@patch.object(orchestrator.summary_agent, "run", return_value="Workbook summary")
@patch("app.services.orchestration.ingestion_service")
@patch("app.services.orchestration.file_registry")
@patch("app.services.orchestration.get_history")
def test_excel_engine_receives_dataframe_built_from_flat_rows(
    mock_get_history,
    mock_registry,
    mock_ingestion,
    mock_summary_run,
    mock_retriever_run,
    mock_answer_run,
):
    sheet1 = pd.DataFrame({"id": [123], "Gender": ["Male"]})
    sheet2 = pd.DataFrame({"id": [124], "Age": [45]})
    mock_ingestion.load_data.return_value = {
        "df": {"Sheet1": sheet1, "Sheet2": sheet2},
        "flat_rows": [
            {"sheet": "Sheet1", "id": "123", "Gender": "Male"},
            {"sheet": "Sheet2", "id": "124", "Age": 45},
        ],
        "semantic_context": "verbose semantic context",
        "row_count": 2,
        "type": "excel",
        "text_heavy": False,
        "sample_data": "",
    }
    mock_registry.get_file_info.return_value = None
    mock_get_history.return_value = []

    with patch.object(
        orchestrator.router,
        "run",
        return_value={"route": "SQL_ENGINE", "use_routing_agent": True, "schema": {"operation": "filter"}},
    ):
        orchestrator.run_pipeline("show rows", "dummy.xlsx", "dummy_index")

    engine_df = mock_retriever_run.call_args.kwargs["input_data"]["df"] if "input_data" in mock_retriever_run.call_args.kwargs else mock_retriever_run.call_args.args[0]["df"]
    assert isinstance(engine_df, pd.DataFrame)
    records = engine_df.to_dict("records")
    assert records[0]["sheet"] == "Sheet1"
    assert records[0]["id"] == "123"
    assert records[0]["Gender"] == "Male"
    assert pd.isna(records[0]["Age"])
    assert records[1]["sheet"] == "Sheet2"
    assert records[1]["id"] == "124"
    assert pd.isna(records[1]["Gender"])
    assert records[1]["Age"] == 45


@patch.object(orchestrator.answer, "run", return_value="ok")
@patch.object(orchestrator.retriever, "run", return_value={"relevant_rows": []})
@patch.object(orchestrator.summary_agent, "run", return_value="CSV summary")
@patch("app.services.orchestration.ingestion_service")
@patch("app.services.orchestration.file_registry")
@patch("app.services.orchestration.get_history")
def test_csv_engine_receives_dataframe_built_from_flat_rows(
    mock_get_history,
    mock_registry,
    mock_ingestion,
    mock_summary_run,
    mock_retriever_run,
    mock_answer_run,
):
    df = pd.DataFrame({"date": ["2024"], "age": [34]})
    mock_ingestion.load_data.return_value = {
        "df": df,
        "flat_rows": [{"id": "0", "date": "2024", "age": 34}],
        "semantic_context": "verbose csv context",
        "row_count": 1,
        "type": "csv",
        "text_heavy": False,
        "sample_data": "",
    }
    mock_registry.get_file_info.return_value = None
    mock_get_history.return_value = []

    with patch.object(
        orchestrator.router,
        "run",
        return_value={"route": "SQL_ENGINE", "use_routing_agent": True, "schema": {"operation": "filter"}},
    ):
        orchestrator.run_pipeline("show rows", "dummy.csv", "dummy_index")

    engine_df = mock_retriever_run.call_args.kwargs["input_data"]["df"] if "input_data" in mock_retriever_run.call_args.kwargs else mock_retriever_run.call_args.args[0]["df"]
    assert isinstance(engine_df, pd.DataFrame)
    assert engine_df.to_dict("records") == [{"id": "0", "date": "2024", "age": 34}]
