import pandas as pd

from app.agents.router import RouterAgent
from app.engines.csv_engine import sql_engine


def test_router_sql_schema_executes_on_csv_engine(monkeypatch):
    agent = RouterAgent()

    monkeypatch.setattr(
        "app.agents.router.llm_client.generate",
        lambda messages, options=None: (
            '{"route":"SQL_ENGINE","schema":{"operation":"avg","columns":["Sales"],'
            '"filters":[{"column":"Date","operator":"=","value":"2024"}],'
            '"group_by":[{"column":"Region","time_grain":"null"}],'
            '"sort":[{"column":"Sales","direction":"desc"}],"limit":5}}'
        ),
    )

    route_result = agent.run(
        {
            "query": "average sales for 2024 by region sorted desc",
            "dataset_profile": "Columns: Date (date), Region (str), Sales (float)",
            "semantic_summary": "Sales by date and region",
            "text_heavy": False,
            "history": [],
        }
    )

    assert route_result["route"] == "SQL_ENGINE"
    schema = route_result["schema"]
    assert schema["operation"] == "avg"
    assert isinstance(schema["filters"], list)
    assert isinstance(schema["sort"], list)

    df = pd.DataFrame(
        {
            "Date": ["2023-01-10", "2024-01-20", "2024-05-20", "2024-02-15"],
            "Region": ["North", "North", "South", "South"],
            "Sales": [100, 300, 500, 700],
        }
    )

    engine_result = sql_engine.execute(df, schema)

    assert "relevant_rows" in engine_result
    assert isinstance(engine_result["relevant_rows"], list)
    assert len(engine_result["relevant_rows"]) == 2
    assert engine_result["relevant_rows"][0]["Sales"] >= engine_result["relevant_rows"][1]["Sales"]
