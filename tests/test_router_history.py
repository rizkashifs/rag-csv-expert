from app.agents.router import RouterAgent


def test_history_follow_up_enriches_keyword_intent():
    agent = RouterAgent()
    result = agent.run({
        "query": "and by region",
        "dataset_profile": "",
        "semantic_summary": "",
        "text_heavy": False,
        "history": [{"user": "What is the total revenue", "assistant": "..."}],
    })

    assert result["route"] == "KEYWORD_ENGINE"
    assert result["schema"]["operation"] == "sum"


def test_non_follow_up_query_uses_current_query_only():
    agent = RouterAgent()
    result = agent.run({
        "query": "average salary",
        "dataset_profile": "",
        "semantic_summary": "",
        "text_heavy": False,
        "history": [{"user": "What is the total revenue", "assistant": "..."}],
    })

    assert result["route"] == "KEYWORD_ENGINE"
    assert result["schema"]["operation"] == "avg"


def test_router_uses_history_service_with_chat_id(monkeypatch):
    agent = RouterAgent()

    monkeypatch.setattr(
        "app.agents.router.get_history",
        lambda chat_id: [{"user": "What is the total revenue", "assistant": "..."}] if chat_id == "chat-1" else [],
    )

    result = agent.run({
        "query": "and by region",
        "dataset_profile": "",
        "semantic_summary": "",
        "text_heavy": False,
        "chat_id": "chat-1",
    })

    assert result["route"] == "KEYWORD_ENGINE"
    assert result["schema"]["operation"] == "sum"


def test_router_returns_enriched_schema_for_keyword_route():
    agent = RouterAgent()
    result = agent.run({
        "query": "total revenue by region",
        "dataset_profile": "",
        "semantic_summary": "",
        "text_heavy": False,
        "history": [],
    })

    assert result["route"] == "KEYWORD_ENGINE"
    assert result["schema"]["operation"] == "sum"
    assert result["schema"]["group_by"] == ["region"]
    assert "sql_plan" in result["schema"]
    assert result["schema"]["sql_plan"]["group_by"] == ["region"]


def test_router_promotes_to_sql_engine_for_complex_aggregation():
    agent = RouterAgent()
    result = agent.run({
        "query": 'total sales by region where "sales" > 1000 and "profit" > 100',
        "dataset_profile": "",
        "semantic_summary": "",
        "text_heavy": False,
        "history": [],
    })

    assert result["route"] == "SQL_ENGINE"
    assert result["schema"]["engine_mode"] == "sql"
    assert len(result["schema"]["filters"]) >= 2


def test_router_returns_semantic_plan_for_text_engine_queries():
    agent = RouterAgent()
    result = agent.run({
        "query": "find rows similar to customer complaining about delays",
        "dataset_profile": "",
        "semantic_summary": "",
        "text_heavy": True,
        "history": [],
    })

    assert result["route"] == "TEXT_TABLE_RAG"
    assert result["schema"]["operation"] == "semantic"
    assert "semantic_plan" in result["schema"]
    assert result["schema"]["semantic_plan"]["query_text"] == "find rows similar to customer complaining about delays"


def test_router_normalizes_sql_plan_to_top_level_fields(monkeypatch):
    agent = RouterAgent()

    monkeypatch.setattr(
        "app.agents.router.llm_client.generate",
        lambda messages, options=None: '{"route":"SQL_ENGINE","schema":{"operation":"sum","sql_plan":{"target_columns":["sales"],"filters":[{"column":"region","operator":"=","value":"west"}],"group_by":["region"],"aggregations":[{"function":"sum","column":"sales"}],"order_by":[{"column":"sales","direction":"desc"}],"limit":5}}}',
    )

    result = agent.run({
        "query": "please compute it",
        "dataset_profile": "",
        "semantic_summary": "",
        "text_heavy": False,
        "history": [],
    })

    assert result["route"] == "SQL_ENGINE"
    assert result["schema"]["columns"] == ["sales"]
    assert result["schema"]["filters"][0]["column"] == "region"
    assert result["schema"]["group_by"] == ["region"]
    assert result["schema"]["limit"] == 5


def test_router_refuses_when_sql_intent_is_incomplete(monkeypatch):
    agent = RouterAgent()

    monkeypatch.setattr(
        "app.agents.router.llm_client.generate",
        lambda messages, options=None: '{"route":"SQL_ENGINE","schema":{"operation":"sum","columns":[],"filters":[]}}',
    )

    result = agent.run({
        "query": "compute this metric for me",
        "dataset_profile": "",
        "semantic_summary": "",
        "text_heavy": False,
        "history": [],
    })

    assert result["route"] == "REFUSE"
    assert "follow_up_questions" in result["schema"]
    assert any("numeric column" in question.lower() for question in result["schema"]["follow_up_questions"])
