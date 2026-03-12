from app.agents.router import RouterAgent


def test_history_follow_up_enriches_sql_intent(monkeypatch):
    agent = RouterAgent()
    
    monkeypatch.setattr(
        "app.agents.router.llm_client.generate",
        lambda messages, options=None: '{"route":"SQL_ENGINE","schema":{"operation":"sum","columns":["revenue"]}}',
    )

    result = agent.run({
        "query": "and by region",
        "dataset_profile": "Columns: revenue (float), region (str)",
        "semantic_summary": "Revenue data",
        "text_heavy": False,
        "history": [{"user": "What is the total revenue", "assistant": "The total revenue is $1M"}],
    })

    assert result["route"] == "SQL_ENGINE"
    assert result["schema"]["operation"] == "sum"


def test_non_follow_up_query_uses_current_query_only(monkeypatch):
    agent = RouterAgent()

    monkeypatch.setattr(
        "app.agents.router.llm_client.generate",
        lambda messages, options=None: '{"route":"SQL_ENGINE","schema":{"operation":"avg","columns":["salary"]}}',
    )

    result = agent.run({
        "query": "average salary",
        "dataset_profile": "Columns: salary (float)",
        "semantic_summary": "Salary data",
        "text_heavy": False,
        "history": [{"user": "What is the total revenue", "assistant": "..."}],
    })

    assert result["route"] == "SQL_ENGINE"
    assert result["schema"]["operation"] == "avg"


def test_router_uses_history_service_with_chat_id(monkeypatch):
    agent = RouterAgent()

    monkeypatch.setattr(
        "app.agents.router.get_history",
        lambda chat_id: [{"user": "What is the total revenue", "assistant": "..."}] if chat_id == "chat-1" else [],
    )
    
    monkeypatch.setattr(
        "app.agents.router.llm_client.generate",
        lambda messages, options=None: '{"route":"SQL_ENGINE","schema":{"operation":"sum","columns":["revenue"]}}',
    )

    result = agent.run({
        "query": "and by region",
        "dataset_profile": "Columns: revenue (float), region (str)",
        "semantic_summary": "Revenue data",
        "text_heavy": False,
        "chat_id": "chat-1",
    })

    assert result["route"] == "SQL_ENGINE"
    assert result["schema"]["operation"] == "sum"


def test_router_returns_enriched_schema_for_sql_route(monkeypatch):
    agent = RouterAgent()

    monkeypatch.setattr(
        "app.agents.router.llm_client.generate",
        lambda messages, options=None: '{"route":"SQL_ENGINE","schema":{"operation":"sum","columns":["revenue"],"group_by":["region"],"sql_plan":{"group_by":["region"]}}}',
    )

    result = agent.run({
        "query": "total revenue by region",
        "dataset_profile": "Columns: revenue (float), region (str)",
        "semantic_summary": "Revenue info",
        "text_heavy": False,
        "history": [],
    })

    assert result["route"] == "SQL_ENGINE"
    assert result["schema"]["operation"] == "sum"
    assert "region" in str(result["schema"]["group_by"])
    assert "sql_plan" in result["schema"]
    assert "region" in str(result["schema"]["sql_plan"]["group_by"])


def test_router_promotes_to_sql_engine_for_complex_aggregation(monkeypatch):
    agent = RouterAgent()

    monkeypatch.setattr(
        "app.agents.router.llm_client.generate",
        lambda messages, options=None: '{"route":"SQL_ENGINE","schema":{"operation":"sum","columns":["sales"],"engine_mode":"sql","filters":[{"column":"sales","operator":">","value":1000},{"column":"profit","operator":">","value":100}]}}',
    )

    result = agent.run({
        "query": 'total sales by region where "sales" > 1000 and "profit" > 100',
        "dataset_profile": "Columns: sales (float), profit (float), region (str)",
        "semantic_summary": "Financial data",
        "text_heavy": False,
        "history": [],
    })

    assert result["route"] == "SQL_ENGINE"
    assert result["schema"]["engine_mode"] == "sql"
    assert len(result["schema"]["filters"]) >= 2


def test_router_returns_semantic_plan_for_text_engine_queries(monkeypatch):
    agent = RouterAgent()

    monkeypatch.setattr(
        "app.agents.router.llm_client.generate",
        lambda messages, options=None: '{"route":"TEXT_TABLE_RAG","schema":{"operation":"semantic","semantic_plan":{"query_text":"find rows similar to customer complaining about delays"}}}',
    )

    result = agent.run({
        "query": "find rows similar to customer complaining about delays",
        "dataset_profile": "Columns: feedback (str)",
        "semantic_summary": "Customer feedback",
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
        "dataset_profile": "Columns: sales (float), region (str)",
        "semantic_summary": "Sales data",
        "text_heavy": False,
        "history": [],
    })

    assert result["route"] == "SQL_ENGINE"
    assert result["schema"]["columns"] == ["sales"]
    assert result["schema"]["filters"][0]["column"] == "region"
    assert "region" in str(result["schema"]["group_by"])
    assert result["schema"]["limit"] == 5


def test_router_refuses_when_sql_intent_is_incomplete(monkeypatch):
    agent = RouterAgent()

    monkeypatch.setattr(
        "app.agents.router.llm_client.generate",
        lambda messages, options=None: '{"route":"SQL_ENGINE","schema":{"operation":"sum","columns":[],"filters":[]}}',
    )

    result = agent.run({
        "query": "compute this metric for me",
        "dataset_profile": "Columns: metric1 (float)",
        "semantic_summary": "Metrics data",
        "text_heavy": False,
        "history": [],
    })

    assert result["route"] == "REFUSE"
    assert "follow_up_questions" in result["schema"]
    assert any("numeric column" in question.lower() for question in result["schema"]["follow_up_questions"])


def test_router_prompt_instructs_model_to_preserve_sheet_scope():
    agent = RouterAgent()
    prompt = agent._build_llm_prompt(
        query="count the number of males in Sheet1",
        dataset_profile="Columns: Gender (str)",
        semantic_summary="Workbook data",
        history_text="",
        text_heavy=False,
    )

    assert 'column "sheet"' in prompt
    assert '{"column": "sheet", "operator": "=", "value": "Sheet1"}' in prompt
    assert "schema.sql_plan.filters" in prompt


def test_router_prompt_instructs_model_to_preserve_semantic_sheet_scope():
    agent = RouterAgent()
    prompt = agent._build_llm_prompt(
        query="find comments about delays in Sheet2",
        dataset_profile="Columns: notes (str), sheet (str)",
        semantic_summary="Workbook feedback data",
        history_text="",
        text_heavy=True,
    )

    assert "semantic_plan.post_filters" in prompt
    assert '{"column": "sheet", "operator": "=", "value": "Sheet1"}' in prompt


def test_router_prompt_instructs_model_to_resolve_follow_up_scope_from_recent_queries():
    agent = RouterAgent()
    prompt = agent._build_llm_prompt(
        query="what is the average age in this?",
        dataset_profile="Columns: Name (str), Age (int), Notes (str)",
        semantic_summary="Employee records with free-text notes",
        history_text="User: find rows where the text mentions Chasse",
        text_heavy=True,
    )

    assert 'uses references like "this", "these", "those"' in prompt
    assert 'preserve the prior semantic scope of rows mentioning "Chasse"' in prompt
    assert 'interpret it as applying the aggregation to the rows scoped by the prior query' in prompt


def test_router_refuses_when_shared_column_exists_in_multiple_sheets_without_sheet_filter(monkeypatch):
    agent = RouterAgent()

    monkeypatch.setattr(
        "app.agents.router.llm_client.generate",
        lambda messages, options=None: (
            '{"route":"SQL_ENGINE","schema":{"operation":"count","columns":["Gender"],'
            '"filters":[{"column":"Gender","operator":"=","value":"Male"}]}}'
        ),
    )

    result = agent.run({
        "query": "count the number of males",
        "dataset_profile": "\n".join(
            [
                "Workbook with 3 sheets",
                "Total rows: 8, Max columns :3",
                "Sheets:",
                "Sheet1: rows = 3, cols=3, columns=[Gender,Age,Dept]",
                "Sheet2: rows = 3, cols=3, columns=[Gender,Salary,Dept]",
                "Sheet3: rows = 2, cols=2, columns=[Region,Sales]",
                "Columns (union): Gender,Age,Dept,Salary,Region,Sales",
            ]
        ),
        "semantic_summary": "Workbook data",
        "text_heavy": False,
        "history": [],
    })

    assert result["route"] == "REFUSE"
    assert any("which sheet should i use" in q.lower() for q in result["schema"]["follow_up_questions"])


def test_router_allows_shared_column_when_sheet_filter_is_present(monkeypatch):
    agent = RouterAgent()

    monkeypatch.setattr(
        "app.agents.router.llm_client.generate",
        lambda messages, options=None: (
            '{"route":"SQL_ENGINE","schema":{"operation":"count","columns":["Gender"],'
            '"filters":[{"column":"sheet","operator":"=","value":"Sheet1"},'
            '{"column":"Gender","operator":"=","value":"Male"}]}}'
        ),
    )

    result = agent.run({
        "query": "count the number of males in Sheet1",
        "dataset_profile": "\n".join(
            [
                "Workbook with 3 sheets",
                "Total rows: 8, Max columns :3",
                "Sheets:",
                "Sheet1: rows = 3, cols=3, columns=[Gender,Age,Dept]",
                "Sheet2: rows = 3, cols=3, columns=[Gender,Salary,Dept]",
                "Sheet3: rows = 2, cols=2, columns=[Region,Sales]",
                "Columns (union): Gender,Age,Dept,Salary,Region,Sales",
            ]
        ),
        "semantic_summary": "Workbook data",
        "text_heavy": False,
        "history": [],
    })

    assert result["route"] == "SQL_ENGINE"
    assert any(f["column"] == "sheet" for f in result["schema"]["filters"])
