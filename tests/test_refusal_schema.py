from app.agents.refusal import RefusalAgent


def test_refusal_agent_returns_standard_refusal_schema():
    agent = RefusalAgent()
    payload = agent.run(
        {
            "schema_context": "Columns: amount, date",
            "route_schema": {"follow_up_questions": ["Which column?", "Which year?"]},
        }
    )

    assert "relevant_rows" in payload
    assert isinstance(payload["relevant_rows"], list)
    assert payload["relevant_rows"][0]["should_ask_user"] is True
    assert "_summary" in payload["relevant_rows"][0]
