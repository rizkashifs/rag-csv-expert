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
