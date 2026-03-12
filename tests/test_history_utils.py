import pytest
from app.utils.history_utils import truncate_history


def test_truncate_history_empty():
    """Test with empty history."""
    result = truncate_history([])
    assert result == ""


def test_truncate_history_single_turn():
    """Test with a single conversation turn."""
    history = [
        {"user": "What is the total sales?", "assistant": "The total sales is $1000"}
    ]
    result = truncate_history(history)
    assert result == "User: What is the total sales?\nAssistant: The total sales is $1000"


def test_truncate_history_multiple_turns():
    """Test with multiple conversation turns."""
    history = [
        {"user": "Show me the data", "assistant": "Here is the data..."},
        {"user": "What is the average?", "assistant": "The average is 50"},
        {"user": "Group by region", "assistant": "Grouped by region..."},
    ]
    result = truncate_history(history)
    expected = (
        "User: Show me the data\n"
        "Assistant: Here is the data...\n"
        "User: What is the average?\n"
        "Assistant: The average is 50\n"
        "User: Group by region\n"
        "Assistant: Grouped by region..."
    )
    assert result == expected


def test_truncate_history_max_turns():
    """Test that only last 5 turns are kept."""
    history = [
        {"user": "Query 1", "assistant": "Response 1"},
        {"user": "Query 2", "assistant": "Response 2"},
        {"user": "Query 3", "assistant": "Response 3"},
        {"user": "Query 4", "assistant": "Response 4"},
        {"user": "Query 5", "assistant": "Response 5"},
        {"user": "Query 6", "assistant": "Response 6"},
        {"user": "Query 7", "assistant": "Response 7"},
    ]
    result = truncate_history(history, max_user_turns=5)
    expected = (
        "User: Query 3\nAssistant: Response 3\n"
        "User: Query 4\nAssistant: Response 4\n"
        "User: Query 5\nAssistant: Response 5\n"
        "User: Query 6\nAssistant: Response 6\n"
        "User: Query 7\nAssistant: Response 7"
    )
    assert result == expected


def test_truncate_history_custom_max():
    """Test with custom max_user_turns."""
    history = [
        {"user": "Query 1", "assistant": "Response 1"},
        {"user": "Query 2", "assistant": "Response 2"},
        {"user": "Query 3", "assistant": "Response 3"},
    ]
    result = truncate_history(history, max_user_turns=2)
    expected = "User: Query 2\nAssistant: Response 2\nUser: Query 3\nAssistant: Response 3"
    assert result == expected


def test_truncate_history_missing_user_key():
    """Test with missing user keys."""
    history = [
        {"user": "Query 1", "assistant": "Response 1"},
        {"assistant": "Response 2"},  # Missing user key
        {"user": "Query 3", "assistant": "Response 3"},
    ]
    result = truncate_history(history)
    expected = (
        "User: Query 1\nAssistant: Response 1\n"
        "Assistant: Response 2\n"
        "User: Query 3\nAssistant: Response 3"
    )
    assert result == expected


def test_truncate_history_empty_user_values():
    """Test with empty user values."""
    history = [
        {"user": "Query 1", "assistant": "Response 1"},
        {"user": "", "assistant": "Response 2"},  # Empty user value
        {"user": "Query 3", "assistant": "Response 3"},
    ]
    result = truncate_history(history)
    expected = (
        "User: Query 1\nAssistant: Response 1\n"
        "Assistant: Response 2\n"
        "User: Query 3\nAssistant: Response 3"
    )
    assert result == expected


def test_truncate_history_keeps_latest_assistant_for_short_follow_up_context():
    history = [
        {"user": "Find rows where the text mentions Chasse", "assistant": "Which sheet should I use?"},
        {"user": "Sheet1", "assistant": "Using Sheet1."},
    ]

    result = truncate_history(history)

    expected = (
        "User: Find rows where the text mentions Chasse\n"
        "Assistant: Which sheet should I use?\n"
        "User: Sheet1\n"
        "Assistant: Using Sheet1."
    )
    assert result == expected


def test_truncate_history_truncates_latest_assistant_to_200_characters():
    long_response = "A" * 210
    history = [
        {"user": "Show me the rows", "assistant": long_response},
    ]

    result = truncate_history(history)

    expected = f"User: Show me the rows\nAssistant: {'A' * 197}..."
    assert result == expected
