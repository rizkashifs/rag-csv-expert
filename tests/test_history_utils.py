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
    assert result == "1. What is the total sales?"


def test_truncate_history_multiple_turns():
    """Test with multiple conversation turns."""
    history = [
        {"user": "Show me the data", "assistant": "Here is the data..."},
        {"user": "What is the average?", "assistant": "The average is 50"},
        {"user": "Group by region", "assistant": "Grouped by region..."},
    ]
    result = truncate_history(history)
    expected = "1. Show me the data\n2. What is the average?\n3. Group by region"
    assert result == expected


def test_truncate_history_max_turns():
    """Test that only last 5 user prompts are kept."""
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
    expected = "1. Query 3\n2. Query 4\n3. Query 5\n4. Query 6\n5. Query 7"
    assert result == expected


def test_truncate_history_custom_max():
    """Test with custom max_user_turns."""
    history = [
        {"user": "Query 1", "assistant": "Response 1"},
        {"user": "Query 2", "assistant": "Response 2"},
        {"user": "Query 3", "assistant": "Response 3"},
    ]
    result = truncate_history(history, max_user_turns=2)
    expected = "1. Query 2\n2. Query 3"
    assert result == expected


def test_truncate_history_missing_user_key():
    """Test with missing user keys."""
    history = [
        {"user": "Query 1", "assistant": "Response 1"},
        {"assistant": "Response 2"},  # Missing user key
        {"user": "Query 3", "assistant": "Response 3"},
    ]
    result = truncate_history(history)
    expected = "1. Query 1\n2. Query 3"
    assert result == expected


def test_truncate_history_empty_user_values():
    """Test with empty user values."""
    history = [
        {"user": "Query 1", "assistant": "Response 1"},
        {"user": "", "assistant": "Response 2"},  # Empty user value
        {"user": "Query 3", "assistant": "Response 3"},
    ]
    result = truncate_history(history)
    expected = "1. Query 1\n2. Query 3"
    assert result == expected
