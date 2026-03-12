from typing import List, Dict, Any


def _truncate_text(text: str, max_length: int = 200) -> str:
    text = text.strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def truncate_history(history: List[Dict[str, Any]], max_user_turns: int = 5) -> str:
    """
    Truncate and format chat history for LLM context.
    
    Args:
        history: List of conversation turns with 'user' and 'assistant' keys
        max_user_turns: Maximum number of user turns to keep (default: 5)
    
    Returns:
        Formatted history string with the last N conversation turns.
        Assistant messages are truncated to 200 characters.
    """
    if not history:
        return ""

    recent_turns = history[-max_user_turns:]
    formatted_lines = []

    for turn in recent_turns:
        user_query = turn.get("user", "").strip()
        assistant_text = turn.get("assistant", "").strip()

        if user_query:
            formatted_lines.append(f"User: {user_query}")
        if assistant_text:
            formatted_lines.append(
                f"Assistant: {_truncate_text(assistant_text, max_length=200)}"
            )

    if not formatted_lines:
        return ""

    return "\n".join(formatted_lines)
