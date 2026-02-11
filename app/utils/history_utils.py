from typing import List, Dict, Any


def truncate_history(history: List[Dict[str, Any]], max_user_turns: int = 5) -> str:
    """
    Truncate and format chat history for LLM context.
    
    Args:
        history: List of conversation turns with 'user' and 'assistant' keys
        max_user_turns: Maximum number of user turns to keep (default: 5)
    
    Returns:
        Formatted history string with only user prompts from the last N turns
    """
    if not history:
        return ""
    
    # Extract only user prompts from the last N turns
    user_prompts = []
    for turn in reversed(history):
        user_query = turn.get("user", "").strip()
        if user_query:
            user_prompts.append(user_query)
            if len(user_prompts) >= max_user_turns:
                break
    
    # Reverse to maintain chronological order
    user_prompts.reverse()
    
    # Format as numbered list
    if not user_prompts:
        return ""
    
    formatted_lines = [f"{i+1}. {prompt}" for i, prompt in enumerate(user_prompts)]
    return "\n".join(formatted_lines)
