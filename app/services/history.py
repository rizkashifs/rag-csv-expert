from collections import defaultdict
from typing import Dict, List


class HistoryService:
    """
    In-memory chat history store used to provide conversational context.
    """

    def __init__(self):
        self._history: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    def get_history(self, chat_id: str) -> List[Dict[str, str]]:
        """
        Returns list of conversation turns with user/assistant messages.
        """
        if not chat_id:
            return []
        return self._history.get(chat_id, [])

    def add_turn(self, chat_id: str, user: str, assistant: str) -> None:
        if not chat_id:
            return
        self._history[chat_id].append({"user": user, "assistant": assistant})


history_service = HistoryService()


def get_history(chat_id: str) -> List[Dict[str, str]]:
    """
    Convenience module function required by routing flow.
    """
    return history_service.get_history(chat_id)

