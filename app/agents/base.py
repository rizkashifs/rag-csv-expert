from abc import ABC, abstractmethod
from typing import Any

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    Ensures all agents have a single 'run' method.
    """
    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """
        Execute the agent's primary responsibility.
        """
        pass
