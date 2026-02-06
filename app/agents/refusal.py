from app.agents.base import BaseAgent


class RefusalAgent(BaseAgent):
    """
    Dedicated agent that returns clarification prompts when confidence is low.
    """

    def run(self, input_data: dict) -> str:
        schema_context = input_data.get("schema_context", "")
        return (
            "I need a bit more detail to answer that. "
            "Please clarify what you want to know (e.g., which column, metric, time period, or filter). "
            "Here is the dataset profile to help you choose:\n\n"
            f"{schema_context}"
        )
