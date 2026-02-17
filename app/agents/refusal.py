from typing import Any, Dict, List

from app.agents.base import BaseAgent


class RefusalAgent(BaseAgent):
    """Dedicated agent that returns clarification prompts when confidence is low."""

    def _build_summary(self, schema_context: str, follow_up_questions: List[str]) -> str:
        if follow_up_questions:
            formatted_questions = "\n".join([f"- {question}" for question in follow_up_questions[:4]])
            return (
                "I need a bit more detail before I can run this request accurately.\n"
                "Please clarify the following:\n"
                f"{formatted_questions}\n\n"
                "Here is the dataset profile to help you choose:\n\n"
                f"{schema_context}"
            )

        return (
            "I need a bit more detail to answer that. "
            "Please clarify what you want to know (e.g., which column, metric, time period, or filter). "
            "Here is the dataset profile to help you choose:\n\n"
            f"{schema_context}"
        )

    def run(self, input_data: dict) -> Dict[str, List[Dict[str, Any]]]:
        schema_context = input_data.get("schema_context", "")
        route_schema = input_data.get("route_schema", {}) or {}
        follow_up_questions = route_schema.get("follow_up_questions", [])

        summary = self._build_summary(schema_context, follow_up_questions)
        return {"relevant_rows": [{"_summary": summary, "should_ask_user": True}]}
