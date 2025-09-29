from typing import Dict, Any

from data_science_pro.api.llm_connector import LLMConnector
from data_science_pro.utils.prompts import ANALYZER_PROMPT, FEATURE_ENGINEER_PROMPT, TRAINER_PROMPT, MODEL_SELECTOR_PROMPT


class Planner:
    """High-level plan generator breaking the goal into actionable steps."""

    def __init__(self, api_key: str):
        self.llm = LLMConnector(api_key)

    def __call__(self, state: Dict[str, Any]):
        goal = state.get("user_query", "")
        analysis = state.get("analysis", {})
        evaluation = state.get("evaluation", {})
        context = {
            "goal": goal,
            "analysis": analysis,
            "evaluation": evaluation,
            "meta": state.get("meta")
        }
        plan = self.llm.run(
            """
            Create a concise action plan for the pipeline:
            - EDA focus areas (based on ANALYZER_PROMPT)
            - Feature engineering ideas (based on FEATURE_ENGINEER_PROMPT)
            - Candidate models (based on MODEL_SELECTOR_PROMPT)
            - Training/evaluation approach (based on TRAINER_PROMPT)
            Return as 4 bullet points.
            """,
            context=context
        )
        state["plan"] = plan
        history = state.get("history", [])
        history.append({"action": "plan"})
        state["history"] = history
        return state


