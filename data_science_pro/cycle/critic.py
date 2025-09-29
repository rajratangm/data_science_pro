from typing import Dict, Any

from data_science_pro.api.llm_connector import LLMConnector


class Critic:
    """Quality assurance node that inspects current state and flags issues."""

    def __init__(self, api_key: str):
        self.llm = LLMConnector(api_key)

    def __call__(self, state: Dict[str, Any]):
        ctx = {
            "analysis": state.get("analysis"),
            "preprocessing": next((h for h in state.get("history", []) if h.get("action") == "preprocess"), None),
            "model": next((h for h in state.get("history", []) if h.get("action") == "train"), None),
            "evaluation": state.get("evaluation"),
            "target_metric": state.get("target_metric"),
        }
        critique = self.llm.run(
            """
            Review the current pipeline state for quality issues:
            - Data leakage risks
            - Target encoding mistakes
            - Evaluation methodology problems
            - Overfitting/underfitting signals
            Provide 3-5 bullets with concrete fixes.
            """,
            context=ctx
        )
        state["critique"] = critique
        history = state.get("history", [])
        history.append({"action": "critique"})
        state["history"] = history
        return state


