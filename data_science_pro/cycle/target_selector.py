from typing import Dict, Any

from data_science_pro.api.llm_connector import LLMConnector


class TargetSelector:
    """Chooses the target column using (priority): user input > LLM based on goal and EDA > heuristic."""

    def __init__(self, api_key: str):
        self.llm = LLMConnector(api_key)

    def __call__(self, state: Dict[str, Any]):
        # 1) If user provided a target, use it when present in data
        user_target = state.get("user_target")
        if user_target and user_target in state["data"].columns:
            state["target"] = user_target
            return state

        # 2) Use LLM to pick the target consistent with goal when possible
        candidates = state.get("analysis", {}).get("target_candidates", [])
        if candidates:
            try:
                decision = self.llm.run(
                    """
                    Based on the dataset columns and the user's goal, pick the best target variable from the candidates.
                    Return only the column name exactly.
                    """,
                    context={
                        "goal": state.get("user_query", ""),
                        "candidates": candidates,
                        "columns": state.get("meta", {}).get("columns", []),
                    },
                )
                chosen = decision.strip()
                if chosen in state["data"].columns:
                    state["target"] = chosen
                    return state
            except Exception:
                pass

            # 3) Fallback heuristic: first candidate
            state["target"] = candidates[0]
        return state


