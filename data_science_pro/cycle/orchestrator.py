from typing import Dict, Any

from data_science_pro.api.llm_connector import LLMConnector
from data_science_pro.utils.prompts import ORCHESTRATOR_PROMPT


class Orchestrator:
    def __init__(self, api_key: str):
        self.llm = LLMConnector(api_key)

    def __call__(self, state: Dict[str, Any]):
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 5)
        target_metric = state.get("target_metric", "accuracy")
        target_value = state.get("target_value", 0.85)
        last_eval = state.get("evaluation", {})
        current_value = 0.0
        if isinstance(last_eval, dict):
            current_value = float(last_eval.get(target_metric, 0.0))

        # Hard stop if reached goal or exceeded iterations
        if current_value >= target_value:
            state["next_action"] = "report"
            return state
        if iteration >= max_iterations:
            state["next_action"] = "report"
            return state

        # Build compact context for LLM routing
        context = {
            "goal": state.get("user_query", ""),
            "iteration": iteration,
            "max_iterations": max_iterations,
            "target_metric": target_metric,
            "target_value": target_value,
            "current_value": current_value,
            "analysis": state.get("analysis", {}),
            "suggestions": state.get("suggestions", ""),
            "data_meta": state.get("meta", {}),
        }

        try:
            decision = self.llm.run(ORCHESTRATOR_PROMPT, context=context)
        except Exception:
            # Fallback heuristic routing
            decision = "preprocess" if iteration == 0 else "train"

        normalized = self._normalize_decision(decision)
        state["next_action"] = normalized
        # Increment iteration when we are in improvement loop steps
        if normalized in {"preprocess", "train", "evaluate"}:
            state["iteration"] = iteration + 1
        return state

    def _normalize_decision(self, decision_text: str) -> str:
        d = decision_text.lower()
        if "report" in d or "finish" in d or "stop" in d:
            return "report"
        if "evaluate" in d or "evaluation" in d:
            return "evaluate"
        if "train" in d or "fit" in d:
            return "train"
        if "preprocess" in d or "clean" in d or "feature" in d:
            return "preprocess"
        if "analy" in d or "eda" in d:
            return "analyze"
        return "preprocess"


