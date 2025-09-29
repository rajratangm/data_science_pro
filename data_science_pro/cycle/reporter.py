from data_science_pro.api.llm_connector import LLMConnector
from data_science_pro.utils.prompts import REPORTER_PROMPT

class Reporter:
    def __init__(self, api_key: str):
        self.llm = LLMConnector(api_key)

    def __call__(self, state: dict):
        context = {
            "goal": state.get("user_query"),
            "meta": state.get("meta"),
            "analysis": state.get("analysis"),
            "retrieved_context": state.get("retrieved_context"),
            "history": state.get("history", []),
            "suggestions": state.get("suggestions"),
            "evaluation": state.get("evaluation"),
            "target_metric": state.get("target_metric"),
            "target_value": state.get("target_value"),
        }
        report = self.llm.run(REPORTER_PROMPT, context=context)
        state["report"] = report
        return state
