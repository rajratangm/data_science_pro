from langchain_core.prompts import ChatPromptTemplate
from data_science_pro.api.llm_connector import LLMConnector
from data_science_pro.utils.prompts import SUGGESTER_PROMPT


class Suggester:
    """Base suggester for pipeline insights."""
    def __init__(self, api_key: str):
        self.llm = LLMConnector(api_key)

    def __call__(self, state: dict):
        dyn_context = {
            "goal": state.get("user_query"),
            "meta": state.get("meta"),
            "analysis": state.get("analysis"),
            "retrieved_context": state.get("retrieved_context"),
            "history": state.get("history", []),
            "evaluation": state.get("evaluation"),
            "iteration": state.get("iteration"),
        }
        suggestion = self.suggest(dyn_context)
        # Track action history for downstream prompts
        history = state.get("history", [])
        history.append({"action": "suggest", "iteration": state.get("iteration", 0)})
        state["history"] = history
        state["suggestions"] = suggestion
        return state

    def suggest(self, context: dict) -> str:
        """Default suggestion using LLM with dynamic context."""
        return self.llm.run(SUGGESTER_PROMPT, context=context)


class ChainOfThoughtSuggester(Suggester):
    """Specialized suggester with step-by-step reasoning."""

    def suggest(self, context: str) -> str:
        prompt = ChatPromptTemplate.from_template(
            "Think step by step about the following context:\n"
            "{context}\n\n"
            "1. Analyze the problem.\n"
            "2. Suggest preprocessing or modeling strategies.\n"
            "3. Recommend best next step.\n"
        )
        return self.llm.run(prompt.format(context=context))
