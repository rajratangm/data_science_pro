from typing import Dict

from data_science_pro.utils.rag import query_relevant


class Retriever:
    def __init__(self, api_key: str, collection_name: str = "csv_knowledge", persist_directory: str = ".rag_store"):
        self.api_key = api_key
        self.collection_name = collection_name
        self.persist_directory = persist_directory

    def __call__(self, state: Dict):
        # Build a richer query using goal, suggestions, and analysis
        analysis = state.get("analysis", {})
        goal = state.get("user_query", "")
        suggestions = state.get("suggestions", "")
        query = (
            f"Goal: {goal}. Columns: {', '.join(analysis.get('unique_counts', {}).keys())}. "
            f"Targets: {', '.join(analysis.get('target_candidates', []))}. Missing: {str(analysis.get('missing_values', {}) )}. "
            f"Latest suggestions: {str(suggestions)[:300]}"
        )
        docs = query_relevant(
            query=query,
            n_results=5,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            api_key=self.api_key,
        )
        state["retrieved_context"] = "\n\n".join(docs) if docs else ""
        return state


