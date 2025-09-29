from typing import Dict

from data_science_pro.utils.rag import build_documents_from_analysis, upsert_documents


class Indexer:
    def __init__(self, api_key: str, collection_name: str = "csv_knowledge", persist_directory: str = ".rag_store"):
        self.api_key = api_key
        self.collection_name = collection_name
        self.persist_directory = persist_directory

    def __call__(self, state: Dict):
        analysis = state.get("analysis", {})
        preview_rows = None
        if "data" in state:
            try:
                preview_rows = state["data"].head(5).to_dict(orient="records")
            except Exception:
                preview_rows = None
        docs = build_documents_from_analysis(analysis, preview_rows)
        upsert_documents(docs, collection_name=self.collection_name, persist_directory=self.persist_directory, api_key=self.api_key)
        state["rag_indexed"] = True
        return state


