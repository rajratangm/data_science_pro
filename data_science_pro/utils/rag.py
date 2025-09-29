import os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions


def ensure_persist_dir(persist_directory: str) -> str:
    os.makedirs(persist_directory, exist_ok=True)
    return persist_directory


def _is_api_key_valid(api_key: Optional[str]) -> bool:
    if not api_key:
        return False
    val = api_key.strip().lower()
    return val not in {"", "test", "dummy", "none"}


def get_collection(collection_name: str = "csv_knowledge", persist_directory: str = ".rag_store", api_key: Optional[str] = None, embedding_model: str = "text-embedding-3-large"):
    persist_directory = ensure_persist_dir(persist_directory)
    client = chromadb.PersistentClient(path=persist_directory)
    # In tests or offline mode, skip embedding function when api_key is missing/invalid
    if _is_api_key_valid(api_key):
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=embedding_model,
        )
        return client.get_or_create_collection(name=collection_name, embedding_function=ef)
    return client.get_or_create_collection(name=collection_name)


def build_documents_from_analysis(analysis: Dict[str, Any], preview_rows: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    missing_values = analysis.get("missing_values", {})
    unique_counts = analysis.get("unique_counts", {})
    target_candidates = analysis.get("target_candidates", [])

    for col, uniq in unique_counts.items():
        mv = missing_values.get(col, 0)
        text = (
            f"Column: {col}. Unique values: {uniq}. Missing values: {mv}. "
            f"Target candidate: {'yes' if col in target_candidates else 'no'}."
        )
        if preview_rows:
            examples = []
            for r in preview_rows[:3]:
                if col in r:
                    examples.append(str(r[col]))
            if examples:
                text += f" Examples: {', '.join(examples)}."
        documents.append({
            "id": f"col::{col}",
            "text": text,
            "metadata": {"type": "column_summary", "column": col}
        })

    overview = (
        "Dataset overview. Columns: " + ", ".join(unique_counts.keys()) + ". "
        + f"Potential targets: {', '.join(target_candidates)}."
    )
    documents.append({
        "id": "dataset::overview",
        "text": overview,
        "metadata": {"type": "dataset_overview"}
    })

    return documents


def upsert_documents(documents: List[Dict[str, Any]], collection_name: str = "csv_knowledge", persist_directory: str = ".rag_store", api_key: Optional[str] = None):
    collection = get_collection(collection_name, persist_directory, api_key=api_key)
    ids = [d["id"] for d in documents]
    metadatas = [d.get("metadata", {}) for d in documents]
    texts = [d["text"] for d in documents]
    # Chroma upsert via add (idempotency based on same ids will append; for simplicity, delete existing ids first)
    try:
        collection.delete(ids=ids)
    except Exception:
        pass
    collection.add(ids=ids, metadatas=metadatas, documents=texts)
    return {"collection_name": collection_name, "persist_directory": persist_directory}


def query_relevant(query: str, n_results: int = 5, collection_name: str = "csv_knowledge", persist_directory: str = ".rag_store", api_key: Optional[str] = None) -> List[str]:
    collection = get_collection(collection_name, persist_directory, api_key=api_key)
    result = collection.query(query_texts=[query], n_results=n_results)
    docs = result.get("documents", [[]])[0]
    return docs


