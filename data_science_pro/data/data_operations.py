import pandas as pd

class DataOperations:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def __call__(self, state: dict):
        df = state["data"]
        target = None
        # Preserve target column from encoding
        if "target" in state and state["target"] in df.columns:
            target = state["target"]
        elif "analysis" in state:
            # Fallback to first candidate
            candidates = state["analysis"].get("target_candidates", [])
            target = candidates[0] if candidates else None

        # Apply preprocessing (could be chosen interactively via LangGraph memory)
        df = df.dropna(axis=0, how="any")
        features = df.drop(columns=[target]) if target and target in df.columns else df
        features = pd.get_dummies(features, drop_first=True)
        if target and target in df.columns:
            df = pd.concat([features, df[[target]]], axis=1)
        else:
            df = features

        state["data"] = df
        if target:
            state["target"] = target
        # Track history
        history = state.get("history", [])
        history.append({"action": "preprocess", "dropped_na": True, "encoded": True})
        state["history"] = history
        return state
