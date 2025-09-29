import pandas as pd

class DataLoader:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def __call__(self, state: dict):
        csv_path = state["csv_path"]
        df = pd.read_csv(csv_path)
        state["data"] = df
        state["meta"] = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict()
        }
        # Initialize history for dynamic prompts
        state["history"] = [{"action": "load", "rows": df.shape[0], "cols": df.shape[1]}]
        return state
