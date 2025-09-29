from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class Trainer:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def __call__(self, state: dict):
        df = state["data"]
        # Pick target from state to avoid encoded target issues
        target = state.get("target") or state["analysis"]["target_candidates"][0]
        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
        )
        model = LogisticRegression(solver='liblinear', max_iter=1000)
        model.fit(X_train, y_train)

        state["model"] = model
        state["split"] = (X_train, X_test, y_train, y_test)
        # Track history
        history = state.get("history", [])
        history.append({"action": "train", "model": "LogisticRegression", "samples": len(y_train)})
        state["history"] = history
        return state
