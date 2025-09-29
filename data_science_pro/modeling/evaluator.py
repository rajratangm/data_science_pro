from sklearn.metrics import accuracy_score, classification_report

class Evaluator:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def __call__(self, state: dict):
        model = state["model"]
        X_train, X_test, y_train, y_test = state["split"]

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        state["evaluation"] = {
            "accuracy": acc,
            "classification_report": report
        }
        # Track history
        history = state.get("history", [])
        history.append({"action": "evaluate", "accuracy": acc})
        state["history"] = history
        return state
