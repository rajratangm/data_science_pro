class DataAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def __call__(self, state: dict):
        df = state["data"]
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        categorical_cols = [c for c in df.columns if c not in numeric_cols]
        basic_stats = {}
        try:
            basic_stats = df[numeric_cols].describe().to_dict() if numeric_cols else {}
        except Exception:
            basic_stats = {}
        missing_values = df.isna().sum().to_dict()
        unique_counts = df.nunique().to_dict()
        target_candidates = [col for col in df.columns if df[col].nunique() <= 20]
        corr_top = {}
        try:
            corr = df[numeric_cols].corr(numeric_only=True)
            # Extract top absolute correlations (upper triangle)
            pairs = (
                corr.where(~corr.isna())
                .abs()
                .unstack()
                .sort_values(ascending=False)
            )
            corr_top = {str(k): float(v) for k, v in list(pairs.items())[:20]}
        except Exception:
            corr_top = {}
        analysis = {
            "missing_values": missing_values,
            "unique_counts": unique_counts,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "basic_stats": basic_stats,
            "top_correlations": corr_top,
            "target_candidates": target_candidates,
        }
        state["analysis"] = analysis
        # Track
        history = state.get("history", [])
        history.append({"action": "analyze", "n_numeric": len(numeric_cols), "n_categorical": len(categorical_cols)})
        state["history"] = history
        return state
