def feature_suggestion_prompt(analyzer_result, user_query, metrics, history=None):
    return f"""
    You are an expert data scientist. Given the following dataset analysis:
    {analyzer_result}
    User query: {user_query}
    Target metrics: {metrics}
    Previous actions: {history if history else 'None'}
    - List the most relevant features to improve the target metrics.
    - Justify each feature suggestion based on the data stats, correlations, and user goals.
    - If feature engineering is needed, suggest new features and explain why.
    """

def model_suggestion_prompt(analyzer_result, user_query, metrics, history=None):
    return f"""
    You are an AI model selection assistant. Given:
    Dataset analysis: {analyzer_result}
    User query: {user_query}
    Target metrics: {metrics}
    Previous actions: {history if history else 'None'}
    - Recommend the best ML models for this scenario.
    - Explain your choices using dataset properties (size, imbalance, feature types).
    - Suggest hyperparameters or tuning strategies if relevant.
    """

def oversample_prompt(analyzer_result, user_query, metrics, history=None):
    return f"""
    You are a data balancing expert. Given:
    Dataset analysis: {analyzer_result}
    User query: {user_query}
    Target metrics: {metrics}
    Previous actions: {history if history else 'None'}
    - Should oversampling or class balancing be applied? Why or why not?
    - If yes, recommend the best technique and parameters.
    """
