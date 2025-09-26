from api import llm_connector
from langchain.memory import ConversationBufferMemory

class Suggester:
    def suggest_hyperparams(self, analyzer_result, model_name, user_query=None):
        prompt = f"Given the analysis: {analyzer_result}, and model: {model_name}, suggest optimal hyperparameters. {user_query if user_query else ''}"
        response = self.llm.generate_response(prompt)
        return response
    def __init__(self, api_key=None, memory=None):
        if not api_key:
            raise ValueError("API key must be provided to Suggester.")
        self.llm = llm_connector.LLMConnector(api_key)
        self.memory = memory or ConversationBufferMemory()
        self.agent = None
        self.tools = []
        self._setup_agent()

    def _setup_agent(self):
        from langchain.agents import initialize_agent, Tool
        from langchain_community.llms import OpenAI  # Updated import

        def suggest_features_tool(prompt):
            return self.llm.generate_response(prompt)

        def suggest_models_tool(prompt):
            return self.llm.generate_response(prompt)

        def oversample_tool(prompt):
            return self.llm.generate_response(prompt)

        self.tools = [
            Tool(
                name="SuggestFeatures",
                func=suggest_features_tool,
                description="Suggest relevant features based on analysis, user query, and metrics."
            ),
            Tool(
                name="SuggestModels",
                func=suggest_models_tool,
                description="Suggest suitable models based on analysis, user query, and metrics."
            ),
            Tool(
                name="Oversample",
                func=oversample_tool,
                description="Suggest if oversampling is needed based on analysis and metrics."
            )
        ]
        self.agent = initialize_agent(
            self.tools,
            OpenAI(openai_api_key=self.llm.api_key),  # Pass API key here
            agent="zero-shot-react-description",
            memory=self.memory,
            verbose=True
        )

    def suggest_next_action(self, analyzer_result, user_query, metrics):
        # Truncate analyzer_result to avoid exceeding model context length
        summary = str(analyzer_result)[:1000]  # Use only first 1000 characters
        prompt = (
            f"You are an expert AI data scientist helping to automate a data science pipeline.\n"
            f"Context: The pipeline cycles through EDA, preprocessing, model selection, training, and testing.\n"
            f"Analysis summary: {summary}\n"
            f"User goal or query: {user_query}\n"
            f"Current metrics or requirements: {metrics}\n"
            "Based on the above, suggest the most effective next action (preprocessing, feature engineering, model selection, etc.), and briefly explain why. If relevant, recommend specific techniques or parameters."
        )
        return self.agent.run({"input": prompt})

    def suggest_features(self, analyzer_result, user_query, metrics):
        prompt = f"Based on the following analysis: {analyzer_result}, and user query: {user_query}, suggest relevant features to consider for the metrics: {metrics}."
        response = self.llm.generate_response(prompt)
        return response

    def suggest_models(self, analyzer_result, user_query, metrics):
        prompt = f"Based on the following analysis: {analyzer_result}, and user query: {user_query}, suggest suitable models to consider for the metrics: {metrics}."
        response = self.llm.generate_response(prompt)
        return response