# from langchain.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

class LLMConnector:
    load_dotenv()
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        self.llm = ChatOpenAI(openai_api_key=self.api_key, model_name=self.model_name, temperature=self.temperature)

    def generate_response(self, prompt):
        # For chat models, pass a list of messages
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        return response.content if hasattr(response, "content") else response