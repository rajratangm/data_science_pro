from langchain_openai import ChatOpenAI
import time

class LLMConnector:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(api_key=api_key, model=model)

    def run(self, prompt: str, context: dict = None, retries: int = 2, backoff_sec: float = 1.0):
        """
        Run an LLM call with optional context (like dataset metadata).
        """
        full_prompt = prompt
        if context:
            full_prompt += f"\n\nContext:\n{context}"
        last_err = None
        for attempt in range(retries + 1):
            try:
                response = self.llm.invoke(full_prompt)
                return response.content
            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(backoff_sec * (2 ** attempt))
                else:
                    raise last_err
