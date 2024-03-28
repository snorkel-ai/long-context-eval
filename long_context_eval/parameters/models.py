import os
from langchain_community.chat_models import ChatOpenAI


TOKEN_LIMITS = {"gpt-3.5-turbo": 16385}


class OpenAIModel:
    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 model_kwargs: dict = dict(temperature=0.8),
                 ):
        api_key = os.getenv('OPENAI_API_KEY')
        self.model = ChatOpenAI(model=model_name,
                                openai_api_key=api_key,
                                **self.model_kwargs)
        
        self.token_limit = TOKEN_LIMITS.get(model_name, None)


class GeminiModel:
    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 model_kwargs: dict = dict(temperature=0.8),
                 ):
        api_key = os.getenv('GOOGLE_API_KEY')
        self.model = None

        self.token_limit = TOKEN_LIMITS.get(model_name, None)

SUPPORTED_MODELS = {"gpt-3.5-turbo": OpenAIModel, "google": GeminiModel}