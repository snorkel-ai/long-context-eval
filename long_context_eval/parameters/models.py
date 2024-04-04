import os
from typing import Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


MAX_CONTEXT_SIZE = {"gpt-3.5-turbo": 16385,
                    "gpt-3.5-turbo-16k": 16385,
                    "gpt-4": 8192,
                    "gpt-4-0125-preview": 128000}


class BaseModel:
    def __init__(self,
                 model_name: str,
                 model_kwargs: Optional[dict],
                 ):
        self.model_name = model_name
        self.model_kwargs = model_kwargs


### OpenAI Models
class OpenAIModel(BaseModel):
    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 model_kwargs: dict = dict(temperature=0.8),
                 ):
        super().__init__(model_name, model_kwargs)
        api_key = os.getenv('OPENAI_API_KEY')
        self.model = ChatOpenAI(model=model_name,
                                openai_api_key=api_key,
                                **model_kwargs)
        self.max_context_size = MAX_CONTEXT_SIZE[model_name]


class OpenAIEmbeddingsModel(BaseModel):
    def __init__(self,
                 model_name: str = "text-embedding-ada-002",
                 model_kwargs: dict = {},
                 ):
        super().__init__(model_name, model_kwargs)
        api_key = os.getenv('OPENAI_API_KEY')
        self.model = OpenAIEmbeddings(model=model_name,
                                openai_api_key=api_key,
                                **model_kwargs)


SUPPORTED_MODELS = {"gpt-3.5-turbo": OpenAIModel,
                    "gpt-3.5-turbo-16k": OpenAIModel,
                    "gpt-4": OpenAIModel,
                    "gpt-4-0125-preview": OpenAIModel,
                    "text-embedding-ada-002": OpenAIEmbeddingsModel}