import os
from typing import Optional
import tiktoken
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_vertexai import VertexAI
from langchain_anthropic import ChatAnthropic


MAX_CONTEXT_SIZE = {"gpt-3.5-turbo": 16385,
                    "gpt-3.5-turbo-16k": 16385,
                    "gpt-4": 8192,
                    "gpt-4-0125-preview": 128000,
                    "gemini-1.5-pro": 128000}


class BaseModel:
    def __init__(self,
                 model_name: str,
                 ):
        self.model_name = model_name


### OpenAI Models
class OpenAIModel(BaseModel):
    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 model_kwargs: dict = dict(temperature=0.8),
                 ):
        super().__init__(model_name)
        api_key = os.getenv('OPENAI_API_KEY')
        self.model = ChatOpenAI(model=model_name,
                                openai_api_key=api_key,
                                **model_kwargs)
        self.max_context_size = MAX_CONTEXT_SIZE[model_name]
        # To get the tokeniser corresponding to a specific model in the OpenAI API:
        self.encoding = tiktoken.encoding_for_model(model_name)

class VertexAIModel(BaseModel):
    def __init__(self,
                 model_name: str = "gemini-pro",
                 model_kwargs: dict = dict(temperature=0.8),
                 ):
        super().__init__(model_name)
        self.model = VertexAI(model_name=model_name,
                              generationConfig=model_kwargs)
        self.max_context_size = MAX_CONTEXT_SIZE[model_name]
        # To get the tokeniser corresponding to a specific model
        self.encoding = tiktoken.encoding_for_model(model_name)


class AnthropicModel(BaseModel):
    def __init__(self, 
                 model_name: str,
                 model_kwargs: dict = dict(temperature=0.8),):
        super().__init__(model_name)
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if "temperature" in model_kwargs:
            self.model = ChatAnthropic(model=model_name,
                                    anthropic_api_key=anthropic_api_key,
                                    temperature=model_kwargs["temperature"])
        else:
            self.model = ChatAnthropic(model=model_name,
                                    anthropic_api_key=anthropic_api_key)
        self.max_context_size = MAX_CONTEXT_SIZE[model_name]
        # To get the tokeniser corresponding to a specific model
        self.encoding = tiktoken.encoding_for_model(model_name)


class OpenAIEmbeddingsModel(BaseModel):
    def __init__(self,
                 model_name: str = "text-embedding-ada-002",
                 model_kwargs: dict = {},
                 ):
        super().__init__(model_name)
        api_key = os.getenv('OPENAI_API_KEY')
        self.model = OpenAIEmbeddings(model=model_name,
                                openai_api_key=api_key,
                                **model_kwargs)


SUPPORTED_MODELS = {"gpt-3.5-turbo": OpenAIModel, #16k context
                    "gpt-3.5-turbo-16k": OpenAIModel, # gpt-3.5-turbo-16k-0613, 16k context
                    "gpt-4": OpenAIModel, # 8k context
                    "gpt-4-0125-preview": OpenAIModel, #128k context
                    "gemini-1.5-pro": VertexAIModel,
                    "text-embedding-ada-002": OpenAIEmbeddingsModel
                    }