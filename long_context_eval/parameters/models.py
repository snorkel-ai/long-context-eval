import os
from typing import Optional
import tiktoken
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_vertexai import VertexAI
from langchain_anthropic import ChatAnthropic
from langchain_together import Together


MAX_CONTEXT_SIZE = {"gpt-3.5-turbo": 16385,
                    "gpt-3.5-turbo-16k": 16385, # Currently points to gpt-3.5-turbo-16k-0613.
                    "gpt-4": 8192,
                    "gpt-4-0125-preview": 128_000,
                    "gpt-4-turbo-2024-04-09": 128_000,
                    "gemini-pro": 32_760,
                    "gemini-1.5-pro-preview-0409": 500_000,
                    "claude-2.1": 150_000,  # Claude is truncated currently since we use GPT-4 tokenizer to count tokens
                    "claude-3-opus-20240229": 150_000,
                    "databricks/dbrx-instruct": 32_000,
                    "mistralai/Mixtral-8x7B-Instruct-v0.1": 25_000,
                    "microsoft/WizardLM-2-8x22B": 55_000,
                    }


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
        self.encoding = tiktoken.encoding_for_model("gpt-4")


class AnthropicModel(BaseModel):
    def __init__(self, 
                 model_name: str,
                 model_kwargs: dict = dict(temperature=0.8),):
        super().__init__(model_name)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if "temperature" in model_kwargs:
            self.model = ChatAnthropic(model=model_name,
                                    anthropic_api_key=api_key,
                                    temperature=model_kwargs["temperature"])
        else:
            self.model = ChatAnthropic(model=model_name,
                                    anthropic_api_key=api_key)
        self.max_context_size = MAX_CONTEXT_SIZE[model_name]
        # To get the tokeniser corresponding to a specific model
        self.encoding = tiktoken.encoding_for_model("gpt-4")


class TogetherAPIModel(BaseModel):
    def __init__(self, 
                 model_name: str,
                 model_kwargs: dict = dict(temperature=0.8)):
        super().__init__(model_name)
        api_key = os.getenv("TOGETHER_API_KEY")
        self.model = Together(model=model_name,
                                together_api_key=api_key,
                                **model_kwargs)
        self.max_context_size = MAX_CONTEXT_SIZE[model_name]
        # To get the tokeniser corresponding to a specific model
        self.encoding = tiktoken.encoding_for_model("gpt-4")


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
                    "gpt-4-turbo-2024-04-09": OpenAIModel,
                    "gemini-pro": VertexAIModel,
                    "gemini-1.5-pro-preview-0409": VertexAIModel,
                    "claude-2.1": AnthropicModel,
                    "claude-3-opus-20240229": AnthropicModel,
                    "databricks/dbrx-instruct": TogetherAPIModel,
                    "mistralai/Mixtral-8x7B-Instruct-v0.1": TogetherAPIModel,
                    "microsoft/WizardLM-2-8x22B": TogetherAPIModel,
                    "text-embedding-ada-002": OpenAIEmbeddingsModel,
                    }