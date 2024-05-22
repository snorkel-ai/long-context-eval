import os
import json
from typing import Optional
from jsonargparse import CLI
from collections import defaultdict
from dataclasses import dataclass, field
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import VertexAI
from langchain.schema import HumanMessage
from langchain.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.output_parsers import JsonOutputParser


###### Output formats.
class SingleDocQuestionGen(BaseModel):
    question: str = Field(description="question to ask")
    answer: str = Field(description="answer to the question")

PROMPT_TEMPLATES = {
    "single_doc_question_gen_prompt": """Ask a factoid question given only the context provided, that can be answered in a few words. Answer the question given the context.
    Format your result in a JSON object with keys 'question' and 'answer'.
    Context: {context}
    Result:""",}

MAX_CONTEXT_SIZE = {"gpt-3.5-turbo": 16385,
                    "gpt-3.5-turbo-16k": 16385, # Currently points to gpt-3.5-turbo-16k-0613.
                    "gpt-4": 8192,
                    "gpt-4-0125-preview": 128_000,
                    "gpt-4-turbo-2024-04-09": 128_000,
                    "gemini-pro": 32_760,
                    "gemini-1.5-pro-preview-0409": 500_000,
                    "gemini-1.5-flash-preview-0514": 2_000_000,
                    "claude-2.1": 150_000,  # Claude is truncated currently since we use GPT-4 tokenizer to count tokens
                    "claude-3-opus-20240229": 150_000, #truncated currently since we use GPT-4 tokenizer to count tokens
                    "databricks/dbrx-instruct": 32_000, #truncated currently since we use GPT-4 tokenizer to count tokens
                    "mistralai/Mixtral-8x7B-Instruct-v0.1": 25_000, #truncated currently since we use GPT-4 tokenizer to count tokens
                    "microsoft/WizardLM-2-8x22B": 55_000, #truncated currently since we use GPT-4 tokenizer to count tokens
                    }

def create_qa_pairs_single_hop(data_path, 
                               qa_pairs_path,
                               model_provider: Optional[str] = "openai",
                               model_name: Optional[str] = "gpt-4",
                            ):
    '''Creating benchmark questions-answers from individual documents'''

    if not os.path.exists(data_path) or not os.listdir(data_path):
            print("No documents for running experiments.")
            exit(0)

    # load files
    print(f"Loading documents. at {data_path}..")
    loader = DirectoryLoader(data_path, glob="**/*.*",
                            show_progress=True,
                            use_multithreading=True,)
    documents = loader.load()
    print("# of loaded documents: ", len(documents))

    # define prompt, model and format
    prompt = PromptTemplate(template=PROMPT_TEMPLATES["single_doc_question_gen_prompt"],
                                           input_variables=["context"])
    format = JsonOutputParser(pydantic_object=SingleDocQuestionGen) 

    if model_provider == "google":
        chat_model = VertexAI(model_name=model_name,)
    else:
        api_key = os.getenv('OPENAI_API_KEY')
        chat_model = ChatOpenAI(model=model_name,
                                openai_api_key=api_key,
                            )
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except:
        encoding = tiktoken.encoding_for_model("gpt-4")
    max_context_size = MAX_CONTEXT_SIZE[model_name]
    
    # iterate through docs to generate QA pairs
    qa_pairs = defaultdict(dict)
    for idx, doc in enumerate(documents):
        print(f"Processing document {idx} of {len(documents)}")
        print(idx, doc.metadata["source"])

        chain = prompt | chat_model | format
        
        num_token = chat_model.get_num_tokens_from_messages(messages=[
        HumanMessage(content=prompt.format(context=doc.page_content))
    ])
        if num_token > max_context_size:
            doc_content = encoding.decode(encoding.encode(
                prompt.format(
                    context=doc.page_content))[:max_context_size])
        else:
            doc_content = doc.page_content

        try:
            _, filename = os.path.split(doc.metadata["source"])
            qa = chain.invoke({"context": doc_content})
            qa_pairs[idx] = {"question": qa["question"], "answer": qa["answer"],
                             "answer_doc": filename}
        except:
                print(f"Error creating QA pair for document {idx}")
                continue

    # Writes result to one single QA test file.
    with open(qa_pairs_path, 'w') as json_file:
        json.dump(qa_pairs, json_file)

    return qa_pairs

@dataclass
class Settings:
    data_path: str
    qa_pairs_path: str
    model_provider: Optional[str] = "openai"
    model_name: Optional[str] = "gpt-4"


if __name__ == '__main__':
    args = CLI(Settings, as_positional=False)
    create_qa_pairs_single_hop(**args.__dict__)