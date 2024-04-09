import os
import json
from typing import Optional
from jsonargparse import CLI
import tiktoken
from datasets import load_dataset
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers.json import SimpleJsonOutputParser


class Title(BaseModel):
    title: str = Field(description="title to the article")


def create_datastore(data_path,
                     hfdataset: Optional[str] = "HuggingFaceTB/cosmopedia-100k",
                     hfdatasetsplit: Optional[str] = "train",
                     hfdatasetfilterdictkey: Optional[str] = "format",
                     hfdatasetfilterdictvalue: Optional[str] = "wiki",
                     hfdatasettextcol: Optional[str] = "text",
                     hfdataset_num_docs: Optional[int] = 100,
                     data_generation_model_name: Optional[str] = "gpt-3.5-turbo",
                     data_generation_model_kwargs: Optional[dict] = dict(temperature=0.8),
                     seed: Optional[int] = 42):
    '''Creates documents from a Hugging Face dataset if ./data folder is empty'''
    print(f"Creating ({hfdataset_num_docs}) documents in {data_path} from {hfdataset} with {data_generation_model_name}")
    ds = load_dataset(hfdataset, split=hfdatasetsplit)
    sample_dataset = ds.filter(lambda example: example[hfdatasetfilterdictkey].startswith(hfdatasetfilterdictvalue))
    sample_dataset = sample_dataset.shuffle(seed=seed).select(range(hfdataset_num_docs))

    # Define the model, prompt and format for chain
    api_key = os.getenv('OPENAI_API_KEY')
    model = ChatOpenAI(model=data_generation_model_name,
                            openai_api_key=api_key,
                            **data_generation_model_kwargs)
    
    # print # of tokens from this dataset
    long_context = "\n\n".join(sample_dataset[hfdatasettextcol])
    encoding = tiktoken.encoding_for_model(data_generation_model_name)
    print("Total # of tokens: ", len(encoding.encode(long_context)))

    #check how many samples we need
    full_text = "\n \n".join(sample_dataset[hfdatasettextcol])
    print("total # of tokens", model.get_num_tokens(full_text))

    title_prompt_str = """Return a JSON object with a `title` for the following text. Do not use any special characters in the title.
    Text: {text}
    JSON: """
    title_prompt = PromptTemplate(template=title_prompt_str, input_variables=["text"])

    parser_title = SimpleJsonOutputParser(pydantic_object=Title)

    # generate a title for the document using the LLM
    chain = title_prompt | model | parser_title

    def get_title(example, chain, hfdatasettextcol):
        try:
            file_title = chain.invoke({"text": example[hfdatasettextcol]})["title"]
        except:
            file_title = example[hfdatasettextcol][:50]
        return {"file_title": file_title}

    # sample_dataset = sample_dataset.map(lambda example: {"file_title": 
    #                                                      chain.invoke(
    #                                                          {"text": example[hfdatasettextcol]})["title"]})
    sample_dataset = sample_dataset.map(lambda example: get_title(example, chain=chain, hfdatasettextcol=hfdatasettextcol))

    #create folder qa_data under ./ using os library and check if it exists
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # write article to a json file.
    for idx, value in enumerate(sample_dataset):
        title = value["file_title"] + ".txt"
        doc_content = value[hfdatasettextcol]
        with open(os.path.join(data_path, title), 'w') as f:
            f.write(json.dumps(doc_content))

@dataclass
class Settings:
    data_path: Optional[str] = "./data"
    hfdataset: Optional[str] = "HuggingFaceTB/cosmopedia-100k"
    hfdatasetsplit: Optional[str] = "train"
    hfdatasetfilterdictkey: Optional[str] = "format"
    hfdatasetfilterdictvalue: Optional[str] = "wiki"
    hfdatasettextcol: Optional[str] = 'text'
    hfdataset_num_docs: Optional[int] = 100
    data_generation_model_name: Optional[str] = "gpt-3.5-turbo"
    data_generation_model_kwargs: Optional[dict] = field(default_factory=lambda: dict(temperature=0.8))


if __name__ == '__main__':
    args = CLI(Settings, as_positional=False)
    create_datastore(**args.__dict__)