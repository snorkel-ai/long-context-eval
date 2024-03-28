import os
import json
from datasets import load_dataset
from langchain.output_parsers.json import SimpleJsonOutputParser
from parameters.formats import Title
from parameters.prompts import TITLE_PROMPT
from parameters.models import OpenAIModel


def create_datastore(data_path):
    '''If ./data folder is empty, extract 100 docs from wikihow in cosmopedia dataset'''
    num_docs = 100
    format = "wiki"
    ds = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")
    sample_dataset = ds.filter(lambda example: example["format"].startswith(format))
    sample_dataset = sample_dataset.shuffle().select(range(num_docs))

    chat_model = OpenAIModel("gpt-3.5-turbo")
    parser_title = SimpleJsonOutputParser(pydantic_object=Title)

    chain = TITLE_PROMPT | chat_model.model | parser_title
    sample_dataset = sample_dataset.map(lambda example: {"title": 
                                                         chain.invoke(
                                                             {"text": example["text"]})["title"]})

    #create folder qa_data under ./ using os library and check if it exists
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # write wikihow article to a json file.
    for idx, value in enumerate(sample_dataset):
        title = value["title"] + ".txt"
        doc_content = value["text"]
        with open(os.path.join(data_path, title), 'w') as f:
            f.write(json.dumps(doc_content))