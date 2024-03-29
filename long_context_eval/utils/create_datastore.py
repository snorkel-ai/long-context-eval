import os
import json
from typing import Optional
from datasets import load_dataset
from langchain.output_parsers.json import SimpleJsonOutputParser
from parameters.formats import Title
from parameters.prompts import TITLE_PROMPT
from parameters.models import OpenAIModel


def create_datastore(data_path,
                     hfdataset: Optional[str] = "HuggingFaceTB/cosmopedia-100k",
                     hfdatasetsplit: Optional[str] = "train",
                     hfdatasetfilterdictkey: Optional[str] = "format",
                     hfdatasetfilterdictvalue: Optional[str] = "wiki",
                     hfdatasettextcol: Optional[str] = "text",
                     hfdataset_num_docs: Optional[int] = 100,):
    '''If ./data folder is empty, extract docs Hugging Face dataset'''
    print(f"Creating ({hfdataset_num_docs}) documents in {data_path} from {hfdataset}")
    ds = load_dataset(hfdataset, split=hfdatasetsplit)
    sample_dataset = ds.filter(lambda example: example[hfdatasetfilterdictkey].startswith(hfdatasetfilterdictvalue))
    sample_dataset = sample_dataset.shuffle().select(range(hfdataset_num_docs))

    chat_model = OpenAIModel("gpt-3.5-turbo")
    parser_title = SimpleJsonOutputParser(pydantic_object=Title)

    chain = TITLE_PROMPT | chat_model.model | parser_title
    sample_dataset = sample_dataset.map(lambda example: {"file_title": 
                                                         chain.invoke(
                                                             {"text": example[hfdatasettextcol]})["title"]})

    #create folder qa_data under ./ using os library and check if it exists
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # write wikihow article to a json file.
    for idx, value in enumerate(sample_dataset):
        title = value["file_title"] + ".txt"
        doc_content = value[hfdatasettextcol]
        with open(os.path.join(data_path, title), 'w') as f:
            f.write(json.dumps(doc_content))