import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Optional
from jsonargparse import CLI

from evals.single_hop_qa import SingleHopQATest

load_dotenv()

@dataclass
class Settings:
    model_name: Optional[str] = "gpt-3.5-turbo"
    data_path: Optional[str] = "./data"
    qa_pairs_path: Optional[str] = "./data.json"
    model_kwargs: Optional[dict] = field(default_factory=lambda: dict(temperature=0.8))
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200
    search_kwargs: Optional[dict] = field(default_factory=lambda: dict(k=10))
    embedding_model_name: Optional[str] = 'text-embedding-ada-002'
    embedding_model_kwargs: Optional[dict] = field(default_factory=lambda: dict())
    hfdataset: Optional[str] = "HuggingFaceTB/cosmopedia-100k"
    hfdatasetsplit: Optional[str] = "train"
    hfdatasetfilterdictkey: Optional[str] = "format"
    hfdatasetfilterdictvalue: Optional[str] = "wiki"
    hfdatasettextcol: Optional[str] = 'text'
    hfdataset_num_docs: Optional[int] = 100
    data_generation_model_name: Optional[str] = "gpt-3.5-turbo"
    data_generation_model_kwargs: Optional[dict] = field(default_factory=lambda: dict(temperature=0.8))
    eval_model_name: Optional[str] = "gpt-3.5-turbo"
    eval_model_kwargs: Optional[dict] = field(default_factory=lambda: dict(temperature=0))
    experiment_tag: Optional[str] = "tag"
    log_path: Optional[str] = "experiments.log"
    data_gen_prompt: Optional[str] = "single_doc_question_gen_prompt"
    task_prompt: Optional[str] = "single_doc_qa_prompt"
    eval_prompt: Optional[str] = "score_qa_prompt"
    seed: Optional[int] = None
    num_docs: Optional[int] = None


def main():
    args = CLI(Settings, as_positional=False)

    # evaluate single hop doc QA
    lctest = SingleHopQATest(**args.__dict__)

    lctest.test_position_accuracy()
    logging.basicConfig(filename=args.log_path,level=logging.DEBUG)
    logging.info(lctest)

    
    

    # lctest.test_rag()


if __name__ == '__main__':
    main()
