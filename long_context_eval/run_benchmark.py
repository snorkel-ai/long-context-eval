import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Optional, Literal
from jsonargparse import CLI

from evals.single_hop_qa import SingleHopQATest

load_dotenv()

@dataclass
class Settings:
    model_name: Optional[str] = "gpt-3.5-turbo"
    data_path: Optional[str] = "./data"
    task_path: Optional[str] = "./data.json"
    results_folder_path: Optional[str] = "./results"
    model_kwargs: Optional[dict] = field(default_factory=lambda: dict(temperature=0.7))
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200
    search_kwargs: Optional[dict] = field(default_factory=lambda: dict(k=4))
    embedding_model_name: Optional[str] = 'text-embedding-ada-002'
    embedding_model_kwargs: Optional[dict] = field(default_factory=lambda: dict())
    eval_model_name: Optional[str] = "gpt-4-turbo-2024-04-09"
    eval_model_kwargs: Optional[dict] = field(default_factory=lambda: dict(temperature=0))
    experiment_tag: Optional[str] = "tag"
    log_path: Optional[str] = "experiments.log"
    data_gen_prompt: Optional[str] = "single_doc_question_gen_prompt"
    task_prompt: Optional[str] = "single_doc_qa_prompt"
    eval_prompt: Optional[str] = "score_qa_prompt"
    seed: Optional[int] = None
    tests: Optional[Literal['all', 'position', 'rag', 'medoid', 'control_medoid']] = 'all'
    document_depth_percents_list: Optional[list] = field(default_factory=lambda: list((0, 25, 50, 75, 100)))
    percent_ctxt_window_used: Optional[list] = field(default_factory=lambda: list((0, 25, 50, 75, 100)))
    num_runs_medoid_vote: Optional[int] = 1
    document_depth_percents_medoid: Optional[int] = 25

def main():
    cliargs = CLI(Settings, as_positional=False)
    args = cliargs.__dict__
    tests = args["tests"]

    # evaluate single hop doc QA
    lctest = SingleHopQATest(**args)

    if tests == "all":
        lctest.test_position_accuracy()
        lctest.test_long_context_length_versus_rag()
    elif tests == "position":
        lctest.test_position_accuracy()
    elif tests == "rag":
        lctest.test_long_context_length_versus_rag()
    elif tests == "medoid":
        lctest.test_medoid_voting()
    elif tests == "control_medoid":
        lctest.test_control_medoid_voting()
    else:
        exit(0)
    logging.basicConfig(filename=cliargs.log_path,level=logging.DEBUG)
    logging.info(lctest)


if __name__ == '__main__':
    main()
