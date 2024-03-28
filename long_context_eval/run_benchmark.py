from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional
from jsonargparse import CLI

from long_context_eval.evals.single_hop_qa import SingleHopQATest

load_dotenv()

@dataclass
class Settings:
    model_name: Optional[str] = "gpt-3.5-turbo"
    data_path: Optional[str] = "./data"
    
    


def main():
    args = CLI(Settings, as_positional=False)

    # evaluate single hop doc QA
    lctest = SingleHopQATest(**args.__dict__)
    lctest.test_position_single_hop()

