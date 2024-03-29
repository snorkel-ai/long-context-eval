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
    model_kwargs: Optional[dict] = field(default_factory=lambda: dict(temperature=0.8))
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200
    search_kwargs: Optional[dict] = field(default_factory=lambda: dict(k=10))
    embedding_model_name: Optional[str] = 'text-embedding-ada-002'
    embedding_model_kwargs: Optional[dict] = field(default_factory=lambda: dict())


def main():
    args = CLI(Settings, as_positional=False)

    # evaluate single hop doc QA
    lctest = SingleHopQATest(**args.__dict__)
    lctest.test_position_single_hop()
    lctest.test_rag()


if __name__ == '__main__':
    main()
