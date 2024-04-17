# Long Context Evaluation

## Evaluating long context model capabilities on your own data

This repository provides a framework to evaluate the long context capabilities of large language models (LLMs) on your own data and tasks. This is similar to the "needle in a haystack" test, except the haystack is your own set of documents, and the needles are a task (for example, a Question-Answer pair) that is created from the documents. This follows an automated task generation >> task completion >> task evaluation process, enabled by LLMs. We strongly recommend manually verifying both the inputs (tasks) and outputs (scores).


### Tests

- Single Document QA
    - [X] Retrieval accuracy versus document depth
    - [X] Retrieval accuracy: Long context versus RAG accuracy
    - [ ] Hallucination indicator: When the document is not present in context
- Multi Document QA
    - [ ] TBD


### Running the benchmark

```zsh
python3 -m venv venv
source venv/bin/activate
```

```zsh
git clone git@github.com:snorkel-ai/long-context-eval.git
cd long-context-eval
pip install -r requirements.txt
```

See instructions on the data and task formats [here](DATA.md)


We use the LangChain library for running models. Set up the appropriate API keys and environment variables to access/run models. See LangChain's documentation. For a list of models that you can test on, see `models.py`. A new model can be easily added in `models.py`. 

```zsh
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
```

```zsh
python long_context_eval/run_benchmark.py --model_name gpt-3.5-turbo \
                                          --data_path ./data/cosmowikidataset \
                                          --task_path ./long_context_eval/tasks/data_cosmowiki.json \
                                          --experiment_tag QAtest \
                                          --seed 42

```

For a full list of arguments, run
```zsh
python long_context_eval/run_benchmark.py --help
```


### Test details

The process of testing long context is as follows:

1. If no task json is provided for the QA task at `task_path`, QA pairs are first generated for each document at `data_path`. Currently only 1 QA pair is generated per document using GPT-4 and QA pairs are saved in `./data.json`.
2. In order to test the model's context window capability, we fill the model's context window by selecting documents at random from `data_path` and corresponding QA pairs until the token limit is reached. We use `tiktoken` to count tokens, so this is approximate for models that do not use the same tokenization.
3. For each document depth, the document containing the answer is positioned at the desired depth in the context, and an LLM response is generated. Currently we test at 0, 25, 50, 75, and 100% depths.
4. For the long context versus RAG test, starting with just the test document, the size of the context window is increased by adding distractor documents. For RAG, Langchain's pipeline using Chroma DB (with default parameters) is used to retrieve documents and used as context to generate the response. Currently we test at 0, 25, 50, 75 and 100% of context size.
5. Model responses are evaluated using LLM-as-a-judge (we recommend manually verifying the results, since auto-evals are known to have errors.)
