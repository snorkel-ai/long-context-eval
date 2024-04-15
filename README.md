# Long Context Evaluation

## Benchmark for testing long context windows on your own data

### Tests

- Single Document QA
    - [ ] Retrieval accuracy versus document depth
    - [ ] Retrieval accuracy: Long context versus RAG accuracy
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


We use the LangChain library for running models. Set up the appropriate API keys and environment variables to access/run models. See LangChain's documentation.

```zsh
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
```

```zsh
python long_context_eval/run_benchmark.py --model_name gpt-3.5-turbo \
                                          --data_path ./data \
                                          --task_path ./data.json \
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
2. In order to test the model's context window capability, we fill the model's context window by selecting documents at random from `data_path` and corresponding QA pairs until the token limit is reached.
3. For each document depth, the document containing the answer is positioned at the desired depth in the context, and an LLM response is generated. Currently we test at 0, 25, 50, 75, and 100% depths.
4. For the RAG test, Langchain's pipeline using Chroma DB (with default parameters) is used to retrieve documents and used as context to generate the response.
5. Model responses are evaluated using LLM-as-a-judge (we recommend manually verifying the results, since auto-evals are known to have errors.)


### To-do
- [X] Test for retrieval scoring at different document depths
- [X] Test for RAG
- [X] Add support for Claude, Google
- [ ] Add support for OSS models
- [ ] Multiprocessing and async while running test
- [ ] Results and visualization (table/charts)
- [ ] Script to run for multiple models
- [ ] Add decorator for logging time
- [ ] Add intermediate depth results (so it doesn't rerun in case of any failure)
- [ ] Test for / handle edge cases
