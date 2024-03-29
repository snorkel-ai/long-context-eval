# Long Context Evaluation

## Benchmark for testing long context windows on your own data

### Tests

- Single Document QA
    - [ ] Retrieval accuracy versus document depth
    - [ ] Retrieval accuracy: Long context versus RAG accuracy
    - [ ] Hallucination indicator: When the document is not present in context


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

```zsh
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
```

```zsh
python3 long_context_eval/run_benchmark.py --data_path ./data --model gpt-3.5-turbo
```


### Test details

TBD

### To-do
- [ ] Multiprocessing while running test
- [ ] Get a suitable baseline RAG pipeline
- [ ] Results and visualization