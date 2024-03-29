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

To run on your own data, add documents to a `data` folder. (As a default, we use `Unstructured` to extract text from documents. If you have PDF/HTML/etc. documents, install the necessary `Unstructured` libraries.)

If there is no data folder, the benchmark will first create 100 documents from `HuggingFaceTB/cosmopedia-100k` dataset (format=wikihow). The dataset contains synthetically generated articles.

We follow LangChain's model naming conventions.

```zsh
python3 long_context_eval/run_benchmark.py --data_path ./data --model gpt-3.5-turbo
```


### Test details

The process of testing long context is as follows:
1. QA pairs are first generated for each document at `data_path`. Currently only 1 QA pair is generated per document using GPT-3.5 Turbo. QA pairs are saved at `./data.json`.
2. In order to test the model's context window capability, we fills the model's context window by selecting documents and related QA pairs (sequentially) until the token limit is reached.
3. For each document depth, the document containing the answer is positioned at the desired depth in the context, and an LLM response is generated.
4. Model responses are evaluated using LLM-as-a-judge.
5. For the RAG test, Langchain's pipeline using Chroma DB (with default parameters) is used. 10 documents are retrieved and used as context.


### To-do
- [ ] Multiprocessing while running test
- [ ] Get a suitable baseline RAG pipeline
- [ ] Log time
- [ ] Results and visualization (table/charts)