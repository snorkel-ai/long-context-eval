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

```zsh
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
```

To run on your own data, add documents to a `data` folder. (As a default, we use `Unstructured` to extract text from documents. If you have PDF/HTML/etc. documents, install the necessary `Unstructured` libraries.)

If there is no data folder, the benchmark will first create 100 documents from `HuggingFaceTB/cosmopedia-100k` dataset (format=wikihow). The dataset contains synthetically generated articles.

We follow LangChain's model naming conventions.

<pre style="background-color:black; color:white;">
<code>
(venv) $ python long_context_eval/run_benchmark.py --data_path ./data

Loading documents...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:02<00:00,  8.62it/s]
# of documents:  20
Truncated # of documents to 18 (context window token limits reached)
Run position test on long context for gpt-3.5-turbo
>>>>Generate llm responses at document depths...
Depth: 10
Depth: 20
Depth: 30
Depth: 40
Depth: 50
Depth: 60
Depth: 70
Depth: 80
Depth: 90
>>>>Evaluating responses using llm-as-a-judge
Accuracy at depth 10: 77.77%
Accuracy at depth 20: 72.22%
Accuracy at depth 30: 66.66%
Accuracy at depth 40: 72.22%
Accuracy at depth 50: 77.77%
Accuracy at depth 60: 77.77%
Accuracy at depth 70: 61.11%
Accuracy at depth 80: 77.77%
Accuracy at depth 90: 55.55%
Position Test Duration: 425.8 seconds
Results saved at ./output/position_test_results_gpt-3.5-turbo.json

Truncated # of documents to 18 (context window token limits reached)
Run RAG test for gpt-3.5-turbo
>>>>Chunk and add to vector store...
>>>>Generate llm responses...
>>>>Evaluating RAG responses using llm-as-a-judge
RAG Accuracy: 22.22%
RAG Test Duration: 22.5 seconds
Results saved at ./output/rag_test_results_gpt-3.5-turbo.json
</code>
</pre>


### Test details

The process of testing long context is as follows:

0. If no documents are provided (i.e. `data_path` does not exist, default `./data/`), 100 articles from `HuggingFaceTB/cosmopedia-100k` dataset (format=wikihow) are saved as documents at `data_path`.
1. QA pairs are first generated for each document at `data_path`. Currently only 1 QA pair is generated per document using GPT-3.5 Turbo. QA pairs are saved in `./data.json`.
2. In order to test the model's context window capability, we fill the model's context window by selecting documents and related QA pairs (sequentially) until the token limit is reached.
3. For each document depth, the document containing the answer is positioned at the desired depth in the context, and an LLM response is generated.
4. For the RAG test, Langchain's pipeline using Chroma DB (with default parameters) is used. 10 documents are retrieved and used as context.
5. Model responses are evaluated using LLM-as-a-judge.


### To-do
- [X] Test for retrieval scoring at different document depths
- [X] Basic test for RAG
- [ ] Multiprocessing and async while running test
- [ ] Add Google models
- [ ] Get a suitable baseline RAG pipeline
- [ ] Results and visualization (table/charts)
- [ ] TQDM progress bars
- [ ] Add decorator for logging time
- [ ] Add intermediate depth results (so it doesn't rerun in case of any failure)
- [ ] Test for / handle edge cases
- [ ] Script to run for multiple models