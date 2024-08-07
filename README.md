# Snorkel Working Memory Test (SWiM)

This is the repo for our paper "Evaluating Language Model Context Windows: A "Working Memory" Test and Inference-time Correction" by Amanda Dsouza, Christopher Glaze, Changho Shin, Frederic Sala.

arXiv: https://arxiv.org/abs/2407.03651

blog post: https://snorkel.ai/long-context-models-in-the-enterprise-benchmarks-and-beyond/

## Overview

This repository provides a Snorkel Working Memory Test (SWiM) to evaluate the long context capabilities of large language models (LLMs) on your own data and tasks. This is an improvement to the "needle in a haystack" (NIAH) test, where the haystack is your own set of documents, and the needles are one or more answer (complete) documents based on which the question is posed.

This is important, as current methods of long context evaluation are either synthetic and unrealistic (such as the NIAH test) or limited to academic datasets (such as [LongBench](https://arxiv.org/abs/2308.14508), [InfiniteBench](https://arxiv.org/abs/2402.13718) and others) which renders them less useful in real world settings. 

SWiM overcomes these limitations by (a) creating realistic tests (distractor QA or question answering based on information contained in one or more documents over a long context) and (b) enabling users to evaluate long context capabilities on their own data and tasks. 

This is done through an LLM-driven task generation >> task validation >> task completion >> task evaluation pipeline. We strongly recommend manually verifying both the inputs (tasks) and outputs (scores). A more detailed description of the methodology is available in our paper.

We also propose a novel approach to mitigate the "lost-in-the-middle" approach using medoid voting.


<p align="center">
  <img src="images/framework.png" width=512px>
</p>


### Supporting tests and tasks

- Effect of position on retrieval accuracy
    - [X] Single Document QA with distractors task (a.k.a single needle in a haystack test)
    - [ ] Multi Document QA with distractors task
- Effect of context size on long-context versus RAG accuracy
    - [X] Single Document QA with distractors task
    - [ ] Multi Document QA with distractors task
- Hallucination Index: Extent to which the model hallucinates when the document is not present in context
    - [ ] Single Document QA with distractors task


## Installation

```zsh
python3 -m venv venv
source venv/bin/activate
```

```zsh
git clone git@github.com:snorkel-ai/long-context-eval.git
cd long-context-eval
pip install -r requirements.txt
```

See instructions on data and task formats [here](./docs/DATA.md)


We use LangChain for running models. Check LangChain's documentation in order to set up the appropriate API keys and environment variables to access/run models. Currently the following model providers are implemented: `OpenAI`, `Anthropic`, `TogetherAI` and `VertexAI`. See `models.py` for the list of models currently supported. A new model provider and/or model can be easily added in `models.py`. Please be aware of the costs associated with using the API's when running the tests.

```zsh
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
```

```zsh
python long_context_eval/run_benchmark.py --model_name gpt-3.5-turbo \
                                          --data_path ./data/cosmowikidataset \
                                          --task_path ./tasks/data_cosmowiki.json \
                                          --eval_model_name gpt-4 \
                                          --experiment_tag QAtest \
                                          --tests position \
                                          --seed 42 \
                                          --document_depth_percents_list "[50]"
```

For a full list of arguments, run
```zsh
python long_context_eval/run_benchmark.py --help
```

## Experiments and Results
Here are some results using the SWiM test. The full set of results and analysis is described in our paper.


<p align="center">
  <img src="images/swim_versus_niah.png" width=512px>
</p>


<p align="center">
  <img src="images/lcm_test.png" width=512px>
</p>

# 
If you find this work useful to your research, please consider citing our paper
```
@misc{dsouza2024evaluatinglanguagemodelcontext,
      title={Evaluating Language Model Context Windows: A "Working Memory" Test and Inference-time Correction}, 
      author={Amanda Dsouza and Christopher Glaze and Changho Shin and Frederic Sala},
      year={2024},
      eprint={2407.03651},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.03651}, 
}
```