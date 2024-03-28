import os
import json
from random import shuffle
from typing import Optional
import tiktoken
from langchain.schema import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_community.document_loaders import DirectoryLoader
import long_context_eval.parameters.formats as formats
import long_context_eval.parameters.prompts as prompts
import long_context_eval.parameters.models as models
from long_context_eval.utils.create_datastore import create_datastore
from long_context_eval.utils.create_qa_pairs import create_qa_pairs_single_hop


class SingleHopQATest:
    """
    This class tests for long context.
    """
    def __init__(self,
                 model_name: Optional[str] = "gpt-3.5-turbo",
                 data_path: Optional[str] = "../data",
                 model_kwargs: Optional[dict] = dict(temperature=0.8)):
        self.model_name = model_name
        self.data_path = data_path
        # Get the correct model based on model name
        self.model = models.SUPPORTED_MODELS[self.model_name](model_kwargs)
        self.encoding = tiktoken.encoding_for_model(self.model_name) ## TBD: Update to get encoding for any provider model
        self.depths = []

    def _get_or_truncate_context_window(self, documents):
        '''fill up the context window, and truncate if necessary'''
        long_context = "\n\n".join([doc.page_content for doc in documents])
        if len(self.encoding.encode(long_context)) > self.model.token_limit:
            truncated_docs = []
            for doc in documents:
                truncated_docs.append(doc)
                if len(self.encoding.encode(
                    "\n\n".join(
                        [doc.page_content for doc in truncated_docs]))) > self.model.token_limit:
                    break
            documents = truncated_docs[:]
            print(f"Truncated documents to {len(documents)} for running position test")
        return documents

    def _test_position_at_depth(self, depth, documents, qa_pairs,
                                prompt, formatter):
        print(f"Depth: {depth}")
        # for each doc, generate llm answer to the question
        answers = {}
        for i in range(len(documents)):
            docs_copy = documents[:]
            
            idx = qa_pairs[i]["id"]
            q = qa_pairs[i]["question"]
            a = qa_pairs[i]["answer"]
            f = qa_pairs[i]["file"]

            docs_copy.pop(idx)

            test_doc = documents[idx]
            test_doc_content = test_doc.page_content

            # partition docs into 0:depth and depth+1:doc_length
            docs_partition_1 = docs_copy[:self.depths[depth]]
            docs_partition_2 = docs_copy[self.depths[depth]:]

            context_partition_1 = ""
            context_partition_2 = ""

            shuffle(docs_partition_1)
            shuffle(docs_partition_2)
            for doc in docs_partition_1:
                context_partition_1 += doc.page_content + "\n\n"
            context_partition_1 += test_doc_content
            for doc in docs_partition_2:
                context_partition_2 += doc.page_content + "\n\n"

            # combine partitions
            context = context_partition_1 + context_partition_2
        
            # now that we have the context, generate an answer to the doc question
            chain = prompt | self.model.model | formatter
            
            num_token = self.model.model.get_num_tokens_from_messages(messages=[
            HumanMessage(content=prompt.SINGLEHOP_QA_PROMPT.format(context=context, question=q))
        ])
            if num_token > self.model.token_limit-100:
                doc_content = self.encoding.decode(self.encoding.encode(
                    prompt.SINGLEHOP_QA_PROMPT.format(context=context, question=q))[:self.model.token_limit-100])
            else:
                doc_content = context

            try:
                qa = chain.invoke({"context": doc_content, "question": q})
                answers[idx] = {"id": idx, "file": f, "question": q, "answer": qa["answer"], "gold_answer": a,
                                "depth": depth, "context_length": num_token}
            except:
                print(f"Error processing document {idx}: {qa}")
                continue
        return answers

    def _evaluate_responses(self, answers_at_depth):
        ## evaluation using llm-as-a-judge
        # define prompt and output format for test
        formatter = SimpleJsonOutputParser(pydantic_object=formats.ScoreQA)
        prompt = prompts.SCORE_QA_PROMPT

        score_output = {}
        for depth, answers in answers_at_depth.items():
            scored_output_at_depth = answers.copy()
            scores = []
            for idx, item in answers.items():
                # score for correctness using llm-as-a-judge
                chain = prompt | self.model.model | formatter
                score_response = chain.invoke({"answer": item["answer"], "gold_answer": item["gold_answer"]})
                scored_output_at_depth[idx]["score"] = score_response["correct"]
                scores.append(score_response["correct"])
            score_output[depth] = scored_output_at_depth
            print(f"Accuracy at depth {depth}: {sum(scores)/len(scores)*100}%")
        return score_output, scores

    def test_position_single_hop(self):
        # define prompt and format for test
        prompt = prompt.SINGLEHOP_QA_PROMPT
        formatter = JsonOutputParser(pydantic_object=formats.SingleDocQA)

        # check if data folder exists
        if not os.path.exists(self.data_path):
            print("Creating (100) documents in ./data/ from HuggingFaceTB/cosmopedia-100k")
            create_datastore()

        # load files
        print("Loading documents...")
        files = os.listdir(self.data_path)
        loader = DirectoryLoader(self.data_path, glob="**/*.*",
                                show_progress=True,
                                use_multithreading=True,)
        documents = loader.load()

        # create QA pairs from documents
        print("Creating QA pairs from documents...")
        qa_pairs = create_qa_pairs_single_hop(documents)

        # fill up the context window, and truncate if necessary
        documents = self.get_or_truncate_context_window(self.model, documents)

        #### Test for position
        print(f"Run position test for {self.model_name}")
        # get number of docs at each depth for test
        self.depths = {i: int(i*len(documents)/100) for i in range(10, 100, 10)}

        # iterate at each depth for the test, generate responses to questions
        print(">>>>Generating answers")
        answers_at_depth = {}
        for depth in list(self.depths.keys())[:1]:
            answers = self._test_position_at_depth(depth, documents, qa_pairs,
                                         prompt, formatter)
            answers_at_depth[depth] = answers

        # evaluate the responses
        print(">>>>Evaluating responses using llm-as-a-judge")
        score_output, scores = self._evaluate_responses(answers_at_depth)