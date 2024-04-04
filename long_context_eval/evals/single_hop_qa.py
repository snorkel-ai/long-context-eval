import os
import time
import json
from datetime import datetime
import random
from typing import Optional
from tqdm import tqdm
import tiktoken
from langchain.schema import HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from parameters import formats as formats
from parameters import prompts as prompts
from parameters import models as models
from utils.create_datastore import create_datastore
from utils.create_qa_pairs import create_qa_pairs_single_hop


class SingleHopQATest:
    """
    This class tests for retrieval with long context and RAG in the single hop document QA setting.
    """
    def __init__(self,
                 model_name: Optional[str] = "gpt-3.5-turbo",
                 data_path: Optional[str] = "../data",
                 qa_pairs_path: Optional[str] = './data.json',
                 model_kwargs: Optional[dict] = dict(temperature=0.8),
                 chunk_size: Optional[int] = 1000,
                 chunk_overlap: Optional[int] = 200,
                 search_kwargs: Optional[dict] = {"k": 10},
                 embedding_model_name: Optional[str] = 'text-embedding-ada-002',
                 embedding_model_kwargs: Optional[dict] = {},
                 hfdataset: Optional[str] = "HuggingFaceTB/cosmopedia-100k",
                 hfdatasetsplit: Optional[str] = "train",
                 hfdatasetfilterdictkey: Optional[str] = "format",
                 hfdatasetfilterdictvalue: Optional[str] = "wiki",
                 hfdatasettextcol: Optional[str] = "text",
                 hfdataset_num_docs: Optional[int] = 100,
                 data_generation_model_name: Optional[str] = "gpt-3.5-turbo",
                 data_generation_model_kwargs: Optional[dict] = dict(temperature=0.8),
                 eval_model_name: Optional[str] = "gpt-3.5-turbo",
                 eval_model_kwargs: Optional[dict] = dict(temperature=0),
                 experiment_tag: Optional[str] = "tag",
                 log_path: Optional[str] = "experiments.log",
                 data_gen_prompt: Optional[str] = "single_doc_question_gen_prompt",
                 task_prompt: Optional[str] = "single_doc_qa_prompt",
                 eval_prompt: Optional[str] = "score_qa_prompt",
                 seed: Optional[int] = None,
                 num_docs: Optional[int] = None,
                 ):
        self.model_name = model_name
        self.data_path = data_path
        self.qa_pairs_path = qa_pairs_path
        self.model_kwargs = model_kwargs
        # RAG parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.search_kwargs = search_kwargs
        self.embedding_model_kwargs = embedding_model_kwargs
        # Creating docs from HF dataset parameters
        self.hfdataset = hfdataset
        self.hfdatasetsplit = hfdatasetsplit
        self.hfdatasetfilterdictkey = hfdatasetfilterdictkey
        self.hfdatasetfilterdictvalue = hfdatasetfilterdictvalue
        self.hfdatasettextcol = hfdatasettextcol
        self.hfdataset_num_docs = hfdataset_num_docs
        self.data_generation_model_name = data_generation_model_name
        self.data_generation_model_kwargs = data_generation_model_kwargs
        self.eval_model_name = eval_model_name
        self.eval_model_kwargs = eval_model_kwargs
        self.experiment_tag = experiment_tag
        self.log_path = log_path
        self.data_gen_prompt = prompts.get_prompt(data_gen_prompt)
        self.task_prompt = prompts.get_prompt(task_prompt)
        self.eval_prompt = prompts.get_prompt(eval_prompt)
        self.num_docs = num_docs
        
        random.seed(seed)


        # Get the correct model based on model name
        self.model = models.SUPPORTED_MODELS[self.model_name](self.model_name, self.model_kwargs)
        self.encoding = tiktoken.encoding_for_model(self.model_name) ## TBD: Update to get encoding for any provider model
        
        # RAG embedding model
        self.embedding_model_name = embedding_model_name
        self.embedding_model = models.SUPPORTED_MODELS[self.embedding_model_name](self.embedding_model_name,
                                                                                  self.embedding_model_kwargs)
        
        # Data generation model
        self.data_generation_model = models.SUPPORTED_MODELS[self.data_generation_model_name](
            self.data_generation_model_name, self.data_generation_model_kwargs)

        # Eval model
        self.eval_model = models.SUPPORTED_MODELS[self.eval_model_name](self.eval_model_name, self.eval_model_kwargs)

        self.documents = []

    def __str__(self):
        vars = []
        for k, v in self.__dict__.items():
            if str(k) == "documents":
                continue
            try:
                vars.append("{} = {}".format(k, v))
            except:
                continue
        return str(vars)

    def _get_or_truncate_context_window(self, documents):
        '''fill up the context window, and truncate if necessary'''
        long_context = "\n\n".join([doc.page_content for doc in documents])
        truncated_docs = []
        if len(self.encoding.encode(long_context)) > self.model.max_context_size-100:
            for i, doc in enumerate(documents):
                temp_doc_list = truncated_docs.copy()
                temp_doc_list.append(doc)
                if len(self.encoding.encode(
                    "\n\n".join(
                        [doc.page_content for doc in temp_doc_list]))) > self.model.max_context_size-100:
                    break
                else:
                    truncated_docs.append(doc)
            print(f"Truncated # of documents to {len(truncated_docs)} (context window token limits reached)")
            print("# of tokens in context: ", len(self.encoding.encode(
                    "\n\n".join(
                        [doc.page_content for doc in truncated_docs]))))
        else:
            return documents
        return truncated_docs

    def _get_responses_at_position(self, position, documents, qa_pairs,
                                prompt, formatter):
        print(f"Position: {position}")
        # for each doc, generate llm answer to the question
        answers = []
        for i in tqdm(range(len(documents))):

            q = qa_pairs[i].get("question", "")

            docs_copy = documents[:]

            docs_copy.pop(i)

            test_doc = documents[i]

            if position == 0:
                docs_partition_1 = [test_doc]
                docs_partition_2 = docs_copy.copy()
                random.shuffle(docs_partition_2)
                start_tokens = 0
                end_tokens = len(self.encoding.encode(
                    "\n\n".join([doc.page_content for doc in docs_partition_1])))
                
                context_doc_names = [doc.metadata["source"] for doc in docs_partition_1]
                context_doc_names.extend([doc.metadata["source"] for doc in docs_partition_2])

            elif position == len(documents)-1:
                docs_partition_1 = docs_copy.copy()
                docs_partition_2 = [test_doc]
                random.shuffle(docs_partition_1)
                start_tokens = len(self.encoding.encode(
                    "\n\n".join([doc.page_content for doc in docs_partition_1])))
                end_tokens = start_tokens + len(self.encoding.encode(
                    "\n\n".join([doc.page_content for doc in docs_partition_2])))
                
                context_doc_names = [doc.metadata["source"] for doc in docs_partition_1]
                context_doc_names.extend([doc.metadata["source"] for doc in docs_partition_2])
            else:
                # partition docs into 0:position-1 and position+1:doc_length
                docs_partition_1 = docs_copy[:position]
                docs_partition_2 = docs_copy[position:]
                random.shuffle(docs_partition_1)
                random.shuffle(docs_partition_2)
                start_tokens = len(self.encoding.encode(
                    "\n\n".join([doc.page_content for doc in docs_partition_1])))
                
                docs_partition_1.append(test_doc)
                end_tokens = len(self.encoding.encode(
                    "\n\n".join([doc.page_content for doc in docs_partition_1])))

                context_doc_names = [doc.metadata["source"] for doc in docs_partition_1]
                context_doc_names.extend([doc.metadata["source"] for doc in docs_partition_2])

            context_partition_1 = "\n\n".join([doc.page_content for doc in docs_partition_1])
            context_partition_2 = "\n\n".join([doc.page_content for doc in docs_partition_2])

            # combine partitions
            context = context_partition_1 + context_partition_2
        
            # now that we have the context, generate an answer to the doc question
            chain = prompt | self.model.model | formatter
            
            num_token = self.model.model.get_num_tokens_from_messages(messages=[
            HumanMessage(content=prompt.format(context=context, question=q))
        ])
            assert num_token < self.model.max_context_size-100, f"{num_token} context size is greater than max context length {self.model.max_context_size-100}"
            ## TODO: Remove this condition since we assert
            if num_token > self.model.max_context_size-100:
                doc_content = self.encoding.decode(self.encoding.encode(
                    prompt.format(context=context, question=q))[:self.model.max_context_size-100])
            else:
                doc_content = context

            try:
                qa = chain.invoke({"context": doc_content, "question": q})
            except:
                print(f"Error generating LLM response for document {i}")
                continue

            answers_dict = qa_pairs[i].copy()
            answers_dict.update({"llm_response": qa, "position": position, "context_length": num_token,
                            "model": self.model_name, "model_kwargs": self.model_kwargs, 
                            "start_tokens": start_tokens, "end_tokens": end_tokens,
                            "context_list": context_doc_names,
                            "answer_document": test_doc.metadata["source"]})
            answers.append(answers_dict)
        return answers

    def _evaluate_responses(self, answers_at_position):
        ## evaluation using llm-as-a-judge
        # define prompt and output format for test
        formatter = SimpleJsonOutputParser(pydantic_object=formats.ScoreQA)

        score_output = {}
        for position, documents in tqdm(answers_at_position.items()):
            scored_output_at_position = []
            scores = []
            for idx, response_dict in enumerate(documents):
                # score for correctness using llm-as-a-judge
                chain = self.eval_prompt | self.eval_model.model | formatter
                score_response = chain.invoke({"answer": response_dict["llm_response"],
                                               "gold_answer": response_dict["answer"]})
                response_dict["score"] = int(score_response["correct"])
                scored_output_at_position.append(response_dict)
                scores.append(score_response["correct"])
            score_output[position] = scored_output_at_position
            print(f"Accuracy at position {position}: {sum(scores)/len(scores)*100:.1f}%")
        return score_output
    
    def _reorder_qa_pairs(self, qa_pairs):
        reordered_qa_pairs = []
        docs_to_remove = []
        for idx, doc in enumerate(self.documents):
            file_name = doc.metadata["source"]
            if file_name not in qa_pairs:
                docs_to_remove.append(idx)
            reordered_qa_pairs.append(qa_pairs[file_name])
        
        # to remove documents for which we have no QA pairs.
        if docs_to_remove:
            print("These documents do not have a QA pair, will be removed from context")
            for idx in docs_to_remove:
                self.documents.pop(idx)
        return reordered_qa_pairs

    def _evaluate_rag_responses(self, rag_answers):
        formatter = SimpleJsonOutputParser(pydantic_object=formats.ScoreQA)
        prompt = prompts.SCORE_QA_PROMPT

        scored_output = rag_answers.copy()
        scores = []
        for idx, item in rag_answers.items():
            # score for correctness using llm-as-a-judge
            chain = prompt | self.model.model | formatter
            score_response = chain.invoke({"answer": item["answer"], "gold_answer": item["gold_answer"]})
            scored_output[idx]["score"] = int(score_response["correct"])
            scores.append(score_response["correct"])
        print(f"RAG Accuracy: {sum(scores)/len(scores)*100:.2f}%")
        return scored_output

    def test_position_accuracy(self):
        print("\n\n")
        test_start_time = time.time()

        # define format for test
        # formatter = JsonOutputParser(pydantic_object=formats.SingleDocQA)
        formatter = StrOutputParser()

        # check if data folder exists
        if not os.path.exists(self.data_path):
            create_datastore(self.data_path, 
                             self.hfdataset,
                             self.hfdatasetsplit,
                             self.hfdatasetfilterdictkey,
                             self.hfdatasetfilterdictvalue,
                             self.hfdatasettextcol,
                             self.hfdataset_num_docs,
                             self.data_generation_model_name,
                             self.data_generation_model_kwargs)

        # load files
        if not self.documents:
            print("Loading documents...")
            files = os.listdir(self.data_path)
            loader = DirectoryLoader(self.data_path, glob="**/*.*",
                                    show_progress=True,)
                                    # use_multithreading=True,)
            self.documents = loader.load()
            self.documents = self.documents[:self.num_docs]
            print("# of documents: ", len(self.documents))

        # create QA pairs from documents
        try:
            qa_pairs = json.load(open(self.qa_pairs_path))
        except Exception as e:
            print(f"Creating QA pairs from documents at {self.data_path} ...")
            qa_pairs = create_qa_pairs_single_hop(self.documents, self.qa_pairs_path,
                                                  self.data_gen_prompt)

        # to keep same order as documents loaded
        qa_pairs = self._reorder_qa_pairs(qa_pairs)

        # fill up the context window, and truncate if necessary
        self.documents = self._get_or_truncate_context_window(self.documents)

        #### Test for position
        print(f"Run position test on long context for {self.model_name}")
        # get number of docs at each depth for test
        self.positions = [0, int(len(self.documents)*.25), int(len(self.documents)*.5), int(len(self.documents)*.75), len(self.documents)] #range(len(self.documents))

        # iterate at each position for the test, generate responses to questions
        print(">>>>Generate llm responses at document depths...")
        answers_at_position = {}
        for position in self.positions:
            answers = self._get_responses_at_position(position, self.documents, qa_pairs,
                                         self.task_prompt, formatter)
            answers_at_position[position] = answers
        # TBD: Should save answers_at_position results

        # evaluate the responses
        print(">>>>Evaluating responses using llm-as-a-judge")
        score_output = self._evaluate_responses(answers_at_position)

        # save score results
        if not os.path.exists("./output"): os.makedirs("./output")
        date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        save_path = f"./output/position_test_results_{self.model_name}_{self.experiment_tag}-{date_time}.json"
        with open(os.path.join(save_path), 'w') as f:
            f.write(json.dumps(score_output))

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        print(f"Position Test Duration: {test_elapsed_time:.1f} seconds")
        print(f"Results saved at {save_path}")


    def test_rag(self):
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        print("\n\n")
        test_start_time = time.time()

        # define prompt and format for test
        prompt = prompts.SINGLEHOP_QA_PROMPT
        formatter = JsonOutputParser(pydantic_object=formats.SingleDocQA)

        # check if data folder exists
        if not os.path.exists(self.data_path):
            print("Creating (100) documents in ./data/ from HuggingFaceTB/cosmopedia-100k")
            create_datastore(self.data_path)

        # load files
        if not self.documents:
            print("Loading documents...")
            files = os.listdir(self.data_path)
            loader = DirectoryLoader(self.data_path, glob="**/*.*",
                                    show_progress=True,
                                    use_multithreading=True,)
            self.documents = loader.load()
            print("# of documents: ", len(self.documents))

        # create QA pairs from documents
        try:
            qa_pairs = json.load(open('data.json'))
        except Exception as e:
            print("Creating QA pairs from documents at ./data.json ...")
            qa_pairs = create_qa_pairs_single_hop(self.documents)

        # since we want to compare RAG performance with long context
        # fill up the context window, and truncate if necessary
        self.documents = self._get_or_truncate_context_window(self.documents)

        # chunk documents and add to vector store
        print(f"Run RAG test for {self.model_name}")
        print(">>>>Chunk and add to vector store...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                       chunk_overlap=self.chunk_overlap)
        splits = text_splitter.split_documents(self.documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=self.embedding_model.model)

        retriever = vectorstore.as_retriever(search_kwargs=self.search_kwargs)

        # for each QA pair, generate llm answer to the question
        print(">>>>Generate llm responses...")
        rag_answers = {}
        for i in range(len(self.documents)):
            idx = qa_pairs[str(i)]["id"]
            q = qa_pairs[str(i)]["question"]
            a = qa_pairs[str(i)]["answer"]
            f = qa_pairs[str(i)]["file"]

            rag_chain = chain = prompt | self.model.model | formatter

            try:
                qa = rag_chain.invoke({"context": retriever | format_docs, "question": q})
                rag_answers[idx] = {"id": idx, "file": f, "question": q, "answer": qa["answer"], "gold_answer": a,
                                    "model": self.model_name, "model_kwargs": self.model_kwargs, 
                                    "embedding_model": self.embedding_model_name, "embedding_model_kwargs": self.embedding_model_kwargs,
                                    "chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap,
                                    "search_kwargs": self.search_kwargs}
            except:
                print(f"Error generating RAG response for document {idx}")
                continue

        # evaluate the responses
        print(">>>>Evaluating RAG responses using llm-as-a-judge")
        score_output = self._evaluate_rag_responses(rag_answers)
        
        # save score results
        if not os.path.exists("./output"): os.makedirs("./output")
        save_path = f"./output/rag_test_results_{self.model_name}.json"
        with open(os.path.join(save_path), 'w') as f:
            f.write(json.dumps(score_output))

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        print(f"RAG Test Duration: {test_elapsed_time:.1f} seconds")
        print(f"Results saved at {save_path}")

        # cleanup
        vectorstore.delete_collection()
