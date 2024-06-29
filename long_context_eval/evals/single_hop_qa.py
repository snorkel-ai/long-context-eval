import os
import re
import time
import json
from datetime import datetime
from collections import defaultdict, OrderedDict
import random
from typing import Optional, Literal
from tqdm import tqdm
import numpy as np
from langchain.schema import HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from parameters import prompts_formats as prompts_formats
from parameters import models as models

# import langchain
# langchain.debug = True


def compute_avg_similarities(embedding_vectors):
    # Convert the list of embedding vectors to a NumPy array
    embedding_vectors = np.array(embedding_vectors)
    
    # Compute the dot product of all pairs of vectors
    dot_products = np.dot(embedding_vectors, embedding_vectors.T)
    
    # Compute the Euclidean norms of all vectors
    norms = np.linalg.norm(embedding_vectors, axis=1)
    
    # Compute the pairwise cosine distances
    cosine_similarities = dot_products / np.outer(norms, norms)
    
    # Set diagonal elements to zero to exclude self-distance
    np.fill_diagonal(cosine_similarities, 0)
    
    # Calculate the average distance of each vector with all other vectors
    avg_similarities = np.mean(cosine_similarities, axis=1)
    
    return avg_similarities


class SingleHopQATest:
    """
    This class tests for retrieval with long context and RAG in the single hop document QA setting.
    """
    def __init__(self,
                 model_name: Optional[str] = "gpt-3.5-turbo",
                 data_path: Optional[str] = "../data",
                 task_path: Optional[str] = './data.json',
                 results_folder_path: Optional[str] = "./results",
                 model_kwargs: Optional[dict] = dict(temperature=0.7),
                 chunk_size: Optional[int] = 1000,
                 chunk_overlap: Optional[int] = 200,
                 search_kwargs: Optional[dict] = {"k": 4},
                 embedding_model_name: Optional[str] = 'text-embedding-ada-002',
                 embedding_model_kwargs: Optional[dict] = {},
                 eval_model_name: Optional[str] = "gpt-4-turbo-2024-04-09",
                 eval_model_kwargs: Optional[dict] = dict(temperature=0),
                 experiment_tag: Optional[str] = "tag",
                 log_path: Optional[str] = "experiments.log",
                 data_gen_prompt: Optional[str] = "single_doc_question_gen_prompt",
                 task_prompt: Optional[str] = "single_doc_qa_prompt",
                 eval_prompt: Optional[str] = "score_qa_prompt",
                 seed: Optional[int] = None,
                 tests: Optional[Literal['all', 'position', 'rag', 'medoid', 'control_medoid']] = 'all',
                 document_depth_percents_list: Optional[list] = list((0, 25, 50, 75, 100)), #applicable to the position test
                 percent_ctxt_window_used: Optional[list] = list((0, 25, 50, 75, 100)), #applicable to the RAG test
                 num_runs_medoid_vote: Optional[int] = 1,
                 document_depth_percents_medoid: Optional[int] = 25,
                 ):
        self.model_name = model_name
        self.data_path = data_path
        self.task_path = task_path
        self.model_kwargs = model_kwargs
        self.results_folder_path = results_folder_path
       
        # RAG parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.search_kwargs = search_kwargs
        self.embedding_model_kwargs = embedding_model_kwargs
        
        self.eval_model_name = eval_model_name
        self.eval_model_kwargs = eval_model_kwargs
        self.experiment_tag = experiment_tag
        self.log_path = log_path
        self.data_gen_prompt, self.data_gen_format = prompts_formats.get_prompt_and_format(data_gen_prompt)
        self.task_prompt, self.task_format = prompts_formats.get_prompt_and_format(task_prompt)
        self.eval_prompt, self.eval_format = prompts_formats.get_prompt_and_format(eval_prompt)
        
        self.rng = random.Random(seed)

        # Get the correct model based on model name
        self.model = models.SUPPORTED_MODELS[self.model_name](self.model_name, self.model_kwargs)
        self.encoding = self.model.encoding

        # RAG embedding model
        if tests in ["rag", "medoid", "control_medoid", "all"]:
            self.embedding_model_name = embedding_model_name
            self.embedding_model = models.SUPPORTED_MODELS[self.embedding_model_name](self.embedding_model_name,
                                                                                    self.embedding_model_kwargs)

        # Eval model
        self.eval_model = models.SUPPORTED_MODELS[self.eval_model_name](self.eval_model_name, self.eval_model_kwargs)

        # medoid voting improvement
        self.num_runs_medoid_vote = num_runs_medoid_vote
        self.document_depth_percents_medoid = document_depth_percents_medoid

        self.loaded_documents = []
        self.documents = {}
        self.qa_pairs = {}

        self.document_depth_percents_list = document_depth_percents_list #if document_depth_percents_list is not None else [0, 25, 50, 75, 100]
        self.percent_ctxt_window_used = percent_ctxt_window_used #if percent_ctxt_window_used is not None else [0, 25, 50, 75, 100]

        # create folder for results
        date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        if not os.path.exists(self.results_folder_path): os.makedirs(self.results_folder_path)
        self.results_folder_path = os.path.join(self.results_folder_path, "-".join([experiment_tag, date_time]))
        os.makedirs(self.results_folder_path)


        self.load_docs_and_qa_pairs()

    def __str__(self):
        vars = []
        for k, v in self.__dict__.items():
            if (str(k) == "documents") or (str(k) == "loaded_documents"):
                continue
            try:
                vars.append("{} = {}".format(k, v))
            except:
                continue
        return str(vars)

    def load_docs_and_qa_pairs(self,):
        """
        Loads documents and corresponding question-answer pairs for experiments.
        
        Checks if the data folder exists and is not empty.
        If documents have not been loaded previously, loads them from the specified data path.
        Then, creates question-answer pairs from the loaded documents.
        If question-answer pairs have been pre-saved, loads them; otherwise, generates new pairs.
        Filters out pairs whose answer documents are not part of the loaded document set.
        
        Returns:
            None
        """
        # check if data folder exists
        if not os.path.exists(self.data_path) or not os.listdir(self.data_path):
            print("No documents for running experiments.")
            exit(0)

        # load files
        if not self.documents:
            print(f"Loading documents. at {self.data_path}..")
            files = os.listdir(self.data_path)
            # dont use multithreading since it affects reproducibility (changes order of docs)
            loader = DirectoryLoader(self.data_path, glob="**/*.*",
                                    show_progress=True,)
                                    # use_multithreading=True,)
            self.loaded_documents = loader.load()
            print("# of loaded documents: ", len(self.loaded_documents))

        # get QA pairs
        try:
            self.qa_pairs = json.load(open(self.task_path))
            qa_pairs_ordered = OrderedDict(self.qa_pairs)
            qa_pairs_ordered_filename = [qapair["answer_doc"] for k, qapair in qa_pairs_ordered.items()]
            print("# of QA pairs to test: ", len(self.qa_pairs))
        except Exception as e:
            print("No QA pairs for running experiments.")
            exit(0)

        # reorder docs according to QA pairs before truncation
        qa_files, remainder_files = [], []
        for doc in self.loaded_documents:
            if os.path.split(doc.metadata["source"])[1] in qa_pairs_ordered_filename:
                qa_files.append(os.path.split(doc.metadata["source"])[1])
            else:
                remainder_files.append(os.path.split(doc.metadata["source"])[1])

        self.documents = {os.path.split(doc.metadata["source"])[1]: doc for doc in self.loaded_documents}

        reordered_docs = []
        for filename in qa_files:
            reordered_docs.append(self.documents[filename])
        for filename in remainder_files:
            reordered_docs.append(self.documents[filename])
        assert len(self.documents) == len(reordered_docs)

        # truncate context window
        self.documents = self._get_or_truncate_context_window(reordered_docs)
        print("# of truncated documents for test: ", len(self.documents))
        self.documents = {os.path.split(doc.metadata["source"])[1]: doc for doc in self.documents}

        # filtering out QA pairs that are not part of the truncated document set
        self.qa_pairs = {k: v for k, v in self.qa_pairs.items() if v["answer_doc"] in self.documents}
        print("# of QA pairs to test after filtering: ", len(self.qa_pairs))

        # with open(os.path.join(self.task_path), 'w') as f:
        #     f.write(json.dumps(self.qa_pairs))

    def _get_or_truncate_context_window(self, documents, proxy_output_tokens=250):
        """
        Fills up the context window with the input documents 
        and truncates if necessary to fit within the maximum context size of the model.

        Args:
            documents (list): A list of document objects.

        Returns:
            list: A list of truncated documents that fit within the maximum context size 
                of the model, or the original list of documents if no truncation is needed.
        """
        long_context = "\n\n".join([doc.page_content for doc in documents])
        print("total # of tokens", len(self.encoding.encode(long_context)))

        truncated_docs = []
        if len(self.encoding.encode(long_context)) > self.model.max_context_size-proxy_output_tokens:
            for i, doc in enumerate(documents):
                temp_doc_list = truncated_docs.copy()
                temp_doc_list.append(doc)
                # print(doc.metadata["source"])
                # print(len(self.encoding.encode(
                #     "\n\n".join(
                #         [doc.page_content for doc in temp_doc_list]))))
                if len(self.encoding.encode(
                    "\n\n".join(
                        [doc.page_content for doc in temp_doc_list]))) > self.model.max_context_size-proxy_output_tokens:
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

    def _create_doc_set_for_long_ctxt_rag(self, qa_pair, num_noisy_docs):
        """
        Creates a set of documents for long-context versus retrieval augmented generation (RAG) test.

        Args:
            qa_pair (dict): A dictionary representing a question-answer pair, with the "answer_doc" key indicating 
                            the document ID containing the answer.
            num_noisy_docs (int): The number of distractor documents to include in the set.

        Returns:
            list: A list of document objects, including the document containing the answer and optionally 
                additional distractor documents.
        """
        f = qa_pair.get("answer_doc", "")

        # get test document
        doc_set = [self.documents[f]]
        
        if num_noisy_docs > 0:
            # get distractor docs
            docs_copy = [doc for k, doc in self.documents.items() if k != f]
            self.rng.shuffle(docs_copy)
            noisy_docs = docs_copy[:num_noisy_docs]
            doc_set.extend(noisy_docs)

        self.rng.shuffle(doc_set)
        return doc_set

    def _get_responses_long_ctxt(self, doc_set, qa_pair, 
                                 prompt, formatter):
        """
        Generates responses for a given set of documents and a question-answer pair.

        Args:
            doc_set (list): A list of document objects constituting the context for the question-answer pair.
            qa_pair (dict): A dictionary representing a question-answer pair.
            prompt (PromptTemplate): The prompt template for generating the response.
            formatter (langchain.output_parsers): The formatter to use for processing the response.
            
        Returns:
            dict: A dictionary containing information about the generated response
        """
        q = qa_pair.get("question", "")
        f = qa_pair.get("answer_doc", "")
        
        context_doc_names = [doc.metadata["source"] for doc in doc_set]
        context = "\n\n".join([doc.page_content for doc in doc_set])

        # now that we have the context, generate an answer to the doc question
        chain = prompt | self.model.model | formatter
        
        num_token = len(self.encoding.encode(
                    prompt.format(context=context, question=q)))
                                         
        assert num_token < self.model.max_context_size, f"{num_token} context size is greater than max context length {self.model.max_context_size}"
        ## TODO: Remove this condition since we assert
        if num_token > self.model.max_context_size:
            doc_content = self.encoding.decode(self.encoding.encode(
                prompt.format(context=context, question=q))[:self.model.max_context_size])
        else:
            doc_content = context

        test_start_time = time.time()
        try:
            qa = chain.invoke({"context": doc_content, "question": q})
        except Exception as e:
            print(f"Error generating LLM response (long context) for document {f}")
            print(e)
            qa = ""
        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        answers_dict = qa_pair.copy()
        answers_dict.update({"llm_response": qa, "context_length": num_token,
                            "model": self.model_name, "model_kwargs": self.model_kwargs, 
                            "context_list": context_doc_names,
                            # "context": context,
                            "answer_document": f,
                            "eval_prompt": self.eval_prompt.template,
                            "eval_model": self.eval_model_name,
                            "llm_response_time_seconds": test_elapsed_time})
        return answers_dict

    def _get_responses_rag(self, retriever, qa_pair,
                           prompt, formatter):
        """
        Generates responses using the retrieval-augmented generation (RAG) approach.

        Args:
            retriever: The document retriever used to fetch relevant documents.
            qa_pair (dict): A dictionary representing a question-answer pair.
            prompt (PromptTemplate): The prompt template for generating the response.
            formatter (langchain.output_parsers): The formatter to use for processing the response.

        Returns:
            dict: A dictionary containing information about the generated response.
        """
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        q = qa_pair.get("question", "")
        f = qa_pair.get("answer_doc", "")

        retrieved_docs = [doc.page_content for doc in retriever.get_relevant_documents(q)]

        # now that we have the context, generate an answer to the doc question
        chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | self.model.model | formatter

        test_start_time = time.time()
        try:
            qa = chain.invoke(q)

        except Exception as e:
            print(f"Error generating LLM response (rag) for document {f}")
            print(e)
            qa = ""
        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        answers_dict = qa_pair.copy()
        answers_dict.update({"llm_response": qa, 
                             "model": self.model_name, "model_kwargs": self.model_kwargs,
                             "answer_document": f, "embedding_model": self.embedding_model_name,
                             "embedding_model_kwargs": self.embedding_model_kwargs,
                             "chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap,
                             "search_kwargs": self.search_kwargs,
                             "retrieved_docs": retrieved_docs,
                             "eval_prompt": self.eval_prompt.template,
                             "eval_model": self.eval_model_name,
                             "llm_response_time_seconds": test_elapsed_time
                             })

        return answers_dict

    def _get_responses_at_positions(self, qa_pair,
                                    prompt, formatter,
                                    answers_at_position,
                                    positions):
        """
        Generates responses at all positions for a single QA pair.

        Args:
            retriever: The document retriever used to fetch relevant documents.
            qa_pair (dict): A dictionary representing a question-answer pair.
            prompt (PromptTemplate): The prompt template for generating the response.
            formatter (langchain.output_parsers): The formatter to use for processing the response.

        Returns:
            dict: A dictionary containing information about the generated response.
        """
        q = qa_pair.get("question", "")
        f = qa_pair.get("answer_doc", "")

        for position in tqdm(positions):
            # for each position, generate llm answer to the question
            # get test document
            test_doc = self.documents[f]

            # get distractor docs
            docs_copy = [doc for k, doc in self.documents.items() if k != f]
            self.rng.shuffle(docs_copy)

            if position == 0:
                # test document goes at the top
                docs_partition_1 = [test_doc]
                docs_partition_2 = docs_copy.copy()
                start_tokens = 0
                end_tokens = len(self.encoding.encode(
                    "\n\n".join([doc.page_content for doc in docs_partition_1])))
            
            elif position == len(self.documents)-1:
                # test document goes at the bottom
                docs_partition_1 = docs_copy.copy()
                docs_partition_2 = [test_doc]
                start_tokens = len(self.encoding.encode(
                    "\n\n".join([doc.page_content for doc in docs_partition_1])))
                end_tokens = start_tokens + len(self.encoding.encode(
                    "\n\n".join([doc.page_content for doc in docs_partition_2])))
            
            else:
                # partition docs into 0:position and position:doc_length
                docs_partition_1 = docs_copy[:position]
                docs_partition_2 = docs_copy[position:]
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
            
            num_token = len(self.encoding.encode(
                    prompt.format(context=context, question=q)))

            assert num_token < self.model.max_context_size, f"{num_token} context size is greater than max context length {self.model.max_context_size}"
            ## TODO: Remove this condition since we assert
            if num_token > self.model.max_context_size:
                doc_content = self.encoding.decode(self.encoding.encode(
                    prompt.format(context=context, question=q))[:self.model.max_context_size])
            else:
                doc_content = context

            test_start_time = time.time()
            try:
                qa = chain.invoke({"context": doc_content, "question": q})
            except Exception as e:
                print(f"Error generating LLM response for document {f}")
                print(e)
                qa = ""
            test_end_time = time.time()
            test_elapsed_time = test_end_time - test_start_time

            answers_dict = qa_pair.copy()
            answers_dict.update({"llm_response": qa, "position": position, "context_length": num_token,
                                    "model": self.model_name, "model_kwargs": self.model_kwargs, 
                                    "start_tokens": start_tokens, "end_tokens": end_tokens,
                                    "context_list": context_doc_names,
                                    "answer_document": test_doc.metadata["source"],
                                    "eval_prompt": self.eval_prompt.template,
                                    "eval_model": self.eval_model_name,
                                    "llm_response_time_seconds": test_elapsed_time})
            answers_at_position[position].append(answers_dict)
        return answers_at_position

    def _evaluate_responses(self, answers_at_position):
        """
        Evaluates the responses using llm-as-a-judge generated by the model at different positions.

        Args:
            answers_at_position (dict): A dictionary containing responses grouped by position.

        Returns:
            dict: A dictionary containing evaluated responses with additional scoring information.
        """
        score_output = {}
        for position, documents in answers_at_position.items():
            scored_output_at_position = []
            scores = []
            for idx, response_dict in enumerate(documents):
                # score for correctness using llm-as-a-judge
                chain = self.eval_prompt | self.eval_model.model | self.eval_format
                score_response = chain.invoke({"question": response_dict["question"],
                                               "answer": response_dict["llm_response"],
                                               "gold_answer": response_dict["answer"]})
                response_dict["score"] = int(score_response["correct"])
                scored_output_at_position.append(response_dict)
                scores.append(score_response["correct"])
            score_output[position] = scored_output_at_position
            print(f"Acc at position {position}: {sum(scores)/len(scores)*100:.1f}%")
        return score_output
    
    def _evaluate_responses_long_ctxt_rag(self, answers):
        """
        Evaluates the responses using llm-as-a-judge generated by the model for different context sizes.

        Args:
            answers (dict): A dictionary containing responses grouped by context size.

        Returns:
            dict: A dictionary containing evaluated responses with additional scoring information.
        """

        chain = self.eval_prompt | self.eval_model.model | self.eval_format

        score_output = {}
        for num_noisy_docs, response_dict in answers.items():
            score_output[num_noisy_docs] = {}
            long_ctxt_responses = response_dict["long_ctxt"]
            rag_responses = response_dict["rag"]
            
            scored_output_long_ctxt = []
            scores_long_ctxt = []

            for idx, long_ctxt_response in enumerate(long_ctxt_responses):
                # score for correctness using llm-as-a-judge
                score_response = chain.invoke({"question": long_ctxt_response["question"],
                                               "answer": long_ctxt_response["llm_response"],
                                               "gold_answer": long_ctxt_response["answer"]})
                long_ctxt_response["score"] = int(score_response["correct"])
                scored_output_long_ctxt.append(long_ctxt_response)
                scores_long_ctxt.append(score_response["correct"])
            score_output[num_noisy_docs]["long_ctxt"] = scored_output_long_ctxt
            print(f"Long context acc with {num_noisy_docs} noisy documents: {sum(scores_long_ctxt)/len(scores_long_ctxt)*100:.1f}%")

            scored_output_rag = []
            scores_rag = []
            for idx, rag_response in enumerate(rag_responses):
                # score for correctness using llm-as-a-judge
                score_response = chain.invoke({"question": rag_response["question"],
                                               "answer": rag_response["llm_response"],
                                               "gold_answer": rag_response["answer"]})
                rag_response["score"] = int(score_response["correct"])
                scored_output_rag.append(rag_response)
                scores_rag.append(score_response["correct"])
            score_output[num_noisy_docs]["rag"] = scored_output_rag
            print(f"RAG acc with {num_noisy_docs} noisy documents: {sum(scores_rag)/len(scores_rag)*100:.1f}%")

        return score_output

    def test_position_accuracy(self):
        """
        Tests the accuracy of the model at different positions within the document set.

        This method evaluates the model's performance by generating responses at predefined positions 
        within the document set. It saves the generated responses and their evaluations for analysis.

        Returns:
            None
        """
        print("\n\n")
        test_start_time = time.time()

        #### Test for position
        positions = [int(len(self.documents)*doc_percent/100) for doc_percent in self.document_depth_percents_list]
        print(f"Run position test on long context for {self.model_name} at positions: {positions}")

        answers_at_position = {i: [] for i in positions}
        print(">>>>Generate llm responses at positions...")
        for idx, qa_pair in self.qa_pairs.items():
            print("QA Pair: ", idx)
            answers_at_position = self._get_responses_at_positions(qa_pair,
                                         self.task_prompt, self.task_format,
                                         answers_at_position, positions)

        # save responses
        model_name = re.sub("/", "-", self.model_name)
        new_path = os.path.join(self.results_folder_path, "responses", "position")
        if not os.path.exists(new_path): os.makedirs(new_path)
        date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        save_path = os.path.join(new_path, f"position_test_results_{self.experiment_tag}-{model_name}-{date_time}.json")
        with open(save_path, 'w') as f:
            f.write(json.dumps(answers_at_position))

        # evaluate the responses
        print(">>>>Evaluating responses using llm-as-a-judge")
        score_output = self._evaluate_responses(answers_at_position)

        # save score results
        new_path = os.path.join(self.results_folder_path, "outputs", "position")
        if not os.path.exists(new_path): os.makedirs(new_path)
        date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        save_path = os.path.join(new_path, f"position_test_results_{self.experiment_tag}-{model_name}-{date_time}.json")
        with open(save_path, 'w') as f:
            f.write(json.dumps(score_output))

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        print(f"Position Test Duration: {test_elapsed_time:.1f} seconds")
        print(f"Results saved at {save_path}")

    def test_long_context_length_versus_rag(self):
        """Tests at what context length, is long context retrieval more or less effective than RAG.
        
        Returns:
            None
        """
        print("\n\n")
        test_start_time = time.time()

        noise_levels = [int(len(self.documents)*doc_percent/100) for doc_percent in self.percent_ctxt_window_used]
        print(f">>>>Generate llm responses w/ long context and RAG at noise levels {noise_levels}...")
        answers_at_noise_level = {}
        for num_noisy_docs in noise_levels: #range(len(self.documents)):
            answers_long_ctxt, answers_rag = [], []
            print("# of noisy documents: ", num_noisy_docs)
            for _, qa_pair in tqdm(self.qa_pairs.items()):
                doc_set = self._create_doc_set_for_long_ctxt_rag(qa_pair, num_noisy_docs)

                answers_dict = self._get_responses_long_ctxt(doc_set, qa_pair,
                                                             self.task_prompt, self.task_format)
                answers_long_ctxt.append(answers_dict)


                # # add sleep if you see request errors
                # time.sleep(40)

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                            chunk_overlap=self.chunk_overlap)
                splits = text_splitter.split_documents(doc_set)
                vectorstore = Chroma.from_documents(documents=splits, embedding=self.embedding_model.model)
                retriever = vectorstore.as_retriever(search_kwargs=self.search_kwargs)

                rag_answers_dict = self._get_responses_rag(retriever, qa_pair,
                                                           self.task_prompt, self.task_format)

                answers_rag.append(rag_answers_dict)

                vectorstore.delete_collection()

            answers_at_noise_level[num_noisy_docs] = {"long_ctxt": answers_long_ctxt,
                                                      "rag": answers_rag}

        # save response results
        model_name = re.sub("/", "-", self.model_name)
        new_path = os.path.join(self.results_folder_path, "responses", "rag")
        if not os.path.exists(new_path): os.makedirs(new_path)
        date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        save_path = os.path.join(new_path, f"long_ctxt_rag_responses_{self.experiment_tag}-{model_name}-{date_time}.json")
        with open(save_path, 'w') as f:
            f.write(json.dumps(answers_at_noise_level))

        # evaluate the responses
        print(">>>>Evaluating RAG responses using llm-as-a-judge")
        score_output = self._evaluate_responses_long_ctxt_rag(answers_at_noise_level)

        # save score results
        new_path = os.path.join(self.results_folder_path, "outputs", "rag")
        if not os.path.exists(new_path): os.makedirs(new_path)
        date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        save_path = os.path.join(new_path, f"long_ctxt_rag_test_results_{self.experiment_tag}-{model_name}-{date_time}.json")
        with open(save_path, 'w') as f:
            f.write(json.dumps(score_output))

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        print(f"RAG Test Duration: {test_elapsed_time:.1f} seconds")
        print(f"Results saved at {save_path}")

    def test_medoid_voting(self):
        """Tests for recall improvement with a medoid voting approach.

            1. Generate responses multiple times, randomly permuting the documents in the context each time (as few as 3 works)
            2. Embed responses using an embedding model
            3. Take the medoid response, i.e. the response with the least avg. dissimilarity to all other responses.
        
        Returns:
            None
        """
        print("\n\n")
        test_start_time = time.time()

        num_noisy_docs = len(self.documents)
        print(f">>>>Generate llm responses w/ long context {self.num_runs_medoid_vote} times for each QA pair...")
        answers_at_permutation = {}
        for idx, qa_pair in self.qa_pairs.items():
            answers_long_ctxt = []
            print("QA pair: ", idx)
            
            for p in tqdm(range(self.num_runs_medoid_vote)):
                doc_set = self._create_doc_set_for_long_ctxt_rag(qa_pair, num_noisy_docs=num_noisy_docs)
                answers_dict = self._get_responses_long_ctxt(doc_set, qa_pair,
                                                                self.task_prompt, self.task_format)
                answers_long_ctxt.append(answers_dict)

            answers_at_permutation[idx] = {"question": qa_pair.get("question", ""),
                                           "answer": qa_pair.get("answer", ""),
                                           "answer_doc": qa_pair.get("answer_doc", ""),
                                           "long_ctxt": answers_long_ctxt,
                                        }

        # save response results
        model_name = re.sub("/", "-", self.model_name)
        new_path = os.path.join(self.results_folder_path, "responses", "medoid")
        if not os.path.exists(new_path): os.makedirs(new_path)
        date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        save_path = os.path.join(new_path, f"maj_vote_{self.num_runs_medoid_vote}_runs_{self.experiment_tag}-{model_name}-{date_time}.json")
        with open(save_path, 'w') as f:
            f.write(json.dumps(answers_at_permutation))

        # get the centroid answer from permutations
        final_response = {num_noisy_docs: []}
        for idx, response_dict in tqdm(answers_at_permutation.items()):
            answer_list = [res["llm_response"] for res in response_dict["long_ctxt"]]

            # get embeddings
            embeds = self.embedding_model.model.embed_documents(answer_list)
            embeds = np.array(embeds)
            avg_similarities = compute_avg_similarities(embedding_vectors=embeds)

            max_arg = np.argmax(avg_similarities)
            best_response = answer_list[max_arg]
            final_response[num_noisy_docs].append({"question": response_dict["question"],
                                                   "answer": response_dict["answer"],
                                                   "answer_doc": response_dict["answer_doc"],
                                                   "llm_response": best_response})
        
        # save response results
        model_name = re.sub("/", "-", self.model_name)
        date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        save_path = os.path.join(new_path, f"maj_vote_{self.experiment_tag}-{model_name}-{date_time}.json")
        with open(save_path, 'w') as f:
            f.write(json.dumps(answers_at_permutation))

        # evaluate the responses
        print(">>>>Evaluating responses using llm-as-a-judge")
        score_output = self._evaluate_responses(final_response)

        # save score results
        new_path = os.path.join(self.results_folder_path, "outputs", "medoid")
        if not os.path.exists(new_path): os.makedirs(new_path)
        date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        save_path = os.path.join(new_path, f"maj_vote_test_results_{self.experiment_tag}-{model_name}-{date_time}.json")
        with open(save_path, 'w') as f:
            f.write(json.dumps(score_output))

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        print(f"Majority Vote Test Duration: {test_elapsed_time:.1f} seconds")
        print(f"Results saved at {save_path}")

    def test_control_medoid_voting(self):
        """Tests for recall improvement with a medoid voting approach.

        As a control experiment for medoid voting, keeping the document depth fixed, run the task multiple times.
        
        Returns:
            None
        """
        print("\n\n")
        test_start_time = time.time()

        position = int(len(self.documents)*self.document_depth_percents_medoid/100)
        print(f"Run position test on long context for {self.model_name} at position: {position}")
    
        answers_at_permutation = {}
        print(f">>>>Generate llm responses at position {position}, {self.num_runs_medoid_vote} times for each QA pair...")
        for idx, qa_pair in self.qa_pairs.items():
            print("QA Pair: ", idx)
            answers_at_position_for_qa_pair = {position: []}
            for p in range(self.num_runs_medoid_vote):

                answers_at_position_for_qa_pair = self._get_responses_at_positions(qa_pair,
                                            self.task_prompt, self.task_format,
                                            answers_at_position_for_qa_pair, [position])
            
            answers_at_permutation[idx] = {"question": qa_pair.get("question", ""),
                                           "answer": qa_pair.get("answer", ""),
                                           "answer_doc": qa_pair.get("answer_doc", ""),
                                           "long_ctxt": answers_at_position_for_qa_pair[position],
                                        }

        # save response results
        model_name = re.sub("/", "-", self.model_name)
        new_path = os.path.join(self.results_folder_path, "responses", "control")
        if not os.path.exists(new_path): os.makedirs(new_path)
        date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        save_path = os.path.join(new_path, f"maj_vote_{self.num_runs_medoid_vote}_runs_{self.experiment_tag}-{model_name}-{date_time}.json")
        with open(save_path, 'w') as f:
            f.write(json.dumps(answers_at_permutation))

        # get the centroid answer from permutations
        final_response = {position: []}
        for idx, response_dict in tqdm(answers_at_permutation.items()):
            answer_list = [res["llm_response"] for res in response_dict["long_ctxt"]]

            # get embeddings
            embeds = self.embedding_model.model.embed_documents(answer_list)
            embeds = np.array(embeds)
            avg_similarities = compute_avg_similarities(embedding_vectors=embeds)

            max_arg = np.argmax(avg_similarities)
            best_response = answer_list[max_arg]
            final_response[position].append({"question": response_dict["question"],
                                                   "answer": response_dict["answer"],
                                                   "answer_doc": response_dict["answer_doc"],
                                                   "llm_response": best_response})
        
        # save response results
        model_name = re.sub("/", "-", self.model_name)
        date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        save_path = os.path.join(new_path, f"maj_vote_{self.experiment_tag}-{model_name}-{date_time}.json")
        with open(save_path, 'w') as f:
            f.write(json.dumps(answers_at_permutation))

        # evaluate the responses
        print(">>>>Evaluating responses using llm-as-a-judge")
        score_output = self._evaluate_responses(final_response)

        # save score results
        new_path = os.path.join(self.results_folder_path, "outputs", "control")
        if not os.path.exists(new_path): os.makedirs(new_path)
        date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        save_path = os.path.join(new_path, f"maj_vote_test_results_{self.experiment_tag}-{model_name}-{date_time}.json")
        with open(save_path, 'w') as f:
            f.write(json.dumps(score_output))

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        print(f"Majority Vote Test Duration: {test_elapsed_time:.1f} seconds")
        print(f"Results saved at {save_path}")
