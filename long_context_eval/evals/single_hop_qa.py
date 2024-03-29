import os
import time
import json
from random import shuffle
from typing import Optional
import tiktoken
from langchain.schema import HumanMessage
from langchain_community.vectorstores import Chroma
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
                 model_kwargs: Optional[dict] = dict(temperature=0.8),
                 chunk_size: Optional[int] = 1000,
                 chunk_overlap: Optional[int] = 200,
                 search_kwargs: Optional[dict] = {"k": 10},
                 embedding_model_name: Optional[str] = 'text-embedding-ada-002',
                 embedding_model_kwargs: Optional[dict] = {}):
        self.model_name = model_name
        self.data_path = data_path
        self.model_kwargs = model_kwargs
        # RAG parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.search_kwargs = search_kwargs
        self.embedding_model_kwargs = embedding_model_kwargs

        # Get the correct model based on model name
        self.model = models.SUPPORTED_MODELS[self.model_name](self.model_name, self.model_kwargs)
        self.encoding = tiktoken.encoding_for_model(self.model_name) ## TBD: Update to get encoding for any provider model
        
        # RAG embedding model
        self.embedding_model_name = embedding_model_name
        self.embedding_model = models.SUPPORTED_MODELS[self.embedding_model_name](self.embedding_model_name,
                                                                                  self.embedding_model_kwargs)
        self.depths = []
        self.documents = []

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
            print(f"Truncated # of documents to {len(documents)} (context window token limits reached)")
        return documents

    def _test_position_at_depth(self, depth, documents, qa_pairs,
                                prompt, formatter):
        print(f"Depth: {depth}")
        # for each doc, generate llm answer to the question
        answers = {}
        for i in range(len(documents)):
            docs_copy = documents[:]
            
            idx = qa_pairs[str(i)]["id"]
            q = qa_pairs[str(i)]["question"]
            a = qa_pairs[str(i)]["answer"]
            f = qa_pairs[str(i)]["file"]

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
            HumanMessage(content=prompts.SINGLEHOP_QA_PROMPT.format(context=context, question=q))
        ])
            if num_token > self.model.token_limit-100:
                doc_content = self.encoding.decode(self.encoding.encode(
                    prompts.SINGLEHOP_QA_PROMPT.format(context=context, question=q))[:self.model.token_limit-100])
            else:
                doc_content = context

            try:
                qa = chain.invoke({"context": doc_content, "question": q})
                answers[idx] = {"id": idx, "file": f, "question": q, "answer": qa["answer"], "gold_answer": a,
                                "depth": depth, "context_length": num_token, "model": self.model_name,
                                "model_kwargs": self.model_kwargs, }
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
                scored_output_at_depth[idx]["score"] = int(score_response["correct"])
                scores.append(score_response["correct"])
            score_output[depth] = scored_output_at_depth
            print(f"Accuracy at depth {depth}: {sum(scores)/len(scores)*100:.1f}%")
        return score_output
    
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

    def test_position_single_hop(self):
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

        # fill up the context window, and truncate if necessary
        self.documents = self._get_or_truncate_context_window(self.documents)

        #### Test for position
        print(f"Run position test on long context for {self.model_name}")
        # get number of docs at each depth for test
        self.depths = {i: int(i*len(self.documents)/100) for i in range(10, 100, 10)}

        # iterate at each depth for the test, generate responses to questions
        print(">>>>Generate llm responses at document depths...")
        answers_at_depth = {}
        for depth in list(self.depths.keys()):
            answers = self._test_position_at_depth(depth, self.documents, qa_pairs,
                                         prompt, formatter)
            answers_at_depth[depth] = answers
        # TBD: Should save answers_at_depth results

        # evaluate the responses
        print(">>>>Evaluating responses using llm-as-a-judge")
        score_output = self._evaluate_responses(answers_at_depth)
        
        # save score results
        if not os.path.exists("./output"): os.makedirs("./output")
        save_path = f"./output/position_test_results_{self.model_name}.json"
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

            qa = rag_chain.invoke({"context": retriever | format_docs, "question": q})
            rag_answers[idx] = {"id": idx, "file": f, "question": q, "answer": qa["answer"], "gold_answer": a,
                                "model": self.model_name, "model_kwargs": self.model_kwargs, 
                                "embedding_model": self.embedding_model_name, "embedding_model_kwargs": self.embedding_model_kwargs,
                                "chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap,
                                "search_kwargs": self.search_kwargs}

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
