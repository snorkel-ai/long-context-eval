import os
import json
import tiktoken
from langchain.schema import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from long_context_eval.parameters.formats import SingleDocQuestionGen
from long_context_eval.parameters.prompts import SINGLEHOP_QUESTION_PROMPT
from long_context_eval.parameters.models import OpenAIModel


def create_qa_pairs_single_hop(documents):
    '''Creating benchmark questions-answers from individual documents'''

    # Set up a parser + inject instructions into the prompt template.
    parser_question_gen = JsonOutputParser(pydantic_object=SingleDocQuestionGen)    

    qa_pairs = {}
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    for idx, doc in enumerate(documents):
        # print(f"Processing document {idx} of {len(documents)}")
        chat_model = OpenAIModel()
        chain = SINGLEHOP_QUESTION_PROMPT | chat_model.model | parser_question_gen
        
        num_token = chat_model.model.get_num_tokens_from_messages(messages=[
        HumanMessage(content=SINGLEHOP_QUESTION_PROMPT.format(context=doc.page_content))
    ])
        if num_token > chat_model.token_limit:
            doc_content = encoding.decode(encoding.encode(
                SINGLEHOP_QUESTION_PROMPT.format(
                    context=doc.page_content))[:chat_model.token_limit])
        else:
            doc_content = doc.page_content
        qa = chain.invoke({"context": doc_content})
        qa_pairs[idx] = {"id": idx, "file": doc.metadata["source"], 
                         "question": qa["question"], "answer": qa["answer"]}

    # Writes result to one single QA test file.
    with open("./data.json", 'w') as f:
        for key, value in qa_pairs.items():
            f.write(json.dumps(value) + "\n")