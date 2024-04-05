import os
import json
from collections import defaultdict
import tiktoken
from langchain.schema import HumanMessage
from parameters import prompts_formats as prompts_formats
from parameters.models import OpenAIModel


def create_qa_pairs_single_hop(documents, qa_pairs_path, prompt, format):
    '''Creating benchmark questions-answers from individual documents'''

    qa_pairs = defaultdict(dict)
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    for idx, doc in enumerate(documents):
        print(f"Processing document {idx} of {len(documents)}")
        print(idx, doc.metadata["source"])
        chat_model = OpenAIModel()
        chain = prompt | chat_model.model | format
        
        num_token = chat_model.model.get_num_tokens_from_messages(messages=[
        HumanMessage(content=prompt.format(context=doc.page_content))
    ])
        if num_token > chat_model.max_context_size:
            doc_content = encoding.decode(encoding.encode(
                prompt.format(
                    context=doc.page_content))[:chat_model.max_context_size])
        else:
            doc_content = doc.page_content

        try:
            qa = chain.invoke({"context": doc_content})
            qa_pairs[doc.metadata["source"]] = {"question": qa["question"], "answer": qa["answer"]}
        except:
                print(f"Error creating QA pair for document {idx}")
                continue

    # Writes result to one single QA test file.
    with open(qa_pairs_path, 'w') as json_file:
        json.dump(qa_pairs, json_file)

    return qa_pairs