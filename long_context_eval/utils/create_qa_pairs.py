import os
import json
from collections import defaultdict
from langchain.schema import HumanMessage
from parameters import prompts_formats as prompts_formats
from parameters.models import OpenAIModel


def create_qa_pairs_single_hop(documents, qa_pairs_path, prompt, format,
                            ):
    '''Creating benchmark questions-answers from individual documents'''
    qa_pairs = defaultdict(dict)

    chat_model = OpenAIModel("gpt-4") #, model_kwargs=dict(temperature=0.0))
    encoding = chat_model.encoding
    for idx, doc in enumerate(documents):
        print(f"Processing document {idx} of {len(documents)}")
        print(idx, doc.metadata["source"])

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
            _, filename = os.path.split(doc.metadata["source"])
            qa = chain.invoke({"context": doc_content})
            qa_pairs[idx] = {"question": qa["question"], "answer": qa["answer"],
                             "answer_doc": filename}
        except:
                print(f"Error creating QA pair for document {idx}")
                continue

    # Writes result to one single QA test file.
    with open(qa_pairs_path, 'w') as json_file:
        json.dump(qa_pairs, json_file)

    return qa_pairs