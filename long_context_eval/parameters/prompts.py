from langchain.prompts.prompt import PromptTemplate


###### define the prompt templates
prompt_templates = {
    "single_doc_question_gen_prompt": """Ask a factoid question given only the context provided. Answer the question given the context.
    Format your result in a JSON object with keys 'question' and 'answer'.
    Context: {context}
    Result:""",
    "single_doc_qa_prompt": """Answer the question given only the context provided. Format your result in a JSON object with keys 'answer'.
    Context: {context}
    Question: {question}
    Result:""",
    "title_prompt": """Return a JSON object with a `title` for the following text. Do not use any special characters in the title.
    Text: {text}
    JSON: """,
    "score_qa_prompt": """Return a JSON object with a `correct` key as a boolean, if the given answer and gold standard answer are the same in meaning (may be worded differently).
    Answer: {answer}
    Gold standard answer: {gold_answer}
    JSON:""",
}

SINGLEHOP_QUESTION_PROMPT = PromptTemplate(template=prompt_templates["single_doc_question_gen_prompt"],
                                           input_variables=["context"])
SINGLEHOP_QA_PROMPT = PromptTemplate(template=prompt_templates["single_doc_qa_prompt"],
                                           input_variables=["context", "question"])
TITLE_PROMPT = PromptTemplate(template=prompt_templates["title_prompt"],
                                           input_variables=["text"])
SCORE_QA_PROMPT = PromptTemplate(template=prompt_templates["score_qa_prompt"],
                                           input_variables=["answer", "gold_answer"])
