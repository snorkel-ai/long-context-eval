from typing import Optional
from langchain.prompts.prompt import PromptTemplate


###### define the prompt templates
PROMPT_TEMPLATES = {
    "single_doc_question_gen_prompt": """Ask a factoid question given only the context provided. Answer the question given the context.
    Format your result in a JSON object with keys 'question' and 'answer'.
    Context: {context}
    Result:""",
    "single_doc_qa_prompt": """Answer the question given only the context provided.
    Context: {context}
    Question: {question}
    Answer:""",
    # Format your result in a JSON object with keys 'answer'.
    "title_prompt": """Return a JSON object with a `title` for the following text. Do not use any special characters in the title.
    Text: {text}
    JSON: """,
    "score_qa_prompt": """Return a JSON object with a `correct` key as a boolean, if the given answer and gold standard answer are the same in meaning (may be worded differently).
    Answer: {answer}
    Gold standard answer: {gold_answer}
    JSON:""",
}


def get_prompt(prompt_str: str):
    if prompt_str == 'single_doc_question_gen_prompt':
        return PromptTemplate(template=PROMPT_TEMPLATES["single_doc_question_gen_prompt"],
                                           input_variables=["context"])
    elif prompt_str == 'single_doc_qa_prompt':
        return PromptTemplate(template=PROMPT_TEMPLATES["single_doc_qa_prompt"],
                                           input_variables=["context", "question"])
    elif prompt_str == 'title_prompt':
        return PromptTemplate(template=PROMPT_TEMPLATES["title_prompt"],
                                           input_variables=["text"])
    elif prompt_str == 'score_qa_prompt':
        return PromptTemplate(template=PROMPT_TEMPLATES["score_qa_prompt"],
                                           input_variables=["answer", "gold_answer"])
    else:
        raise
