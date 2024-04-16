from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser


###### Output formats.
class SingleDocQuestionGen(BaseModel):
    question: str = Field(description="question to ask")
    answer: str = Field(description="answer to the question")

class SingleDocQA(BaseModel):
    answer: str = Field(description="answer to the question")

class Title(BaseModel):
    title: str = Field(description="title to the article")

class ScoreQA(BaseModel):
    correct: bool = Field(description="boolean flag if answer and gold standard answer are the same")

###### define the prompt templates
PROMPT_TEMPLATES = {
    "single_doc_question_gen_prompt": """Ask a factoid question given only the context provided, that can be answered in a few words. Answer the question given the context.
    Format your result in a JSON object with keys 'question' and 'answer'.
    Context: {context}
    Result:""",
    "single_doc_qa_prompt": """Answer the question given only the context provided.
    Context: {context}
    Question: {question}
    Answer:""",
    "single_doc_qa_prompt_anthropic": """Answer the question given only the context provided. Just provide the answer without any reasoning.
    Context: {context}
    Question: {question}
    Answer:""",
    # Format your result in a JSON object with keys 'answer'.
    "title_prompt": """For the question provided, return a JSON object with a `title` for the following text. Do not use any special characters in the title.
    Text: {text}
    JSON: """,
    "score_qa_prompt_old": """Return a JSON object with a `correct` key as a boolean, if the given answer and gold standard answer are the same in meaning (may be worded differently).
    Answer: {answer}
    Gold standard answer: {gold_answer}
    JSON:""",
    "score_qa_prompt": """For the question provided, return a JSON object with a `correct` key as a boolean, if the given answer and gold standard answer are the same in meaning (may be worded differently).
    Question: {question}
    Answer: {answer}
    Gold standard answer: {gold_answer}
    JSON:""",
}


def get_prompt_and_format(prompt_str: str):
    if prompt_str == 'single_doc_question_gen_prompt':
        return PromptTemplate(template=PROMPT_TEMPLATES["single_doc_question_gen_prompt"],
                                           input_variables=["context"]), JsonOutputParser(pydantic_object=SingleDocQuestionGen) 
    elif prompt_str == 'single_doc_qa_prompt':
        return PromptTemplate(template=PROMPT_TEMPLATES["single_doc_qa_prompt"],
                                           input_variables=["context", "question"]), StrOutputParser()
    elif prompt_str == 'single_doc_qa_prompt_anthropic':
        system = (
            "You are a question answering assistant."
        )
        human = PROMPT_TEMPLATES["single_doc_qa_prompt_anthropic"]
        return ChatPromptTemplate.from_messages([("system", system), ("human", human)]), StrOutputParser()
    elif prompt_str == 'title_prompt':
        return PromptTemplate(template=PROMPT_TEMPLATES["title_prompt"],
                                           input_variables=["text"]), SimpleJsonOutputParser(pydantic_object=Title)
    elif prompt_str == 'score_qa_prompt_old':
        return PromptTemplate(template=PROMPT_TEMPLATES["score_qa_prompt_old"],
                                           input_variables=["answer", "gold_answer"]), SimpleJsonOutputParser(pydantic_object=ScoreQA)
    elif prompt_str == 'score_qa_prompt':
        return PromptTemplate(template=PROMPT_TEMPLATES["score_qa_prompt"],
                                           input_variables=["question", "answer", "gold_answer"]), SimpleJsonOutputParser(pydantic_object=ScoreQA)
    else:
        raise
