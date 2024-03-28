from langchain_core.pydantic_v1 import BaseModel, Field


###### Define your desired output data structure.
class SingleDocQuestionGen(BaseModel):
    question: str = Field(description="question to ask")
    answer: str = Field(description="answer to the question")

class SingleDocQA(BaseModel):
    answer: str = Field(description="answer to the question")

class Title(BaseModel):
    title: str = Field(description="title to the article")

class ScoreQA(BaseModel):
    correct: bool = Field(description="boolean flag if answer and gold standard answer are the same")
