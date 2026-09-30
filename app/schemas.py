from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


class ContextChunk(BaseModel):
    page: int
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    retrieved_context: list[ContextChunk]
