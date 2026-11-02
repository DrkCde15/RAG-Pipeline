from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=4000)


class ChatSource(BaseModel):
    document_id: str
    source: str
    filename: str
    page: int | None = None
    chunk_id: str
    score: float
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]
    retrieval_scores: list[float]
    processing_time: float
