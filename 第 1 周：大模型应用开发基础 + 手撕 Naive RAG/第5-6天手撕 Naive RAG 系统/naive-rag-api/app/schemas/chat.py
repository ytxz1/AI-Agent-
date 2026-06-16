from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    chunk_index: int
    content_preview: str
    score: float | None = None


class RetrievedChunk(BaseModel):
    content: str
    metadata: dict
    score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    retrieved_chunks: list[RetrievedChunk]

