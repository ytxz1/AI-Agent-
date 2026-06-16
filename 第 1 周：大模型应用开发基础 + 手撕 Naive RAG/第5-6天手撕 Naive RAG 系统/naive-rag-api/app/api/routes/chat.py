from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import rag_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/query", response_model=ChatResponse)
def query(request: ChatRequest) -> ChatResponse:
    return rag_service.answer(question=request.question, top_k=request.top_k)

