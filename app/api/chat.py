from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.schemas import AskRequest, AskResponse
from app.services.rag_service import RAGService


router = APIRouter(prefix="/ask", tags=["RAG"])

settings = get_settings()
rag = RAGService(settings)


@router.post("", response_model=AskResponse)
def ask_question(request: AskRequest):
    try:
        top_k = request.top_k or settings.top_k
        result = rag.answer(request.question, top_k)
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"RAG query failed: {exc}",
        ) from exc
