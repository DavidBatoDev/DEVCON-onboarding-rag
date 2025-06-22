from fastapi import APIRouter
from app.api.v1.models import AskRequest, AskResponse
from app.services.rag_engine import ask_with_rag

router = APIRouter()

@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    response = ask_with_rag(request.query)
    return AskResponse(answer=response)