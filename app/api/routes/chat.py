from typing import Annotated

import openai
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.services.chat import ChatService

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=2000)


class SourceOut(BaseModel):
    filename: str
    chunk_index: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    session_id: str


def get_chat_service(request: Request) -> ChatService:
    service: ChatService = request.app.state.chat_service
    return service


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    try:
        result = await service.handle(body.session_id, body.message)
    except openai.APIError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The language model service is unavailable. Please try again.",
        ) from exc
    return ChatResponse(
        answer=result.answer,
        sources=[SourceOut(**vars(s)) for s in result.sources],
        session_id=result.session_id,
    )