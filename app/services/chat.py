import json
from dataclasses import dataclass
from datetime import date
from typing import Any

from redis.asyncio import Redis

from app.llm.base import LLMClient, LLMResponse
from app.memory.chat_memory import ChatMemory
from app.services.booking import BOOK_TOOL, BookingService, BookingStore
from app.services.embedding import Embedder
from app.services.prompt import build_messages
from app.services.retrieval import Retriever
from app.services.vector_store import VectorStore

MAX_TOOL_ROUNDS = 3
FALLBACK_ANSWER = "Sorry, I couldn't complete that. Please try rephrasing."


@dataclass(frozen=True)
class Source:
    filename: str
    chunk_index: int
    score: float


@dataclass(frozen=True)
class ChatResult:
    answer: str
    sources: list[Source]
    session_id: str


def _assistant_tool_message(response: LLMResponse) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": response.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                },
            }
            for call in response.tool_calls
        ],
    }


class ChatService:
    def __init__(
        self,
        llm: LLMClient,
        memory: ChatMemory,
        retriever: Retriever,
        booking: BookingService,
    ) -> None:
        self._llm = llm
        self._memory = memory
        self._retriever = retriever
        self._booking = booking

    async def _run_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "book_interview":
            return await self._booking.book(arguments)
        return f"Unknown tool '{name}'."

    async def handle(self, session_id: str, message: str) -> ChatResult:
        history = await self._memory.get(session_id)
        chunks = await self._retriever.retrieve(message, history)
        messages = build_messages(history, chunks, message, date.today())

        used_booking_tool = False
        response = await self._llm.complete(messages, tools=[BOOK_TOOL])
        for _ in range(MAX_TOOL_ROUNDS):
            if not response.tool_calls:
                break
            used_booking_tool = True
            messages.append(_assistant_tool_message(response))
            for call in response.tool_calls:
                result = await self._run_tool(call.name, call.arguments)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
            response = await self._llm.complete(messages, tools=[BOOK_TOOL])

        answer = (response.content or "").strip() or FALLBACK_ANSWER

        await self._memory.append(session_id, "user", message)
        await self._memory.append(session_id, "assistant", answer)

        sources = (
            []
            if used_booking_tool
            else [Source(c.filename, c.chunk_index, c.score) for c in chunks]
        )
        return ChatResult(answer=answer, sources=sources, session_id=session_id)


def build_chat_service(
    *,
    llm: LLMClient,
    redis: "Redis[str]",
    embedder: Embedder,
    vector_store: VectorStore,
    booking_store: BookingStore,
    history_limit: int,
    ttl_seconds: int,
    top_k: int,
    min_score: float,
) -> ChatService:
    return ChatService(
        llm=llm,
        memory=ChatMemory(redis, history_limit, ttl_seconds),
        retriever=Retriever(embedder, vector_store, llm, top_k, min_score),
        booking=BookingService(booking_store),
    )