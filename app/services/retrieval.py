from collections.abc import Sequence
from dataclasses import dataclass

from app.llm.base import LLMClient
from app.services.embedding import Embedder
from app.services.vector_store import VectorStore

REWRITE_SYSTEM_PROMPT = (
    "You rewrite a user's latest message into a standalone search query. "
    "Use the conversation to replace pronouns and references like 'the second one' "
    "or 'what about him' with the actual subject. "
    "If the message is already standalone, return it unchanged. "
    "Output ONLY the query text, nothing else."
)

HISTORY_WINDOW = 6  # last N messages are enough to resolve references


@dataclass(frozen=True)
class RetrievedChunk:
    filename: str
    chunk_index: int
    text: str
    score: float


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLMClient,
        top_k: int,
        min_score: float,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._top_k = top_k
        self._min_score = min_score

    async def standalone_question(
        self, question: str, history: Sequence[dict[str, str]]
    ) -> str:
        """Turn a follow-up into a self-contained question using the history."""
        if not history:
            return question

        convo = "\n".join(
            f"{m['role']}: {m['content']}" for m in history[-HISTORY_WINDOW:]
        )
        response = await self._llm.complete(
            [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Conversation:\n{convo}\n\nLatest message: {question}",
                },
            ]
        )
        rewritten = (response.content or "").strip()
        return rewritten or question

    async def retrieve(
        self, question: str, history: Sequence[dict[str, str]]
    ) -> list[RetrievedChunk]:
        query = await self.standalone_question(question, history)
        vector = await self._embedder.embed_query(query)
        hits = await self._store.search(vector, top_k=self._top_k)
        return [
            RetrievedChunk(
                filename=hit.filename,
                chunk_index=hit.chunk_index,
                text=hit.text,
                score=hit.score,
            )
            for hit in hits
            if hit.score >= self._min_score
        ]