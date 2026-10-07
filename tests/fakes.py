import hashlib
import math
import uuid
from collections.abc import Sequence

from app.services.embedding import Vector
from app.services.vector_store import SearchHit


class FakeEmbedder:
    """Deterministic embedder: the same text always gives the same vector."""

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> Vector:
        return self._embed(text)

    def _embed(self, text: str) -> Vector:
        digest = hashlib.sha256(text.encode()).digest()
        raw = [digest[i] / 255 for i in range(self._dimension)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]


class InMemoryVectorStore:
    """Dict-backed stand-in for Qdrant, with switches to simulate failures."""

    def __init__(self) -> None:
        self._rows: dict[tuple[uuid.UUID, int], tuple[Vector, str, str]] = {}
        self.deleted: list[uuid.UUID] = []
        self.fail_on_upsert = False

    async def ensure_collection(self, vector_size: int) -> None:
        return None

    async def upsert_chunks(
        self,
        document_id: uuid.UUID,
        filename: str,
        chunks: Sequence[str],
        vectors: Sequence[Vector],
    ) -> None:
        if self.fail_on_upsert:
            raise RuntimeError("vector store unavailable")
        for index, (text, vector) in enumerate(zip(chunks, vectors, strict=True)):
            self._rows[(document_id, index)] = (list(vector), text, filename)

    async def delete_document(self, document_id: uuid.UUID) -> None:
        self.deleted.append(document_id)
        for key in [key for key in self._rows if key[0] == document_id]:
            del self._rows[key]

    async def search(self, vector: Vector, top_k: int = 5) -> list[SearchHit]:
        scored = [
            (sum(a * b for a, b in zip(vector, stored, strict=True)), key, text, filename)
            for key, (stored, text, filename) in self._rows.items()
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchHit(
                text=text,
                score=score,
                document_id=key[0],
                filename=filename,
                chunk_index=key[1],
            )
            for score, key, text, filename in scored[:top_k]
        ]

    def count(self, document_id: uuid.UUID) -> int:
        return sum(1 for key in self._rows if key[0] == document_id)