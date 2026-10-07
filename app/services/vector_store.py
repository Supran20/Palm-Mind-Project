import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.services.embedding import Vector

UPSERT_BATCH_SIZE = 128


@dataclass(frozen=True, slots=True)
class SearchHit:
    text: str
    score: float
    document_id: uuid.UUID
    filename: str
    chunk_index: int


def build_point_id(document_id: uuid.UUID, chunk_index: int) -> uuid.UUID:
    """Deterministic ID: re-ingesting the same document overwrites instead of duplicating."""
    return uuid.uuid5(document_id, str(chunk_index))


class QdrantVectorStore:
    def __init__(self, client: AsyncQdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    async def ensure_collection(self, vector_size: int) -> None:
        """Create the collection (and an index on document_id) if it doesn't exist."""
        if await self._client.collection_exists(self._collection):
            return
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        await self._client.create_payload_index(
            collection_name=self._collection,
            field_name="document_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    async def upsert_chunks(
        self,
        document_id: uuid.UUID,
        filename: str,
        chunks: Sequence[str],
        vectors: Sequence[Vector],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        points = [
            PointStruct(
                id=str(build_point_id(document_id, index)),
                vector=list(vector),
                payload={
                    "document_id": str(document_id),
                    "filename": filename,
                    "chunk_index": index,
                    "text": text,
                },
            )
            for index, (text, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        for start in range(0, len(points), UPSERT_BATCH_SIZE):
            await self._client.upsert(
                collection_name=self._collection,
                points=points[start : start + UPSERT_BATCH_SIZE],
                wait=True,
            )

    async def delete_document(self, document_id: uuid.UUID) -> None:
        """Remove every chunk of a document (used to roll back a failed ingestion)."""
        await self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=str(document_id)),
                        )
                    ]
                )
            ),
            wait=True,
        )

    async def search(self, vector: Vector, top_k: int = 5) -> list[SearchHit]:
        response = await self._client.query_points(
            collection_name=self._collection,
            query=list(vector),
            limit=top_k,
            with_payload=True,
        )
        hits: list[SearchHit] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                SearchHit(
                    text=str(payload["text"]),
                    score=point.score,
                    document_id=uuid.UUID(str(payload["document_id"])),
                    filename=str(payload["filename"]),
                    chunk_index=int(payload["chunk_index"]),
                )
            )
        return hits


class VectorStore(Protocol):
    async def ensure_collection(self, vector_size: int) -> None: ...

    async def upsert_chunks(
        self,
        document_id: uuid.UUID,
        filename: str,
        chunks: Sequence[str],
        vectors: Sequence[Vector],
    ) -> None: ...

    async def delete_document(self, document_id: uuid.UUID) -> None: ...

    async def search(self, vector: Vector, top_k: int = 5) -> list[SearchHit]: ...