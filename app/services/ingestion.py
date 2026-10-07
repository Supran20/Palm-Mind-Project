import asyncio
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.services.chunking import (
    ChunkingStrategy,
    get_chunker,
)
from app.services.embedding import Embedder
from app.services.extraction import EmptyDocumentError, extract_text
from app.services.vector_store import VectorStore


class InvalidChunkingConfigError(Exception):
    """Raised when chunk_size / chunk_overlap are inconsistent."""


class IngestionService:
    def __init__(
        self,
        session: AsyncSession,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        self._session = session
        self._embedder = embedder
        self._vector_store = vector_store

    async def ingest(
        self,
        filename: str,
        content: bytes,
        strategy: ChunkingStrategy,
        chunk_size: int,
        chunk_overlap: int,
    ) -> Document:
        safe_name = Path(filename).name

        try:
            chunker = get_chunker(strategy, chunk_size, chunk_overlap)
        except ValueError as exc:
            raise InvalidChunkingConfigError(str(exc)) from exc

        text = await asyncio.to_thread(extract_text, safe_name, content)
        chunks = chunker.chunk(text)
        if not chunks:
            raise EmptyDocumentError("The document produced no chunks.")

        vectors = await self._embedder.embed_documents(chunks)

        document = Document(
            id=uuid.uuid4(),
            filename=safe_name,
            file_type=Path(safe_name).suffix.lower(),
            file_size_bytes=len(content),
            chunking_strategy=strategy.value,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap if strategy is ChunkingStrategy.FIXED else None,
            num_chunks=len(chunks),
        )

        try:
            await self._vector_store.upsert_chunks(document.id, safe_name, chunks, vectors)
            self._session.add(document)
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            await self._vector_store.delete_document(document.id)
            raise

        return document