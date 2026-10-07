import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.services.chunking import ChunkingStrategy
from app.services.extraction import UnsupportedFileTypeError
from app.services.ingestion import IngestionService, InvalidChunkingConfigError
from tests.fakes import InMemoryVectorStore


async def count_documents(session: AsyncSession) -> int:
    result = await session.execute(select(func.count()).select_from(Document))
    return result.scalar_one()


async def test_paragraph_ingest_saves_vectors_and_metadata(
    service: IngestionService, session: AsyncSession, store: InMemoryVectorStore
) -> None:
    content = b"First paragraph about coffee.\n\nSecond paragraph about mountains."
    document = await service.ingest("notes.txt", content, ChunkingStrategy.PARAGRAPH, 40, 0)

    assert document.num_chunks == 2
    assert store.count(document.id) == 2
    saved = await session.get(Document, document.id)
    assert saved is not None
    assert saved.chunking_strategy == "paragraph"
    assert saved.chunk_overlap is None


async def test_fixed_ingest_records_overlap(
    service: IngestionService, store: InMemoryVectorStore
) -> None:
    document = await service.ingest("a.txt", b"a" * 250, ChunkingStrategy.FIXED, 100, 20)

    assert document.num_chunks == 3
    assert document.chunk_overlap == 20
    assert store.count(document.id) == 3


async def test_filename_is_stripped_of_directories(service: IngestionService) -> None:
    document = await service.ingest(
        "../../etc/evil.txt", b"hello there", ChunkingStrategy.FIXED, 100, 0
    )
    assert document.filename == "evil.txt"


async def test_unsupported_type_is_rejected(service: IngestionService) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        await service.ingest("photo.png", b"data", ChunkingStrategy.FIXED, 100, 0)


async def test_bad_chunk_config_writes_nothing(
    service: IngestionService, session: AsyncSession
) -> None:
    with pytest.raises(InvalidChunkingConfigError):
        await service.ingest("a.txt", b"hello", ChunkingStrategy.FIXED, 100, 100)
    assert await count_documents(session) == 0


async def test_vector_store_failure_leaves_no_metadata(
    service: IngestionService, session: AsyncSession, store: InMemoryVectorStore
) -> None:
    store.fail_on_upsert = True
    with pytest.raises(RuntimeError, match="vector store unavailable"):
        await service.ingest("a.txt", b"hello world", ChunkingStrategy.FIXED, 100, 0)
    assert await count_documents(session) == 0


async def test_database_failure_removes_stored_vectors(
    service: IngestionService,
    session: AsyncSession,
    store: InMemoryVectorStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_commit() -> None:
        raise RuntimeError("db down")

    monkeypatch.setattr(session, "commit", failing_commit)

    with pytest.raises(RuntimeError, match="db down"):
        await service.ingest("a.txt", b"hello world", ChunkingStrategy.FIXED, 100, 0)

    assert len(store.deleted) == 1
    assert store.count(store.deleted[0]) == 0
    assert await count_documents(session) == 0