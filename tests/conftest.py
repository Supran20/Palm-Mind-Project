from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.routes.ingestion import router
from app.core.config import Settings, get_settings
from app.core.dependencies import get_ingestion_service
from app.db.session import Base
from app.models import document as _document_model  # noqa: F401  (registers the table)
from app.services.ingestion import IngestionService
from tests.fakes import FakeEmbedder, InMemoryVectorStore


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as db_session:
        yield db_session
    await engine.dispose()


@pytest.fixture
def service(
    session: AsyncSession, embedder: FakeEmbedder, store: InMemoryVectorStore
) -> IngestionService:
    return IngestionService(session, embedder, store)


@pytest.fixture
async def client(service: IngestionService) -> AsyncIterator[AsyncClient]:
    def override_service() -> IngestionService:
        return service

    def override_settings() -> Settings:
        return Settings(
            database_url="sqlite+aiosqlite:///:memory:",
            qdrant_url="http://unused",
            redis_url="redis://unused",
            max_upload_mb=1,
        )

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_ingestion_service] = override_service
    app.dependency_overrides[get_settings] = override_settings

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        yield http