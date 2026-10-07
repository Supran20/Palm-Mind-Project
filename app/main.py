from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient

from app.api.routes.ingestion import router as ingestion_router
from app.core.config import get_settings
from app.db.session import Base, create_engine, create_session_factory
from app.services.embedding import SentenceTransformerEmbedder
from app.services.vector_store import QdrantVectorStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    embedder = SentenceTransformerEmbedder(settings.embedding_model)

    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    vector_store = QdrantVectorStore(qdrant, settings.qdrant_collection)
    await vector_store.ensure_collection(embedder.dimension)

    engine = create_engine(settings.database_url)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.session_factory = create_session_factory(engine)

    yield

    await qdrant.close()
    await engine.dispose()


app = FastAPI(title="Palm Mind RAG API", version="0.1.0", lifespan=lifespan)
app.include_router(ingestion_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}