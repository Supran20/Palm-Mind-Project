from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from app.api.routes.chat import router as chat_router
from app.api.routes.ingestion import router as ingestion_router
from app.core.config import get_settings
from app.db import booking as _booking  # noqa: F401  (registers the table)
from app.db.session import Base, create_engine, create_session_factory
from app.llm.groq_client import OpenAICompatibleClient
from app.services.booking import SqlBookingStore
from app.services.chat import build_chat_service
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
    session_factory = create_session_factory(engine)

    redis: Redis[str] = Redis.from_url(settings.redis_url, decode_responses=True)
    llm = OpenAICompatibleClient(
        settings.groq_api_key.get_secret_value(),
        settings.llm_base_url,
        settings.llm_model,
    )

    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.session_factory = session_factory
    app.state.chat_service = build_chat_service(
        llm=llm,
        redis=redis,
        embedder=embedder,
        vector_store=vector_store,
        booking_store=SqlBookingStore(session_factory),
        history_limit=settings.chat_history_limit,
        ttl_seconds=settings.chat_ttl_seconds,
        top_k=settings.retrieval_top_k,
        min_score=settings.retrieval_min_score,
    )

    yield

    await redis.aclose()
    await qdrant.close()
    await engine.dispose()


app = FastAPI(title="Palm Mind RAG API", version="0.1.0", lifespan=lifespan)
app.include_router(ingestion_router)
app.include_router(chat_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}