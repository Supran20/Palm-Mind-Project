import asyncio
import sys

from qdrant_client import AsyncQdrantClient

from app.core.config import get_settings
from app.services.embedding import SentenceTransformerEmbedder
from app.services.vector_store import QdrantVectorStore


async def main(question: str) -> None:
    settings = get_settings()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    client = AsyncQdrantClient(url=settings.qdrant_url)
    store = QdrantVectorStore(client, settings.qdrant_collection)

    query = await embedder.embed_query(question)
    for hit in await store.search(query, top_k=3):
        print(f"{hit.score:.3f}  [{hit.filename} #{hit.chunk_index}]  {hit.text[:150]!r}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "What backend technologies are used?"))