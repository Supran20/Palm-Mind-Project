import asyncio
import uuid

from qdrant_client import AsyncQdrantClient

from app.core.config import get_settings
from app.services.embedding import SentenceTransformerEmbedder
from app.services.vector_store import QdrantVectorStore

COLLECTION = "smoke_test"


async def main() -> None:
    settings = get_settings()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    client = AsyncQdrantClient(url=settings.qdrant_url)
    store = QdrantVectorStore(client, COLLECTION)

    await store.ensure_collection(embedder.dimension)
    document_id = uuid.uuid4()
    chunks = [
        "Espresso is made by forcing hot water through finely ground coffee.",
        "The Himalayas contain eight of the ten tallest mountains on Earth.",
        "Pour-over brewing uses a paper filter and a slow stream of water.",
    ]
    vectors = await embedder.embed_documents(chunks)
    await store.upsert_chunks(document_id, "smoke.txt", chunks, vectors)

    query = await embedder.embed_query("What's a good way to make a hot caffeinated drink?")
    for hit in await store.search(query, top_k=3):
        print(f"{hit.score:.3f}  {hit.text}")

    await store.delete_document(document_id)
    await client.delete_collection(COLLECTION)
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())