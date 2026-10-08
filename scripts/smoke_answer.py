import asyncio
import sys
from datetime import date

from qdrant_client import AsyncQdrantClient

from app.core.config import get_settings
from app.llm.groq_client import OpenAICompatibleClient
from app.services.embedding import SentenceTransformerEmbedder
from app.services.prompt import build_messages
from app.services.retrieval import Retriever
from app.services.vector_store import QdrantVectorStore


async def main(question: str) -> None:
    s = get_settings()
    embedder = SentenceTransformerEmbedder(s.embedding_model)
    client = AsyncQdrantClient(url=s.qdrant_url)
    store = QdrantVectorStore(client, s.qdrant_collection)
    llm = OpenAICompatibleClient(
        s.groq_api_key.get_secret_value(), s.llm_base_url, s.llm_model
    )
    retriever = Retriever(
        embedder, store, llm, s.retrieval_top_k, s.retrieval_min_score
    )

    history: list[dict[str, str]] = []
    chunks = await retriever.retrieve(question, history)
    print(f"Retrieved {len(chunks)} chunk(s) above cutoff")

    messages = build_messages(history, chunks, question, date.today())
    response = await llm.complete(messages)
    print(f"\nQ: {question}\nA: {response.content}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "Where did he study?"))