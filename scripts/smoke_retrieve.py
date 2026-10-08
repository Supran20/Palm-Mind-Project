import asyncio
import sys

from qdrant_client import AsyncQdrantClient

from app.core.config import get_settings
from app.llm.groq_client import OpenAICompatibleClient
from app.services.embedding import SentenceTransformerEmbedder
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
    # min_score=0.0 so we can SEE every score and choose a good cutoff
    retriever = Retriever(embedder, store, llm, top_k=4, min_score=0.0)

    # Test 1: standalone question
    print(f"\n== Q: {question}")
    for c in await retriever.retrieve(question, history=[]):
        print(f"{c.score:.3f}  [{c.filename} #{c.chunk_index}]  {c.text[:100]!r}")

    # Test 2: follow-up that only makes sense with history
    history = [
        {"role": "user", "content": "Where did he study?"},
        {"role": "assistant", "content": "He studied Computer Engineering at Himalaya College of Engineering."},
    ]
    follow_up = "And where has he worked?"
    rewritten = await retriever.standalone_question(follow_up, history)
    print(f"\n== Follow-up: {follow_up!r}\n   Rewritten: {rewritten!r}")
    for c in await retriever.retrieve(follow_up, history):
        print(f"{c.score:.3f}  [{c.filename} #{c.chunk_index}]  {c.text[:100]!r}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "Where did he study?"))