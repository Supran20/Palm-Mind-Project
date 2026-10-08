# scripts/smoke_chat.py
import asyncio

from redis.asyncio import Redis

from app.core.config import get_settings
from app.llm.groq_client import OpenAICompatibleClient
from app.memory.chat_memory import ChatMemory


async def main() -> None:
    s = get_settings()

    llm = OpenAICompatibleClient(
        s.groq_api_key.get_secret_value(), s.llm_base_url, s.llm_model
    )
    reply = await llm.complete([{"role": "user", "content": "Say hi in 3 words"}])
    print("LLM:", reply.content)

    redis: Redis[str] = Redis.from_url(s.redis_url, decode_responses=True)
    mem = ChatMemory(redis, 20, 60)
    await mem.append("demo", "user", "hello")
    print("Redis:", await mem.get("demo"))
    await redis.aclose()


asyncio.run(main())