import json
from redis.asyncio import Redis


class ChatMemory:
    """Stores only user/assistant text turns, capped and expiring."""

    def __init__(self, redis: "Redis[str]", max_messages: int, ttl_seconds: int) -> None:
        self._redis = redis
        self._max = max_messages
        self._ttl = ttl_seconds

    @staticmethod
    def _key(session_id: str) -> str:
        return f"chat:{session_id}"

    async def get(self, session_id: str) -> list[dict[str, str]]:
        raw = await self._redis.lrange(self._key(session_id), 0, -1)
        return [json.loads(item) for item in raw]

    async def append(self, session_id: str, role: str, content: str) -> None:
        key = self._key(session_id)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.rpush(key, json.dumps({"role": role, "content": content}))
            pipe.ltrim(key, -self._max, -1)
            pipe.expire(key, self._ttl)
            await pipe.execute()