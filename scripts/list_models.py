    # scripts/list_models.py
import asyncio

from openai import AsyncOpenAI

from app.core.config import get_settings


async def main() -> None:
    s = get_settings()
    client = AsyncOpenAI(
        api_key=s.groq_api_key.get_secret_value(), base_url=s.llm_base_url
    )
    models = await client.models.list()
    for m in sorted(models.data, key=lambda x: x.id):
        print(m.id)


asyncio.run(main())