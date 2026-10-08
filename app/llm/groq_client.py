import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from openai import AsyncOpenAI

from app.llm.base import LLMResponse, ToolCall


class OpenAICompatibleClient:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": list(messages),
            "temperature": 0.2,
        }
        if tools:
            kwargs["tools"] = list(tools)
            kwargs["tool_choice"] = "auto"

        resp = await self._client.chat.completions.create(**cast(Any, kwargs))
        msg = resp.choices[0].message
        calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments or "{}"),
            )
            for tc in (msg.tool_calls or [])
            if tc.type == "function"
        ]
        return LLMResponse(content=msg.content, tool_calls=calls)