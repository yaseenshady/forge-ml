"""OpenAI / GPT provider adapter."""
from __future__ import annotations

import openai as oai

from forge_ml.utils.config import config
from .base import AgentMessage, AgentResult, BaseProvider


class OpenAIProvider(BaseProvider):
    name = "openai"
    _DEFAULT_MODEL = "gpt-4o"

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self.model = model
        self._client: oai.AsyncOpenAI | None = None

    def _get_client(self) -> oai.AsyncOpenAI:
        if self._client is None:
            self._client = oai.AsyncOpenAI(api_key=config.openai_api_key)
        return self._client

    def is_available(self) -> bool:
        return bool(config.openai_api_key)

    async def complete(
        self,
        messages: list[AgentMessage],
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> AgentResult:
        client = self._get_client()
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages += [{"role": m.role, "content": m.content} for m in messages]

        response = await client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content or ""
        usage = response.usage
        total_tokens = usage.total_tokens if usage else 0
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # Approximate GPT-4o pricing
        cost = (input_tokens / 1_000_000) * 5.0 + (output_tokens / 1_000_000) * 15.0

        return AgentResult(
            content=content,
            provider=self.name,
            model=self.model,
            tokens_used=total_tokens,
            cost_usd=cost,
            raw=response,
        )
