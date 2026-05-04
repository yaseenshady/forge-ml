"""Anthropic Claude provider adapter with prompt caching."""
from __future__ import annotations

import anthropic

from forge_ml.utils.config import config
from .base import AgentMessage, AgentResult, BaseProvider


class ClaudeProvider(BaseProvider):
    name = "claude"
    _DEFAULT_MODEL = "claude-sonnet-4-5"

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self.model = model
        self._client: anthropic.AsyncAnthropic | None = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)
        return self._client

    def is_available(self) -> bool:
        return bool(config.anthropic_api_key)

    async def complete(
        self,
        messages: list[AgentMessage],
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> AgentResult:
        client = self._get_client()

        # Enable prompt caching on the system prompt if present
        sys_blocks: list[dict] = []
        if system:
            sys_blocks = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        response = await client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=sys_blocks or anthropic.NOT_GIVEN,
            messages=api_messages,
        )

        content = response.content[0].text if response.content else ""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens

        # Approximate cost (Sonnet pricing)
        cost = (input_tokens / 1_000_000) * 3.0 + (output_tokens / 1_000_000) * 15.0

        return AgentResult(
            content=content,
            provider=self.name,
            model=self.model,
            tokens_used=total_tokens,
            cost_usd=cost,
            raw=response,
        )
