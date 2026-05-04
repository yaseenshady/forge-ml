"""Abstract base for all AI provider adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class AgentMessage:
    role: str  # "user" | "assistant"
    content: str
    provider: str = ""
    metadata: dict[str, Any] | None = None


@dataclass
class AgentResult:
    content: str
    provider: str
    model: str
    tokens_used: int = 0
    cost_usd: float = 0.0
    raw: Any = None


class BaseProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def complete(
        self,
        messages: list[AgentMessage],
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> AgentResult: ...

    @abstractmethod
    def is_available(self) -> bool: ...
