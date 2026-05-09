"""Base class for CLI-subprocess LLM providers (claude, codex, gh copilot)."""
from __future__ import annotations

import asyncio
import shutil
import time
from abc import abstractmethod

from .base import AgentMessage, AgentResult, BaseProvider


class CLIProvider(BaseProvider):
    """Runs a local CLI tool in a subprocess and captures its stdout as the completion."""

    #: Override in subclass — the executable name to check with shutil.which()
    _cli_binary: str = ""

    def is_available(self) -> bool:
        return bool(self._cli_binary and shutil.which(self._cli_binary))

    @abstractmethod
    def _build_command(self, prompt: str, system: str) -> list[str]:
        """Return the argv list to run."""
        ...

    def _build_prompt(self, messages: list[AgentMessage], system: str) -> str:
        """Flatten system + messages into a single string for CLI input."""
        parts: list[str] = []
        if system:
            parts.append(f"[SYSTEM]\n{system}\n")
        for m in messages:
            tag = "USER" if m.role == "user" else "ASSISTANT"
            parts.append(f"[{tag}]\n{m.content}")
        return "\n\n".join(parts)

    async def complete(
        self,
        messages: list[AgentMessage],
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> AgentResult:
        prompt = self._build_prompt(messages, system)
        cmd = self._build_command(prompt, system)

        t0 = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        elapsed = time.monotonic() - t0

        output = stdout.decode("utf-8", errors="replace").strip()

        # Some CLIs emit a preamble before the actual answer; strip known noise
        output = self._clean_output(output)

        if proc.returncode != 0 and not output:
            err = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"{self.name} CLI failed (exit {proc.returncode}): {err[:400]}")

        return AgentResult(
            content=output,
            provider=self.name,
            model=self._cli_binary,
            tokens_used=0,       # CLIs don't expose token counts
            cost_usd=0.0,        # no per-token billing through CLI auth
        )

    def _clean_output(self, text: str) -> str:
        """Strip CLI noise. Override in subclasses for provider-specific cleanup."""
        return text
