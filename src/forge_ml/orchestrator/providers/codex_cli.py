"""OpenAI Codex CLI provider — shells out to `codex --approval-mode full-auto`."""
from __future__ import annotations

import re

from .cli_base import CLIProvider


class CodexCLIProvider(CLIProvider):
    """
    Uses the locally installed `codex` CLI in full-auto mode.

    Flags used:
      --approval-mode full-auto   Approve all file/shell actions automatically.
      --quiet                     Suppress progress spinners and banners.
    """

    name = "codex-cli"
    _cli_binary = "codex"

    def _build_command(self, prompt: str, system: str) -> list[str]:
        return [
            "codex",
            "--approval-mode", "full-auto",
            "--quiet",
            prompt,
        ]

    def _clean_output(self, text: str) -> str:
        # Strip ANSI escape codes that codex sometimes emits even with --quiet
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text).strip()
