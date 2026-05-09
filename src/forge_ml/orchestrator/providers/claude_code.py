"""Claude Code CLI provider — shells out to `claude -p --dangerously-skip-permissions`."""
from __future__ import annotations

from .cli_base import CLIProvider


class ClaudeCodeCLIProvider(CLIProvider):
    """
    Uses the locally installed `claude` CLI (Claude Code) in print mode.

    Flags used:
      -p / --print                    Non-interactive, print output and exit.
      --dangerously-skip-permissions  Skip all permission prompts (full-auto).
    """

    name = "claude-code-cli"
    _cli_binary = "claude"

    def _build_command(self, prompt: str, system: str) -> list[str]:
        return [
            "claude",
            "--print",
            "--dangerously-skip-permissions",
            prompt,
        ]
