"""GitHub Copilot CLI provider — shells out to `gh copilot suggest`."""
from __future__ import annotations

import re

from .cli_base import CLIProvider


class CopilotCLIProvider(CLIProvider):
    """
    Uses the locally installed `gh copilot` CLI.

    gh copilot suggest -t shell  — best for shell/infra tasks
    gh copilot explain           — best for explaining code

    For general ML pipeline prompts we use `suggest -t shell` as the closest
    general-purpose mode available in the Copilot CLI.
    """

    name = "copilot-cli"
    _cli_binary = "gh"

    def is_available(self) -> bool:
        import shutil
        # Require both gh and the copilot extension
        if not shutil.which("gh"):
            return False
        import subprocess
        try:
            r = subprocess.run(
                ["gh", "copilot", "--version"],
                capture_output=True, timeout=5,
            )
            return r.returncode == 0
        except Exception:
            return False

    def _build_command(self, prompt: str, system: str) -> list[str]:
        # gh copilot suggest reads the target from stdin; pass prompt as the request
        return [
            "gh", "copilot", "suggest",
            "-t", "shell",
            "--", prompt[:2000],   # CLI has an input length limit
        ]

    def _clean_output(self, text: str) -> str:
        # Strip ANSI and the "Suggestion:" prefix gh copilot emits
        ansi = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        text = ansi.sub("", text).strip()
        # Remove leading "Suggestion:" or "Command:" labels
        text = re.sub(r"^(Suggestion|Command):\s*", "", text, flags=re.IGNORECASE)
        return text
