"""Global configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ForgeConfig:
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    kaggle_username: str = field(default_factory=lambda: os.getenv("KAGGLE_USERNAME", ""))
    kaggle_key: str = field(default_factory=lambda: os.getenv("KAGGLE_KEY", ""))
    default_provider: str = field(
        default_factory=lambda: os.getenv("FORGE_DEFAULT_PROVIDER", "claude")
    )
    max_parallel_agents: int = field(
        default_factory=lambda: int(os.getenv("FORGE_MAX_AGENTS", "4"))
    )
    workspace_dir: Path = field(
        default_factory=lambda: Path(os.getenv("FORGE_WORKSPACE", "~/.forge-ml")).expanduser()
    )

    def available_providers(self) -> list[str]:
        providers = []
        if self.anthropic_api_key:
            providers.append("claude")
        if self.openai_api_key:
            providers.append("openai")
        if self.gemini_api_key:
            providers.append("gemini")
        return providers or ["claude"]


config = ForgeConfig()
