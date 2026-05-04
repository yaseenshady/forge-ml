"""Agent-driven ML research: finds SOTA approaches for a given task."""
from __future__ import annotations

from dataclasses import dataclass

from forge_ml.orchestrator.providers.base import AgentMessage
from forge_ml.orchestrator.providers.claude import ClaudeProvider
from forge_ml.orchestrator.providers.openai import OpenAIProvider
from forge_ml.utils.config import config
from forge_ml.utils.logging import agent_log


RESEARCH_SYSTEM = """You are an ML research specialist.
For the given task:
1. Identify the task type (classification, regression, generation, clustering, etc.)
2. List 3-5 SOTA model architectures or algorithms with brief rationale
3. Identify key challenges and evaluation metrics
4. Recommend one dataset to start with

Be concise. Output structured plain text."""


@dataclass
class ResearchReport:
    task: str
    task_type: str
    approaches: list[str]
    recommended_dataset: str
    evaluation_metrics: list[str]
    raw_text: str


async def research_task(task: str) -> ResearchReport:
    """Run multi-provider research: Claude for reasoning, OpenAI as fallback."""
    providers = []
    if config.anthropic_api_key:
        providers.append(ClaudeProvider())
    if config.openai_api_key:
        providers.append(OpenAIProvider())
    if not providers:
        raise RuntimeError("No providers configured.")

    provider = providers[0]
    agent_log(provider.name, f"researching: {task[:60]}")

    messages = [AgentMessage(role="user", content=f"ML Task: {task}")]
    result = await provider.complete(messages, system=RESEARCH_SYSTEM, temperature=0.2)

    # Lightweight parse — let the LLM structure the text
    text = result.content
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    return ResearchReport(
        task=task,
        task_type="unknown",  # orchestrator extracts this
        approaches=lines[:5],
        recommended_dataset="",
        evaluation_metrics=[],
        raw_text=text,
    )
