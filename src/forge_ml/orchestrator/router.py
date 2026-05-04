"""Multi-agent orchestrator: routes subtasks to providers and runs the full pipeline."""
from __future__ import annotations

import asyncio
from typing import Callable, Coroutine, Any

from forge_ml.utils.config import config
from forge_ml.utils.logging import agent_log, log
from .context import TaskContext
from .providers.base import AgentMessage, BaseProvider
from .providers.claude import ClaudeProvider
from .providers.openai import OpenAIProvider


def _build_providers() -> dict[str, BaseProvider]:
    candidates: dict[str, BaseProvider] = {
        "claude": ClaudeProvider(),
        "openai": OpenAIProvider(),
    }
    return {name: p for name, p in candidates.items() if p.is_available()}


class Orchestrator:
    """
    Routes ML pipeline stages across available AI providers.

    Stages:
      1. research  — understand the task, find SOTA approaches
      2. data      — find and validate a dataset
      3. build     — generate model training code
      4. optimize  — suggest hyperparameter search space
      5. summarize — produce final report
    """

    STAGE_SYSTEM = {
        "research": (
            "You are an ML research agent. Given a task description, "
            "identify the best-known approaches, relevant datasets, and model architectures. "
            "Be concise and actionable."
        ),
        "data": (
            "You are an ML data agent. Given a task and research summary, "
            "recommend the single best publicly available dataset (HuggingFace, Kaggle, or UCI). "
            "Return JSON with keys: name, source, target_column, task_type, why."
        ),
        "build": (
            "You are an ML engineer agent. Write clean, runnable Python that trains a model "
            "on the given dataset. Use scikit-learn for tabular tasks, HuggingFace transformers "
            "for text/image. Return only executable code, no prose."
        ),
        "optimize": (
            "You are an ML optimization agent. Given model code and eval results, "
            "propose an Optuna hyperparameter search space as Python code. "
            "Return only the objective function."
        ),
        "summarize": (
            "You are an ML reporting agent. Summarize the pipeline run: task, dataset chosen, "
            "model, best metrics, cost. 3-5 bullet points, plain English."
        ),
    }

    def __init__(self) -> None:
        self.providers = _build_providers()
        if not self.providers:
            raise RuntimeError("No AI providers configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")

    def _pick_provider(self, stage: str) -> BaseProvider:
        """Round-robin provider selection; prefer default, fall back to any available."""
        preferred = config.default_provider
        if preferred in self.providers:
            return self.providers[preferred]
        return next(iter(self.providers.values()))

    async def _call(self, stage: str, user_prompt: str, ctx: TaskContext) -> str:
        provider = self._pick_provider(stage)
        system = self.STAGE_SYSTEM[stage]
        messages = [AgentMessage(role="user", content=user_prompt)]
        agent_log(provider.name, f"[{stage}] starting...")
        result = await provider.complete(messages, system=system)
        ctx.add_cost(result.cost_usd, result.provider)
        ctx.add_log(f"{stage}({result.provider}): {len(result.content)} chars, ${result.cost_usd:.4f}")
        agent_log(provider.name, f"[{stage}] done. ${result.cost_usd:.4f}")
        return result.content

    async def run(self, task: str) -> TaskContext:
        ctx = TaskContext(task_description=task)
        log(f"Forge starting: {task}", "success")

        # Stage 1: Research
        research = await self._call("research", f"Task: {task}", ctx)
        ctx.research_summary = research

        # Stage 2: Data (can run concurrently with nothing else yet, but isolated here)
        data_prompt = f"Task: {task}\n\nResearch summary:\n{research}"
        data_resp = await self._call("data", data_prompt, ctx)
        ctx.add_log(f"data_response: {data_resp[:200]}")
        # Try to parse JSON
        import json, re
        m = re.search(r"\{.*\}", data_resp, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group())
                ctx.dataset_name = d.get("name", "")
                ctx.dataset_source = d.get("source", "")
                ctx.target_column = d.get("target_column", "")
                ctx.task_type = d.get("task_type", "")
            except json.JSONDecodeError:
                pass

        # Stage 3: Build
        build_prompt = (
            f"Task: {task}\n"
            f"Dataset: {ctx.dataset_name} ({ctx.dataset_source})\n"
            f"Target column: {ctx.target_column}\n"
            f"Task type: {ctx.task_type}\n\n"
            "Write the training script."
        )
        build_resp = await self._call("build", build_prompt, ctx)
        ctx.model_type = "generated"
        ctx.add_log(f"build_code: {len(build_resp)} chars")

        # Stage 4: Optimize
        opt_prompt = f"Model code:\n{build_resp[:3000]}\n\nPropose an Optuna objective function."
        opt_resp = await self._call("optimize", opt_prompt, ctx)
        ctx.add_log(f"optuna_code: {len(opt_resp)} chars")

        # Stage 5: Summarize
        summary_prompt = (
            f"Task: {task}\n"
            f"Dataset: {ctx.dataset_name} from {ctx.dataset_source}\n"
            f"Task type: {ctx.task_type}\n"
            f"Total cost: ${ctx.total_cost_usd:.4f}\n"
            f"Providers used: {', '.join(ctx.providers_used)}\n"
            "Summarize the run."
        )
        summary = await self._call("summarize", summary_prompt, ctx)
        ctx.add_log("pipeline complete")

        log("\n--- FORGE SUMMARY ---", "success")
        log(summary, "info")
        log(f"\nTotal cost: ${ctx.total_cost_usd:.4f} | Providers: {', '.join(ctx.providers_used)}", "success")

        return ctx
