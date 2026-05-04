"""Summarize a completed forge-ml pipeline run."""
from __future__ import annotations

from forge_ml.orchestrator.context import TaskContext


def summarize_run(ctx: TaskContext) -> str:
    lines = [
        f"Task: {ctx.task_description}",
        f"Dataset: {ctx.dataset_name} ({ctx.dataset_source})",
        f"Task type: {ctx.task_type}",
        f"Best score: {ctx.best_score}" if ctx.best_score else "Score: not yet computed",
        f"Providers used: {', '.join(ctx.providers_used)}",
        f"Total cost: ${ctx.total_cost_usd:.4f}",
    ]
    return "\n".join(lines)
