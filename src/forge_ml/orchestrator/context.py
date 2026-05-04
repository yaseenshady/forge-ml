"""Shared task context passed between agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskContext:
    """Mutable shared state for one forge-ml run."""
    task_description: str = ""

    # Research phase
    research_summary: str = ""
    suggested_approaches: list[str] = field(default_factory=list)

    # Data phase
    dataset_name: str = ""
    dataset_source: str = ""  # "huggingface" | "kaggle" | "uci" | "local"
    dataset_path: str = ""
    feature_columns: list[str] = field(default_factory=list)
    target_column: str = ""
    task_type: str = ""  # "classification" | "regression" | "generation" | "clustering"

    # Model phase
    model_type: str = ""
    model_path: str = ""
    eval_metrics: dict[str, float] = field(default_factory=dict)

    # Optimization phase
    best_params: dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0

    # Meta
    providers_used: list[str] = field(default_factory=list)
    total_cost_usd: float = 0.0
    log: list[str] = field(default_factory=list)

    def add_log(self, msg: str) -> None:
        self.log.append(msg)

    def add_cost(self, usd: float, provider: str) -> None:
        self.total_cost_usd += usd
        if provider not in self.providers_used:
            self.providers_used.append(provider)
