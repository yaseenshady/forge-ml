"""Basic smoke tests."""
import os
import pytest
from forge_ml.utils.config import ForgeConfig


def test_default_provider():
    cfg = ForgeConfig()
    assert cfg.default_provider in ("claude", "openai", "gemini")


def test_available_providers_empty():
    cfg = ForgeConfig(anthropic_api_key="", openai_api_key="", gemini_api_key="")
    # Should return fallback list
    result = cfg.available_providers()
    assert isinstance(result, list)


def test_context_cost_tracking():
    from forge_ml.orchestrator.context import TaskContext
    ctx = TaskContext(task_description="test")
    ctx.add_cost(0.05, "claude")
    ctx.add_cost(0.03, "openai")
    assert abs(ctx.total_cost_usd - 0.08) < 1e-9
    assert "claude" in ctx.providers_used
    assert "openai" in ctx.providers_used
