"""Multi-agent orchestrator: routes subtasks to providers and runs the full pipeline."""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from forge_ml.utils.config import config
from forge_ml.utils.logging import agent_log, log
from .context import TaskContext
from .providers.base import AgentMessage, BaseProvider
from .providers.claude import ClaudeProvider
from .providers.openai import OpenAIProvider
from .providers.claude_code import ClaudeCodeCLIProvider
from .providers.codex_cli import CodexCLIProvider
from .providers.copilot_cli import CopilotCLIProvider


def _build_providers() -> dict[str, BaseProvider]:
    """
    Discover all available providers — CLI tools take priority over API keys
    because they require no extra credentials and run with full-auto permissions.

    Priority order per name slot:
      claude   → claude-code CLI  > Anthropic API
      codex    → codex CLI        (no API fallback)
      copilot  → gh copilot CLI   (no API fallback)
      openai   → OpenAI API       (fallback when codex CLI absent)
    """
    candidates: dict[str, BaseProvider] = {}

    # CLI providers (preferred — no API key needed)
    claude_cli = ClaudeCodeCLIProvider()
    codex_cli = CodexCLIProvider()
    copilot_cli = CopilotCLIProvider()

    if claude_cli.is_available():
        candidates["claude"] = claude_cli
    elif ClaudeProvider().is_available():
        candidates["claude"] = ClaudeProvider()

    if codex_cli.is_available():
        candidates["codex"] = codex_cli

    if copilot_cli.is_available():
        candidates["copilot"] = copilot_cli

    # OpenAI API as fallback for code-gen stages if codex CLI is absent
    openai_api = OpenAIProvider()
    if openai_api.is_available():
        candidates["openai"] = openai_api

    return candidates


class Orchestrator:
    """
    Routes ML pipeline stages across available AI providers.

    Stages:
      1. research  — understand the task, find SOTA approaches
      2. data      — find and validate a dataset
      3. build     — generate model training code
      4. execute   — actually run the code in a subprocess, capture real metrics
      5. optimize  — Optuna hyperparameter search on the real dataset
      6. summarize — produce final report with real numbers
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
            "for text/image. "
            "IMPORTANT: the last line of stdout MUST be a JSON dict of metrics, e.g.: "
            '{"accuracy": 0.93, "f1_macro": 0.91}  '
            "Return only executable code, no prose, no markdown fences."
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
        """
        Route each stage to the best available provider.

        Preference order per stage:
          research / data / summarize  → claude-code CLI > claude API > any
          build / optimize             → codex CLI > copilot CLI > openai API > claude > any
        """
        stage_prefs: dict[str, list[str]] = {
            "research":  ["claude", "copilot", "codex", "openai"],
            "data":      ["claude", "copilot", "codex", "openai"],
            "summarize": ["claude", "copilot", "codex", "openai"],
            "build":     ["codex", "copilot", "openai", "claude"],
            "optimize":  ["codex", "copilot", "openai", "claude"],
        }
        for name in stage_prefs.get(stage, ["claude", "openai"]):
            if name in self.providers:
                return self.providers[name]
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

    # ------------------------------------------------------------------
    # Execution helpers (run in thread so they don't block the event loop)
    # ------------------------------------------------------------------

    def _try_load_dataset(self, ctx: TaskContext):
        """Load the chosen dataset into a DataFrame. Returns None on failure."""
        if not ctx.dataset_name or ctx.dataset_source not in ("huggingface",):
            return None
        try:
            from forge_ml.data.loader import load_to_dataframe
            log(f"Loading dataset: {ctx.dataset_name} ({ctx.dataset_source})…", "info")
            df = load_to_dataframe(ctx.dataset_name, ctx.dataset_source)
            log(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} cols", "success")
            return df
        except Exception as exc:
            log(f"Dataset load failed ({exc}); skipping real execution.", "error")
            return None

    def _try_auto_train(self, df, ctx: TaskContext) -> dict[str, float]:
        """Run auto_train on the DataFrame. Returns metrics dict or {} on failure."""
        if df is None or not ctx.target_column:
            return {}
        try:
            from forge_ml.model.builder import auto_train
            from forge_ml.model.evaluator import evaluate_classification, evaluate_regression
            log("Running baseline auto_train…", "info")
            result = auto_train(df, ctx.target_column, task_type=ctx.task_type or "classification")
            ctx.model_type = result.model_type
            metrics = {result.metric_name: result.score}
            ctx.eval_metrics = metrics
            log(f"Baseline: {result.metric_name}={result.score:.4f}", "success")
            return metrics
        except Exception as exc:
            log(f"auto_train failed ({exc}); continuing without baseline metrics.", "error")
            return {}

    def _try_optimize(self, df, ctx: TaskContext) -> dict[str, Any]:
        """Run Optuna optimizer. Returns best_params dict or {} on failure."""
        if df is None or not ctx.target_column:
            return {}
        try:
            from forge_ml.model.optimizer import optimize
            log("Running Optuna optimization (30 trials)…", "info")
            result = optimize(df, ctx.target_column, task_type=ctx.task_type or "classification", n_trials=30)
            ctx.best_params = result["best_params"]
            ctx.best_score = result["best_score"]
            log(f"Best score: {result['best_score']:.4f} | params: {result['best_params']}", "success")
            return result
        except Exception as exc:
            log(f"Optuna failed ({exc}); skipping optimization.", "error")
            return {}

    def _try_execute_script(self, code: str, ctx: TaskContext) -> dict[str, float]:
        """Run LLM-generated script in a subprocess. Returns parsed metrics or {}."""
        from forge_ml.model.executor import run_script
        log("Executing generated training script…", "info")
        exec_result = run_script(code, timeout=120)
        ctx.generated_script = code
        ctx.execution_stdout = exec_result.stdout[-2000:] if exec_result.stdout else ""
        ctx.execution_stderr = exec_result.stderr[-1000:] if exec_result.stderr else ""
        ctx.script_success = exec_result.success
        if exec_result.metrics:
            ctx.eval_metrics = exec_result.metrics
        status = exec_result.summary()
        log(f"Script execution: {status}", "success" if exec_result.success else "error")
        return exec_result.metrics

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    async def run(self, task: str) -> TaskContext:
        ctx = TaskContext(task_description=task)
        log(f"Forge starting: {task}", "success")

        # ── Stage 1: Research ──────────────────────────────────────────
        research = await self._call("research", f"Task: {task}", ctx)
        ctx.research_summary = research

        # ── Stage 2: Data ──────────────────────────────────────────────
        data_prompt = f"Task: {task}\n\nResearch summary:\n{research}"
        data_resp = await self._call("data", data_prompt, ctx)
        ctx.add_log(f"data_response: {data_resp[:200]}")

        m = re.search(r"\{.*?\}", data_resp, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group())
                ctx.dataset_name = d.get("name", "")
                ctx.dataset_source = d.get("source", "huggingface").lower()
                ctx.target_column = d.get("target_column", "")
                ctx.task_type = d.get("task_type", "classification")
            except json.JSONDecodeError:
                pass

        # ── Stage 3: Build (LLM generates script) ─────────────────────
        build_prompt = (
            f"Task: {task}\n"
            f"Dataset: {ctx.dataset_name} ({ctx.dataset_source})\n"
            f"Target column: {ctx.target_column}\n"
            f"Task type: {ctx.task_type}\n\n"
            "Write the training script. Remember: last stdout line must be a JSON metrics dict."
        )
        build_resp = await self._call("build", build_prompt, ctx)

        # Strip markdown fences if the LLM wrapped the code
        clean_code = re.sub(r"^```(?:python)?\n?", "", build_resp.strip(), flags=re.MULTILINE)
        clean_code = re.sub(r"\n?```$", "", clean_code.strip(), flags=re.MULTILINE)
        ctx.add_log(f"build_code: {len(clean_code)} chars")

        # ── Stage 4a: Execute generated script in subprocess ──────────
        script_metrics = await asyncio.to_thread(self._try_execute_script, clean_code, ctx)

        # ── Stage 4b: Baseline via auto_train (always reliable) ───────
        df = await asyncio.to_thread(self._try_load_dataset, ctx)
        baseline_metrics = await asyncio.to_thread(self._try_auto_train, df, ctx)

        # Merge: prefer script metrics, fall back to baseline
        combined_metrics = {**baseline_metrics, **script_metrics}
        if combined_metrics:
            ctx.eval_metrics = combined_metrics

        # ── Stage 4c: Optuna optimization on real data ─────────────────
        opt_result = await asyncio.to_thread(self._try_optimize, df, ctx)

        # ── Stage 5: LLM Optimize prompt (code suggestions) ───────────
        metrics_str = json.dumps(combined_metrics or {"note": "no metrics captured"}, indent=2)
        opt_prompt = (
            f"Model code:\n{clean_code[:2000]}\n\n"
            f"Actual eval metrics:\n{metrics_str}\n\n"
            f"Optuna best params: {json.dumps(ctx.best_params)}\n"
            f"Optuna best score: {ctx.best_score:.4f}\n\n"
            "Propose further improvements or an Optuna objective function."
        )
        opt_resp = await self._call("optimize", opt_prompt, ctx)
        ctx.add_log(f"optuna_code: {len(opt_resp)} chars")

        # ── Stage 6: Summarize with real numbers ──────────────────────
        summary_prompt = (
            f"Task: {task}\n"
            f"Dataset: {ctx.dataset_name} from {ctx.dataset_source}\n"
            f"Task type: {ctx.task_type}\n"
            f"Eval metrics: {json.dumps(ctx.eval_metrics)}\n"
            f"Optuna best score: {ctx.best_score:.4f}\n"
            f"Optuna best params: {json.dumps(ctx.best_params)}\n"
            f"Script executed successfully: {getattr(ctx, 'script_success', False)}\n"
            f"Total cost: ${ctx.total_cost_usd:.4f}\n"
            f"Providers used: {', '.join(ctx.providers_used)}\n"
            "Summarize the run in 3-5 plain-English bullet points."
        )
        summary = await self._call("summarize", summary_prompt, ctx)
        ctx.add_log("pipeline complete")

        log("\n--- FORGE SUMMARY ---", "success")
        log(summary, "info")
        log(f"\nTotal cost: ${ctx.total_cost_usd:.4f} | Providers: {', '.join(ctx.providers_used)}", "success")
        if ctx.eval_metrics:
            log(f"Metrics: {ctx.eval_metrics}", "success")

        return ctx
