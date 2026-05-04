"""forge-ml CLI — voice-friendly commands."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from forge_ml.orchestrator import Orchestrator
from forge_ml.utils.config import config
from forge_ml.utils.logging import log

app = typer.Typer(
    name="forge",
    help="Multi-agent ML: research → data → build → optimize",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    task: str = typer.Argument(..., help="Natural-language ML task description"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save context JSON here"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Force a specific provider"),
):
    """Run the full forge-ml pipeline for a task."""
    if provider:
        import os
        os.environ["FORGE_DEFAULT_PROVIDER"] = provider

    async def _run():
        orchestrator = Orchestrator()
        ctx = await orchestrator.run(task)
        if output:
            output.write_text(json.dumps(ctx.__dict__, default=str, indent=2))
            log(f"Context saved to {output}", "success")
        return ctx

    asyncio.run(_run())


@app.command()
def providers():
    """List configured AI providers."""
    available = config.available_providers()
    if available:
        log(f"Available providers: {', '.join(available)}", "success")
    else:
        log("No providers configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.", "error")


@app.command()
def search(
    query: str = typer.Argument(..., help="Dataset search query"),
    limit: int = typer.Option(5, "--limit", "-n"),
):
    """Search for datasets on HuggingFace."""
    from forge_ml.data.sources import huggingface

    async def _search():
        results = await huggingface.search_datasets(query, limit=limit)
        for i, r in enumerate(results, 1):
            console.print(f"[cyan]{i}.[/cyan] [bold]{r['id']}[/bold]  downloads={r['downloads']}")

    asyncio.run(_search())


if __name__ == "__main__":
    app()
