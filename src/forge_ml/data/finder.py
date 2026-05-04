"""Agentic dataset finder: asks an LLM to pick the best dataset, then validates it exists."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from forge_ml.data.sources import huggingface, kaggle
from forge_ml.utils.logging import log


@dataclass
class DatasetCandidate:
    name: str
    source: str
    score: float
    meta: dict


async def find_datasets(task: str, top_k: int = 3) -> list[DatasetCandidate]:
    """Search HuggingFace (and Kaggle if configured) for relevant datasets."""
    log(f"Searching datasets for: {task}", "info")

    hf_task, kag_task = await asyncio.gather(
        huggingface.search_datasets(task, limit=top_k),
        asyncio.to_thread(kaggle.search_datasets, task, top_k),
    )

    candidates = []
    for d in hf_task:
        candidates.append(DatasetCandidate(
            name=d["id"],
            source="huggingface",
            score=float(d.get("downloads", 0)),
            meta=d,
        ))
    for d in kag_task:
        candidates.append(DatasetCandidate(
            name=d.get("ref", d.get("title", "")),
            source="kaggle",
            score=float(d.get("downloads", 0)),
            meta=d,
        ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    log(f"Found {len(candidates)} candidates", "success")
    return candidates[:top_k]
