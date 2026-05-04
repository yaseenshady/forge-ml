"""HuggingFace Hub dataset search and download."""
from __future__ import annotations

import httpx


HF_API = "https://huggingface.co/api/datasets"


async def search_datasets(query: str, limit: int = 5) -> list[dict]:
    """Search HuggingFace Hub for datasets matching query."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(HF_API, params={"search": query, "limit": limit, "full": "false"})
        resp.raise_for_status()
        results = resp.json()
    return [
        {
            "id": d.get("id", ""),
            "downloads": d.get("downloads", 0),
            "likes": d.get("likes", 0),
            "tags": d.get("tags", []),
            "source": "huggingface",
        }
        for d in results
    ]


def load_dataset(name: str, split: str = "train"):
    """Load a HuggingFace dataset. Returns a datasets.Dataset."""
    from datasets import load_dataset as hf_load
    return hf_load(name, split=split, trust_remote_code=False)
