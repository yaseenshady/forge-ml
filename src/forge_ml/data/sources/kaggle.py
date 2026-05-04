"""Kaggle dataset search (requires KAGGLE_USERNAME + KAGGLE_KEY env vars)."""
from __future__ import annotations

from forge_ml.utils.config import config


def is_available() -> bool:
    return bool(config.kaggle_username and config.kaggle_key)


def search_datasets(query: str, limit: int = 5) -> list[dict]:
    if not is_available():
        return []
    try:
        import kaggle  # type: ignore
        kaggle.api.authenticate()
        results = kaggle.api.dataset_list(search=query, page=1)
        return [
            {
                "ref": d.ref,
                "title": d.title,
                "size": d.size,
                "downloads": d.downloadCount,
                "source": "kaggle",
            }
            for d in results[:limit]
        ]
    except Exception:
        return []
