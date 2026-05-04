"""Load a dataset into a pandas DataFrame."""
from __future__ import annotations

import pandas as pd


def load_to_dataframe(name: str, source: str, split: str = "train") -> pd.DataFrame:
    """Load dataset to DataFrame regardless of source."""
    if source == "huggingface":
        from forge_ml.data.sources.huggingface import load_dataset
        ds = load_dataset(name, split=split)
        return ds.to_pandas()
    elif source == "local":
        if name.endswith(".csv"):
            return pd.read_csv(name)
        elif name.endswith(".parquet"):
            return pd.read_parquet(name)
        else:
            raise ValueError(f"Unsupported local format: {name}")
    else:
        raise NotImplementedError(f"Loader for source '{source}' not yet implemented.")
