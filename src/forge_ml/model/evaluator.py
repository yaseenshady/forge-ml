"""Standard model evaluation metrics."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    r2_score, mean_squared_error, mean_absolute_error,
)


def evaluate_classification(y_true, y_pred, y_prob=None) -> dict[str, float]:
    result: dict[str, float] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }
    if y_prob is not None:
        try:
            result["roc_auc"] = round(float(roc_auc_score(y_true, y_prob, multi_class="ovr")), 4)
        except Exception:
            pass
    return result


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "rmse": round(float(mean_squared_error(y_true, y_pred) ** 0.5), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
    }
