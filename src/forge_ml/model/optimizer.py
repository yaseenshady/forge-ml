"""Optuna-based hyperparameter optimizer."""
from __future__ import annotations

from typing import Any, Callable

import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize(
    df: pd.DataFrame,
    target_column: str,
    task_type: str = "classification",
    n_trials: int = 30,
    cv: int = 3,
) -> dict[str, Any]:
    """Run Optuna study and return best hyperparams."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    X = df.drop(columns=[target_column]).select_dtypes(include=["number"])
    y = df[target_column]

    if task_type == "classification":
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(y.astype(str))
        scoring = "accuracy"
        ModelCls = RandomForestClassifier
    else:
        scoring = "r2"
        ModelCls = RandomForestRegressor

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
        model = ModelCls(**params, random_state=42, n_jobs=-1)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return {
        "best_params": study.best_params,
        "best_score": round(study.best_value, 4),
        "n_trials": n_trials,
    }
