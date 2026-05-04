"""Generate and run ML model training pipelines."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score


CLASSIFIERS = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
}

REGRESSORS = {
    "random_forest": RandomForestRegressor,
    "ridge": Ridge,
}


@dataclass
class TrainResult:
    model: Any
    score: float
    metric_name: str
    model_type: str
    task_type: str


def auto_train(
    df: pd.DataFrame,
    target_column: str,
    task_type: str = "classification",
    model_key: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainResult:
    """Fit a baseline model. Returns TrainResult with score."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categoricals
    X = X.select_dtypes(include=["number"])  # keep numeric only for now

    if task_type == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        clf_cls = CLASSIFIERS.get(model_key, RandomForestClassifier)
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf_cls(random_state=random_state))])
        metric_name = "accuracy"
    else:
        reg_cls = REGRESSORS.get(model_key, RandomForestRegressor)
        pipeline = Pipeline([("scaler", StandardScaler()), ("reg", reg_cls(random_state=random_state))])
        metric_name = "r2"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    score = accuracy_score(y_test, preds) if task_type == "classification" else r2_score(y_test, preds)

    return TrainResult(
        model=pipeline,
        score=round(float(score), 4),
        metric_name=metric_name,
        model_type=model_key,
        task_type=task_type,
    )
