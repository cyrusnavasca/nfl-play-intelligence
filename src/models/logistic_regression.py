"""Logistic Regression estimator for play-type classification."""
from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.config import load_model_hyperparameters

__all__ = ["build_lr_classifier"]


def build_lr_classifier(
    *,
    hyperparameters: dict[str, Any] | None = None,
) -> Pipeline:
    """
    Standard-scaled logistic regression.

    Wrapped in a Pipeline so features (which vary widely in scale, e.g.
    ``game_seconds_remaining`` vs epa rates) are standardized before the linear
    fit. Hyperparameters are passed to ``LogisticRegression``.
    """
    params = hyperparameters or load_model_hyperparameters("logistic_regression")
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**params)),
        ]
    )
