"""Random Forest estimator for play-type classification."""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier

from src.models.config import load_model_hyperparameters

__all__ = ["build_rf_classifier"]


def build_rf_classifier(
    *,
    hyperparameters: dict[str, Any] | None = None,
) -> RandomForestClassifier:
    params = hyperparameters or load_model_hyperparameters("random_forest")
    return RandomForestClassifier(**params)
