"""Random Forest estimators for both modeling tasks."""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.models.config import load_model_hyperparameters

__all__ = ["build_rf_classifier", "build_rf_regressor"]


def build_rf_classifier(
    *,
    hyperparameters: dict[str, Any] | None = None,
) -> RandomForestClassifier:
    params = hyperparameters or load_model_hyperparameters("play_type", "random_forest")
    return RandomForestClassifier(**params)


def build_rf_regressor(
    *,
    hyperparameters: dict[str, Any] | None = None,
) -> RandomForestRegressor:
    params = hyperparameters or load_model_hyperparameters("yards_gained", "random_forest")
    return RandomForestRegressor(**params)
