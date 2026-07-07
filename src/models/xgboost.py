"""XGBoost estimators for both modeling tasks."""
from __future__ import annotations

from typing import Any

from xgboost import XGBClassifier, XGBRegressor

from src.models.config import load_model_hyperparameters

__all__ = ["build_xgb_classifier", "build_xgb_regressor"]


def build_xgb_classifier(
    *,
    hyperparameters: dict[str, Any] | None = None,
) -> XGBClassifier:
    params = hyperparameters or load_model_hyperparameters("play_type", "xgboost")
    return XGBClassifier(**params)


def build_xgb_regressor(
    *,
    hyperparameters: dict[str, Any] | None = None,
) -> XGBRegressor:
    params = hyperparameters or load_model_hyperparameters("yards_gained", "xgboost")
    return XGBRegressor(**params)
