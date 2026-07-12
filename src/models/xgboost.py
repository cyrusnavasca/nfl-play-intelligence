"""XGBoost estimator for play-type classification."""
from __future__ import annotations

from typing import Any

from xgboost import XGBClassifier

from src.models.config import load_model_hyperparameters

__all__ = ["build_xgb_classifier"]


def build_xgb_classifier(
    *,
    hyperparameters: dict[str, Any] | None = None,
) -> XGBClassifier:
    params = hyperparameters or load_model_hyperparameters("xgboost")
    return XGBClassifier(**params)
