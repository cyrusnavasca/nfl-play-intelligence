"""Baseline estimator for play-type classification."""
from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier

from src.models.config import load_model_hyperparameters

__all__ = ["build_baseline_classifier"]


def build_baseline_classifier(
    *,
    hyperparameters: dict[str, Any] | None = None,
) -> DummyClassifier:
    """Prior-probability classifier (sanity-check baseline for play type)."""
    params = hyperparameters or load_model_hyperparameters("baseline")
    return DummyClassifier(**params)
