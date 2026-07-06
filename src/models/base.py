"""Baseline estimators for play-type classification and yards-gained regression."""
from __future__ import annotations

from sklearn.dummy import DummyClassifier, DummyRegressor

from src.models.config import load_model_hyperparameters

__all__ = ["build_baseline_classifier", "build_baseline_regressor"]


def build_baseline_classifier() -> DummyClassifier:
    """Prior-probability classifier (sanity-check baseline for play type)."""
    params = load_model_hyperparameters("play_type", "baseline")
    return DummyClassifier(**params)


def build_baseline_regressor() -> DummyRegressor:
    """Mean predictor (sanity-check baseline for yards gained)."""
    params = load_model_hyperparameters("yards_gained", "baseline")
    return DummyRegressor(**params)
