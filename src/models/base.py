"""Baseline estimators for play-type classification and yards-gained regression."""
from __future__ import annotations

from sklearn.dummy import DummyClassifier, DummyRegressor

__all__ = ["build_baseline_classifier", "build_baseline_regressor"]


def build_baseline_classifier() -> DummyClassifier:
    """Prior-probability classifier (sanity-check baseline for play type)."""
    return DummyClassifier(strategy="prior")


def build_baseline_regressor() -> DummyRegressor:
    """Mean predictor (sanity-check baseline for yards gained)."""
    return DummyRegressor(strategy="mean")
