"""Random Forest estimators for both modeling tasks."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.data.schema import SEED

__all__ = ["build_rf_classifier", "build_rf_regressor"]


def build_rf_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


def build_rf_regressor() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=SEED,
        n_jobs=-1,
    )
