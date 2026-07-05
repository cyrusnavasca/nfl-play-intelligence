"""XGBoost estimators for both modeling tasks."""
from __future__ import annotations

from xgboost import XGBClassifier, XGBRegressor

from src.data.schema import SEED

__all__ = ["build_xgb_classifier", "build_xgb_regressor"]


def build_xgb_classifier() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED,
        verbosity=0,
    )


def build_xgb_regressor() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        verbosity=0,
    )
