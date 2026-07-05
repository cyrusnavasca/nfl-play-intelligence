"""
Model builders for play-type classification and yards-gained regression.

Pipelines import from here; no CV logic lives in this package.

Reference: ``docs/modeling_plan.md`` Phase 3.
"""
from __future__ import annotations

from collections.abc import Callable

from sklearn.base import ClassifierMixin, RegressorMixin

from src.data.schema import MODEL_REGISTRY_KEYS
from src.models.base import build_baseline_classifier, build_baseline_regressor
from src.models.random_forest import build_rf_classifier, build_rf_regressor
from src.models.xgboost import build_xgb_classifier, build_xgb_regressor

__all__ = [
    "CLASSIFIER_BUILDERS",
    "REGRESSOR_BUILDERS",
    "build_baseline_classifier",
    "build_baseline_regressor",
    "build_rf_classifier",
    "build_rf_regressor",
    "build_xgb_classifier",
    "build_xgb_regressor",
]

CLASSIFIER_BUILDERS: dict[str, Callable[[], ClassifierMixin]] = {
    "baseline": build_baseline_classifier,
    "random_forest": build_rf_classifier,
    "xgboost": build_xgb_classifier,
}

REGRESSOR_BUILDERS: dict[str, Callable[[], RegressorMixin]] = {
    "baseline": build_baseline_regressor,
    "random_forest": build_rf_regressor,
    "xgboost": build_xgb_regressor,
}

if tuple(CLASSIFIER_BUILDERS) != MODEL_REGISTRY_KEYS:
    raise RuntimeError(
        "CLASSIFIER_BUILDERS keys must match MODEL_REGISTRY_KEYS in data/schema.py"
    )
if tuple(REGRESSOR_BUILDERS) != MODEL_REGISTRY_KEYS:
    raise RuntimeError(
        "REGRESSOR_BUILDERS keys must match MODEL_REGISTRY_KEYS in data/schema.py"
    )
