"""
Model builders for play-type classification and yards-gained regression.

Pipelines import from here; no CV logic lives in this package.

Reference: ``docs/modeling_plan.md`` Phase 3.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from sklearn.base import ClassifierMixin, RegressorMixin

from src.data.schema import MODEL_REGISTRY_KEYS, ModelingTask
from src.models.base import build_baseline_classifier, build_baseline_regressor
from src.models.random_forest import build_rf_classifier, build_rf_regressor
from src.models.xgboost import build_xgb_classifier, build_xgb_regressor
from src.utils.experiment_profile import get_active_profile_or_none

__all__ = [
    "CLASSIFIER_BUILDERS",
    "REGRESSOR_BUILDERS",
    "build_baseline_classifier",
    "build_baseline_regressor",
    "build_rf_classifier",
    "build_rf_regressor",
    "build_xgb_classifier",
    "build_xgb_regressor",
    "hyperparameters_from_experiment_config",
    "iter_classifier_builders",
    "iter_regressor_builders",
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


def _profile_model_keys(task: ModelingTask) -> tuple[str, ...] | None:
    profile = get_active_profile_or_none()
    if profile is None or not profile.has_task(task):
        return None
    return profile.task_model_keys(task)


def iter_classifier_builders(
    model_keys: tuple[str, ...] | None = None,
) -> Iterator[tuple[str, Callable[[], ClassifierMixin]]]:
    """Yield ``(model_key, builder)`` pairs for the active profile or *model_keys*."""
    keys = model_keys or _profile_model_keys("play_type") or MODEL_REGISTRY_KEYS
    for model_key in keys:
        yield model_key, CLASSIFIER_BUILDERS[model_key]


def iter_regressor_builders(
    model_keys: tuple[str, ...] | None = None,
) -> Iterator[tuple[str, Callable[[], RegressorMixin]]]:
    """Yield ``(model_key, builder)`` pairs for the active profile or *model_keys*."""
    keys = model_keys or _profile_model_keys("yards_gained") or MODEL_REGISTRY_KEYS
    for model_key in keys:
        yield model_key, REGRESSOR_BUILDERS[model_key]


def hyperparameters_from_experiment_config(
    experiment_config: dict[str, Any],
    task: ModelingTask,
    model_key: str,
) -> dict[str, Any]:
    """Read saved hyperparameters from an experiment snapshot."""
    return dict(
        experiment_config["tasks"][task]["models"][model_key]["hyperparameters"]
    )
