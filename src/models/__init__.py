"""
Model builders for play-type classification.

Pipelines import from here; no CV logic lives in this package.

Reference: ``docs/modeling_plan.md`` Phase 3.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from sklearn.base import ClassifierMixin

from src.data.schema import MODEL_REGISTRY_KEYS
from src.models.base import build_baseline_classifier
from src.models.random_forest import build_rf_classifier
from src.models.xgboost import build_xgb_classifier
from src.utils.experiment_profile import get_active_profile_or_none

__all__ = [
    "CLASSIFIER_BUILDERS",
    "build_baseline_classifier",
    "build_rf_classifier",
    "build_xgb_classifier",
    "hyperparameters_from_experiment_config",
    "iter_classifier_builders",
]

CLASSIFIER_BUILDERS: dict[str, Callable[[], ClassifierMixin]] = {
    "baseline": build_baseline_classifier,
    "random_forest": build_rf_classifier,
    "xgboost": build_xgb_classifier,
}

if tuple(CLASSIFIER_BUILDERS) != MODEL_REGISTRY_KEYS:
    raise RuntimeError(
        "CLASSIFIER_BUILDERS keys must match MODEL_REGISTRY_KEYS in data/schema.py"
    )


def _profile_model_keys() -> tuple[str, ...] | None:
    profile = get_active_profile_or_none()
    if profile is None:
        return None
    return profile.model_keys()


def iter_classifier_builders(
    model_keys: tuple[str, ...] | None = None,
) -> Iterator[tuple[str, Callable[[], ClassifierMixin]]]:
    """Yield ``(model_key, builder)`` pairs for the active profile or *model_keys*."""
    keys = model_keys or _profile_model_keys() or MODEL_REGISTRY_KEYS
    for model_key in keys:
        yield model_key, CLASSIFIER_BUILDERS[model_key]


def hyperparameters_from_experiment_config(
    experiment_config: dict[str, Any],
    model_key: str,
) -> dict[str, Any]:
    """Read saved hyperparameters from an experiment snapshot."""
    return dict(
        experiment_config["models"][model_key]["hyperparameters"]
    )
