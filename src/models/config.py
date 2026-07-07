"""Hyperparameter access from the active experiment profile."""
from __future__ import annotations

from typing import Any

from src.data.schema import MODEL_REGISTRY_KEYS, ModelingTask
from src.utils.experiment_profile import get_active_profile

__all__ = [
    "load_model_hyperparameters",
    "snapshot_task_hyperparameters",
]


def load_model_hyperparameters(task: ModelingTask, model_key: str) -> dict[str, Any]:
    """Return hyperparameters for *model_key* on *task* from the active profile."""
    if model_key not in MODEL_REGISTRY_KEYS:
        raise KeyError(f"unknown model key: {model_key!r}")
    return get_active_profile().model_hyperparameters(task, model_key)


def snapshot_task_hyperparameters(
    task: ModelingTask,
    models_config: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Snapshot configured model hyperparameters for experiment config."""
    if models_config is None:
        models_config = get_active_profile().task_models_config(task)

    return {
        model_key: {"hyperparameters": dict(hyperparameters)}
        for model_key, hyperparameters in models_config.items()
    }
