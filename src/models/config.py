"""Load static model hyperparameter configs from ``configs/models/``."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.data.schema import MODEL_REGISTRY_KEYS, ModelingTask

__all__ = [
    "MODEL_CONFIGS_ROOT",
    "load_model_hyperparameters",
    "snapshot_task_hyperparameters",
]

MODEL_CONFIGS_ROOT = Path("configs/models")


def load_model_hyperparameters(task: ModelingTask, model_key: str) -> dict[str, Any]:
    """Return hyperparameter dict for *model_key* on *task* from static YAML."""
    if model_key not in MODEL_REGISTRY_KEYS:
        raise KeyError(f"unknown model key: {model_key!r}")

    path = MODEL_CONFIGS_ROOT / task / f"{model_key}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"model config not found: {path}")

    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"model config must be a mapping: {path}")
    return payload


def snapshot_task_hyperparameters(task: ModelingTask) -> dict[str, dict[str, Any]]:
    """Snapshot all registry model hyperparameters for experiment config."""
    return {
        model_key: {
            "hyperparameters": load_model_hyperparameters(task, model_key),
        }
        for model_key in MODEL_REGISTRY_KEYS
    }
