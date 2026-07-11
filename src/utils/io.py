"""
Artifact writers for modeling pipelines.

Reference: ``docs/modeling_plan.md`` Phase 2.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer

from src.data.schema import ModelingTask
from src.preprocessing.target_transform import TargetTransform
from src.utils.experiments import (
    best_model_task_dir,
    resolve_task_artifacts_dir,
    task_experiment_dir,
)

MODEL_FILENAME = "model.joblib"
TARGET_TRANSFORM_FILENAME = "target_transform.joblib"
FEATURE_IMPUTER_FILENAME = "imputer.joblib"

__all__ = [
    "FEATURE_IMPUTER_FILENAME",
    "MODEL_FILENAME",
    "TARGET_TRANSFORM_FILENAME",
    "ensure_artifacts_dir",
    "load_feature_imputer",
    "load_model",
    "load_target_transform",
    "save_feature_imputer",
    "save_model",
    "save_target_transform",
    "write_cv_results",
]


def ensure_artifacts_dir(
    task: ModelingTask,
    *,
    experiment_id: str | None = None,
) -> Path:
    """Create and return the artifact directory for *task*."""
    if experiment_id:
        return task_experiment_dir(experiment_id, task)
    return resolve_task_artifacts_dir(task)


def save_model(
    model: BaseEstimator,
    task: ModelingTask,
    *,
    experiment_id: str | None = None,
    to_best_model: bool = False,
    filename: str = MODEL_FILENAME,
) -> Path:
    """Persist a fitted estimator under an experiment or ``best_model/<task>/``."""
    if to_best_model:
        out_dir = best_model_task_dir(task)
    elif experiment_id:
        out_dir = task_experiment_dir(experiment_id, task)
    else:
        out_dir = resolve_task_artifacts_dir(task)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    joblib.dump(model, out_path)
    return out_path


def save_target_transform(
    transform: TargetTransform,
    task: ModelingTask,
    *,
    experiment_id: str | None = None,
    to_best_model: bool = False,
    filename: str = TARGET_TRANSFORM_FILENAME,
) -> Path:
    """Persist a fitted target transform alongside task artifacts."""
    if to_best_model:
        out_dir = best_model_task_dir(task)
    elif experiment_id:
        out_dir = task_experiment_dir(experiment_id, task)
    else:
        out_dir = resolve_task_artifacts_dir(task)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    joblib.dump(transform, out_path)
    return out_path


def load_model(path: Path | str) -> BaseEstimator:
    """Load a fitted estimator from a joblib artifact."""
    return joblib.load(path)


def load_target_transform(path: Path | str) -> TargetTransform:
    """Load a fitted target transform from a joblib artifact."""
    transform = joblib.load(path)
    if not isinstance(transform, TargetTransform):
        raise TypeError(
            f"expected TargetTransform at {path}, got {type(transform).__name__}"
        )
    return transform


def save_feature_imputer(
    imputer: SimpleImputer,
    task: ModelingTask,
    *,
    experiment_id: str | None = None,
    to_best_model: bool = False,
    filename: str = FEATURE_IMPUTER_FILENAME,
) -> Path:
    """Persist a fitted median imputer alongside task artifacts."""
    if to_best_model:
        out_dir = best_model_task_dir(task)
    elif experiment_id:
        out_dir = task_experiment_dir(experiment_id, task)
    else:
        out_dir = resolve_task_artifacts_dir(task)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    joblib.dump(imputer, out_path)
    return out_path


def load_feature_imputer(path: Path | str) -> SimpleImputer:
    """Load a fitted feature imputer from a joblib artifact."""
    imputer = joblib.load(path)
    if not isinstance(imputer, SimpleImputer):
        raise TypeError(
            f"expected SimpleImputer at {path}, got {type(imputer).__name__}"
        )
    return imputer


def write_cv_results(
    records: list[dict],
    task: ModelingTask,
    *,
    experiment_id: str | None = None,
    filename: str = "cv_results.csv",
) -> Path:
    """Write fold-level CV records for *task*."""
    out_dir = ensure_artifacts_dir(task, experiment_id=experiment_id)
    out_path = out_dir / filename
    pd.DataFrame(records).to_csv(out_path, index=False)
    return out_path
