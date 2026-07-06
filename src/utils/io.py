"""
Artifact writers for modeling pipelines.

Reference: ``docs/modeling_plan.md`` Phase 2.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator

from src.data.schema import ModelingTask
from src.utils.experiments import (
    best_model_task_dir,
    resolve_task_artifacts_dir,
    task_experiment_dir,
)

__all__ = [
    "ensure_artifacts_dir",
    "save_feature_importance",
    "save_model",
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
    filename: str = "model.joblib",
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


def save_feature_importance(
    model: BaseEstimator,
    feature_names: list[str],
    task: ModelingTask,
    *,
    experiment_id: str | None = None,
    to_best_model: bool = False,
    filename: str = "feature_importance.csv",
) -> Path | None:
    """Write feature importances when the estimator exposes them."""
    if not hasattr(model, "feature_importances_"):
        return None

    importances = getattr(model, "feature_importances_")
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    if to_best_model:
        out_dir = best_model_task_dir(task)
    elif experiment_id:
        out_dir = task_experiment_dir(experiment_id, task)
    else:
        out_dir = resolve_task_artifacts_dir(task)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    frame.to_csv(out_path, index=False)
    return out_path


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
