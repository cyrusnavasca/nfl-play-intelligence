"""
Artifact writers for the modeling pipeline.

Reference: ``docs/modeling_plan.md`` Phase 2.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator

from src.utils.experiments import (
    best_model_dir,
    experiment_root,
    resolve_artifacts_dir,
)

MODEL_FILENAME = "model.joblib"

__all__ = [
    "MODEL_FILENAME",
    "ensure_artifacts_dir",
    "load_model",
    "save_model",
    "write_cv_results",
]


def ensure_artifacts_dir(*, experiment_id: str | None = None) -> Path:
    """Create and return the artifact directory."""
    if experiment_id:
        return experiment_root(experiment_id)
    return resolve_artifacts_dir()


def save_model(
    model: BaseEstimator,
    *,
    experiment_id: str | None = None,
    to_best_model: bool = False,
    filename: str = MODEL_FILENAME,
) -> Path:
    """Persist a fitted estimator under an experiment or ``best_model/``."""
    if to_best_model:
        out_dir = best_model_dir()
    elif experiment_id:
        out_dir = experiment_root(experiment_id)
    else:
        out_dir = resolve_artifacts_dir()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    joblib.dump(model, out_path)
    return out_path


def load_model(path: Path | str) -> BaseEstimator:
    """Load a fitted estimator from a joblib artifact."""
    return joblib.load(path)


def write_cv_results(
    records: list[dict],
    *,
    experiment_id: str | None = None,
    filename: str = "cv_results.csv",
) -> Path:
    """Write fold-level CV records."""
    out_dir = ensure_artifacts_dir(experiment_id=experiment_id)
    out_path = out_dir / filename
    pd.DataFrame(records).to_csv(out_path, index=False)
    return out_path
