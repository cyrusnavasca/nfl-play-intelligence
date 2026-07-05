"""
Artifact writers for modeling pipelines.

Reference: ``docs/modeling_plan.md`` Phase 2.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator

from src.data.schema import (
    MODELING_ARTIFACTS_DIR,
    PLAY_TYPE_ARTIFACTS_DIR,
    YARDS_GAINED_ARTIFACTS_DIR,
    ModelingTask,
)

__all__ = [
    "ensure_artifacts_dir",
    "save_model",
    "write_cv_results",
    "write_run_summary",
]

_TASK_DIRS: dict[ModelingTask, Path] = {
    "play_type": PLAY_TYPE_ARTIFACTS_DIR,
    "yards_gained": YARDS_GAINED_ARTIFACTS_DIR,
}


def ensure_artifacts_dir(task: ModelingTask) -> Path:
    """Create and return the artifact directory for *task*."""
    path = _TASK_DIRS[task]
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model(
    model: BaseEstimator,
    task: ModelingTask,
    *,
    subdir: str = "best_model",
    filename: str = "model.joblib",
) -> Path:
    """Persist a fitted estimator under ``artifacts/modeling/<task>/<subdir>/``."""
    out_dir = ensure_artifacts_dir(task) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    joblib.dump(model, out_path)
    return out_path


def write_cv_results(
    records: list[dict],
    task: ModelingTask,
    *,
    filename: str = "cv_results.csv",
) -> Path:
    """Write fold-level CV records to ``artifacts/modeling/<task>/``."""
    out_dir = ensure_artifacts_dir(task)
    out_path = out_dir / filename
    pd.DataFrame(records).to_csv(out_path, index=False)
    return out_path


def write_run_summary(
    summary: dict[str, Any],
    *,
    path: Path | None = None,
) -> Path:
    """
    Write pipeline run metadata as JSON.

    Adds an ISO-8601 UTC timestamp when ``timestamp`` is not already set.
    """
    out_path = path or (MODELING_ARTIFACTS_DIR / "run_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(summary)
    payload.setdefault(
        "timestamp",
        datetime.now(timezone.utc).isoformat(),
    )

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out_path
