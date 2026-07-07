"""
Extract and persist model feature importances under each experiment task.

Layout::

    experiments/<id>/<task>/feature_importance/<model_key>.csv

Reference: ``docs/handoff_metrics_and_hyperparams.md`` (model artifact snapshot).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator

from src.data.schema import MODELING_ARTIFACTS_DIR, MODEL_REGISTRY_KEYS, ModelingTask

__all__ = [
    "FEATURE_IMPORTANCE_DIRNAME",
    "enrich_models_snapshot",
    "export_play_type_feature_importances",
    "feature_importance_path",
    "feature_importance_relpath",
    "save_feature_importance",
    "to_feature_importance_frame",
]

FEATURE_IMPORTANCE_DIRNAME = "feature_importance"


def _task_experiment_dir(experiment_id: str, task: ModelingTask) -> Path:
    return MODELING_ARTIFACTS_DIR / "experiments" / experiment_id / task


def _best_model_task_dir(task: ModelingTask) -> Path:
    return MODELING_ARTIFACTS_DIR / "best_model" / task


def feature_importance_relpath(model_key: str) -> str:
    """Relative path from a task artifact dir to a model importance CSV."""
    return f"{FEATURE_IMPORTANCE_DIRNAME}/{model_key}.csv"


def feature_importance_path(task_dir: Path, model_key: str) -> Path:
    """Absolute path for a model's feature-importance CSV under *task_dir*."""
    return task_dir / feature_importance_relpath(model_key)


def to_feature_importance_frame(
    model: BaseEstimator,
    feature_names: list[str],
) -> pd.DataFrame | None:
    """Return a sorted importance frame when the estimator exposes importances."""
    if not hasattr(model, "feature_importances_"):
        return None

    importances = getattr(model, "feature_importances_")
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _resolve_output_dir(
    task: ModelingTask,
    *,
    experiment_id: str | None,
    to_best_model: bool,
) -> Path:
    if to_best_model:
        return _best_model_task_dir(task)
    if experiment_id:
        return _task_experiment_dir(experiment_id, task)
    raise ValueError("save_feature_importance requires experiment_id or to_best_model=True")


def save_feature_importance(
    model: BaseEstimator,
    feature_names: list[str],
    task: ModelingTask,
    model_key: str,
    *,
    experiment_id: str | None = None,
    to_best_model: bool = False,
) -> Path | None:
    """Write ``feature_importance/<model_key>.csv`` when importances are available."""
    if model_key not in MODEL_REGISTRY_KEYS:
        raise KeyError(f"unknown model key: {model_key!r}")

    frame = to_feature_importance_frame(model, feature_names)
    if frame is None:
        return None

    out_dir = _resolve_output_dir(
        task,
        experiment_id=experiment_id,
        to_best_model=to_best_model,
    )
    out_path = feature_importance_path(out_dir, model_key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return out_path


def enrich_models_snapshot(
    task: ModelingTask,
    experiment_id: str,
    models: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Add ``feature_importance`` relative paths to a models snapshot when files exist."""
    task_dir = _task_experiment_dir(experiment_id, task)
    enriched: dict[str, dict[str, object]] = {}

    for model_key, entry in models.items():
        updated = dict(entry)
        if feature_importance_path(task_dir, model_key).exists():
            updated["feature_importance"] = feature_importance_relpath(model_key)
        enriched[model_key] = updated

    return enriched


def export_play_type_feature_importances(
    *,
    experiment_id: str,
) -> dict[str, Path]:
    """
    Fit each play-type classifier on the full dataset and save importances.

    Matches the full-data refit path used in ``play_type.predict``.
    """
    from src.data.loaders import load_play_type_dataset
    from src.models import iter_classifier_builders

    X, y = load_play_type_dataset()
    feature_names = X.columns.tolist()
    saved: dict[str, Path] = {}

    for model_key, builder in iter_classifier_builders():
        model = builder()
        model.fit(X, y)
        out_path = save_feature_importance(
            model,
            feature_names,
            "play_type",
            model_key,
            experiment_id=experiment_id,
        )
        if out_path is not None:
            saved[model_key] = out_path

    return saved
