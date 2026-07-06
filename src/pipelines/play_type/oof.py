"""
Out-of-fold pass probabilities for play-type classifiers.

Exports ``oof_predictions.parquet`` for Task 2 nested CV.

Usage (from project root):
    python3 -m src.pipelines.play_type.oof
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders import load_play_type_dataset
from src.evaluation.model_selection import select_best_model
from src.evaluation.oof import build_oof_dataframe
from src.pipelines.play_type.train import (
    MODEL_COMPARISON_FILENAME,
    run_play_type_cross_validation,
    train_play_type,
)
from src.utils.experiments import (
    get_active_experiment,
    resolve_task_artifacts_dir,
    set_active_experiment,
)
from src.utils.io import ensure_artifacts_dir

__all__ = [
    "OOF_PREDICTIONS_FILENAME",
    "export_oof_predictions",
    "get_best_oof_proba",
]

OOF_PREDICTIONS_FILENAME = "oof_predictions.parquet"


def export_oof_predictions(*, experiment_id: str | None = None) -> pd.DataFrame:
    """
    Re-run play-type CV and write OOF pass probabilities per model.

    Columns: ``y_true``, ``oof_proba_baseline``, ``oof_proba_random_forest``,
    ``oof_proba_xgboost``.
    """
    _, y = load_play_type_dataset()
    expected_rows = len(y)

    _, oof_by_model, comparison = run_play_type_cross_validation()
    frame = build_oof_dataframe(y, oof_by_model)

    if len(frame) != expected_rows:
        raise AssertionError(
            f"OOF frame row count {len(frame):,} != expected {expected_rows:,}"
        )
    if frame.isna().any().any():
        nan_cols = frame.columns[frame.isna().any()].tolist()
        raise AssertionError(f"OOF frame contains NaNs in columns: {nan_cols}")

    exp_id = experiment_id or get_active_experiment("play_type")
    if exp_id is None:
        _, exp_id = train_play_type()

    out_dir = ensure_artifacts_dir("play_type", experiment_id=exp_id)
    comparison_path = out_dir / MODEL_COMPARISON_FILENAME
    comparison.to_csv(comparison_path, index=False)

    out_path = out_dir / OOF_PREDICTIONS_FILENAME
    frame.to_parquet(out_path, index=False)
    set_active_experiment("play_type", exp_id)
    return frame


def get_best_oof_proba(
    *,
    oof_path: Path | None = None,
    comparison_path: Path | None = None,
    experiment_id: str | None = None,
) -> pd.Series:
    """
    Return OOF pass probabilities for the best play-type model (by ``roc_auc``).

    Reads artifacts from the active or specified experiment. Falls back to legacy
    flat ``artifacts/modeling/play_type/`` when present.
    """
    artifacts_dir = resolve_task_artifacts_dir(
        "play_type",
        experiment_id=experiment_id,
    )
    oof_file = oof_path or (artifacts_dir / OOF_PREDICTIONS_FILENAME)
    comparison_file = comparison_path or (artifacts_dir / MODEL_COMPARISON_FILENAME)

    if not comparison_file.exists():
        train_play_type(experiment_id=experiment_id)
        artifacts_dir = resolve_task_artifacts_dir(
            "play_type",
            experiment_id=experiment_id,
        )
        oof_file = oof_path or (artifacts_dir / OOF_PREDICTIONS_FILENAME)
        comparison_file = comparison_path or (
            artifacts_dir / MODEL_COMPARISON_FILENAME
        )

    if not oof_file.exists():
        export_oof_predictions(experiment_id=experiment_id)
        artifacts_dir = resolve_task_artifacts_dir(
            "play_type",
            experiment_id=experiment_id,
        )
        oof_file = oof_path or (artifacts_dir / OOF_PREDICTIONS_FILENAME)
        comparison_file = comparison_path or (
            artifacts_dir / MODEL_COMPARISON_FILENAME
        )

    comparison = pd.read_csv(comparison_file)
    best_model = select_best_model(
        comparison,
        "roc_auc",
        higher_is_better=True,
    )

    oof_df = pd.read_parquet(oof_file)
    col_name = f"oof_proba_{best_model}"
    if col_name not in oof_df.columns:
        raise KeyError(
            f"OOF parquet missing column {col_name!r}; "
            f"available: {oof_df.columns.tolist()}"
        )
    return oof_df[col_name]


if __name__ == "__main__":
    oof_df = export_oof_predictions()
    artifacts_dir = resolve_task_artifacts_dir("play_type")
    out_path = artifacts_dir / OOF_PREDICTIONS_FILENAME
    print(f"OOF export complete → {out_path}")
    print(f"  rows: {len(oof_df):,}")
    print(f"  columns: {', '.join(oof_df.columns)}")
