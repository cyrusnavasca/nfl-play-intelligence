"""
Play-type classification CV pipeline.

Wires data loaders, model builders, and shared evaluation into artifacts.

Usage (from project root):
    python3 -m src.pipelines.play_type.train
    python3 -m src.pipelines.play_type.train --experiment log_loss_v2
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.data.loaders import load_play_type_dataset
from src.data.schema import N_FOLDS, SEED
from src.evaluation.cross_validation import cross_validate_classifier, stratified_folds
from src.evaluation.model_selection import summarize_cv_results
from src.models import CLASSIFIER_BUILDERS
from src.utils.experiments import (
    allocate_experiment_id,
    build_task_result_config,
    set_active_experiment,
    update_experiment_config,
)
from src.utils.io import ensure_artifacts_dir, write_cv_results

__all__ = ["run_play_type_cross_validation", "train_play_type"]

MODEL_COMPARISON_FILENAME = "model_comparison.csv"


def _numeric_impute_columns(X: pd.DataFrame) -> list[str]:
    return X.select_dtypes(include=np.number).columns.tolist()


def run_play_type_cross_validation() -> tuple[
    list[dict],
    dict[str, np.ndarray],
    pd.DataFrame,
]:
    """
    Run 5-fold stratified CV for every registered classifier.

    Returns fold-level records, OOF probability arrays per model, and a
    comparison summary (mean ± std per metric).
    """
    X, y = load_play_type_dataset()
    folds = stratified_folds(y, N_FOLDS, SEED)
    impute_cols = _numeric_impute_columns(X)

    records: list[dict] = []
    oof_by_model: dict[str, np.ndarray] = {}

    for model_key, builder in CLASSIFIER_BUILDERS.items():
        result = cross_validate_classifier(
            builder(),
            X,
            y,
            folds,
            impute_numeric_cols=impute_cols,
        )
        oof_by_model[model_key] = result.oof
        for fold_metrics in result.fold_metrics:
            records.append({"model": model_key, **fold_metrics})

    comparison = summarize_cv_results(records)
    return records, oof_by_model, comparison


def train_play_type(*, experiment_id: str | None = None) -> tuple[pd.DataFrame, str]:
    """Run play-type CV and write artifacts under ``experiments/<id>/play_type/``."""
    exp_id = experiment_id or allocate_experiment_id()
    update_experiment_config(
        exp_id,
        {
            "seed": SEED,
            "n_folds": N_FOLDS,
        },
    )

    records, _, comparison = run_play_type_cross_validation()

    write_cv_results(records, "play_type", experiment_id=exp_id)
    out_dir = ensure_artifacts_dir("play_type", experiment_id=exp_id)
    comparison_path = out_dir / MODEL_COMPARISON_FILENAME
    comparison.to_csv(comparison_path, index=False)

    set_active_experiment("play_type", exp_id)
    update_experiment_config(
        exp_id,
        {
            "tasks": {
                "play_type": build_task_result_config(
                    "play_type",
                    comparison,
                    metric="roc_auc",
                    higher_is_better=True,
                ),
            }
        },
    )

    return comparison, exp_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run play-type model comparison CV")
    parser.add_argument(
        "--experiment",
        help="Experiment id (default: next exp_NNN)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    comparison_df, exp_id = train_play_type(experiment_id=args.experiment)
    print(f"Play-type CV complete → experiments/{exp_id}/play_type/{MODEL_COMPARISON_FILENAME}")
    print(comparison_df.to_string(index=False))
