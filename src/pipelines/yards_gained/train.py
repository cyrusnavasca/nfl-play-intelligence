"""
Yards-gained regression holdout pipeline.

Builds an augmented feature frame with OOF ``pred_pass_proba``, splits 80/20,
trains regressors on train, and evaluates on holdout.

Usage (from project root):
    python3 -m src.pipelines.yards_gained.train
    python3 -m src.pipelines.yards_gained.train --experiment tweedie_v1
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.data.loaders import load_yards_gained_dataset, yards_numeric_columns
from src.data.schema import SEED, TARGET_REG, TASK2_GENERATED_FEATURES
from src.evaluation.cross_validation import _impute_columns_train_test
from src.evaluation.metrics import regression_metrics
from src.evaluation.model_selection import select_best_model, summarize_cv_results
from src.models import REGRESSOR_BUILDERS
from src.pipelines.play_type.oof import get_best_oof_proba
from src.preprocessing.split_data import train_test_split_frame
from src.utils.experiments import (
    allocate_experiment_id,
    build_task_result_config,
    set_active_experiment,
    task_experiment_dir,
    update_experiment_config,
)
from src.utils.io import ensure_artifacts_dir

__all__ = [
    "HOLDOUT_RESULTS_FILENAME",
    "MODEL_COMPARISON_FILENAME",
    "TEST_PREDICTIONS_FILENAME",
    "build_augmented_yards_frame",
    "run_yards_gained_holdout",
    "train_yards_gained",
]

HOLDOUT_RESULTS_FILENAME = "holdout_results.csv"
MODEL_COMPARISON_FILENAME = "model_comparison.csv"
TEST_PREDICTIONS_FILENAME = "test_predictions.parquet"


def build_augmented_yards_frame(
    *,
    play_type_experiment_id: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build yards feature frame with OOF ``pred_pass_proba`` from play-type CV.

    Replaces the selection-baseline proba baked into the modeling parquet with
    out-of-fold probabilities from the best play-type model.
    """
    X_raw, y = load_yards_gained_dataset()
    numeric_cols = yards_numeric_columns(X_raw)
    oof_proba = get_best_oof_proba(experiment_id=play_type_experiment_id)

    if len(oof_proba) != len(y):
        raise AssertionError(
            f"OOF proba length {len(oof_proba):,} != yards target length {len(y):,}"
        )

    proba_col = TASK2_GENERATED_FEATURES[0]
    X = X_raw[numeric_cols].copy()
    X[proba_col] = oof_proba.values
    return X, y


def run_yards_gained_holdout(
    *,
    play_type_experiment_id: str | None = None,
) -> tuple[
    list[dict],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Train all registered regressors on 80% and score on 20% holdout.

    Returns holdout records, comparison summary, holdout results table, and
    test predictions (one row per holdout play).
    """
    X, y = build_augmented_yards_frame(
        play_type_experiment_id=play_type_experiment_id,
    )
    frame = X.copy()
    frame[TARGET_REG] = y

    X_train, X_test, y_train, y_test = train_test_split_frame(frame, TARGET_REG)
    impute_cols = yards_numeric_columns(X_train)

    records: list[dict] = []
    test_preds: dict[str, np.ndarray] = {}

    for model_key, builder in REGRESSOR_BUILDERS.items():
        X_train_imp, X_test_imp = _impute_columns_train_test(
            X_train,
            X_test,
            impute_cols,
        )

        model = builder()
        model.fit(X_train_imp, y_train)
        y_pred = model.predict(X_test_imp)
        test_preds[model_key] = np.asarray(y_pred, dtype=float)

        metrics = regression_metrics(y_test, y_pred)
        records.append({"model": model_key, **metrics})

    holdout_df = pd.DataFrame(records)
    comparison = summarize_cv_results(records)

    pred_frame = pd.DataFrame({"y_true": y_test.values})
    for model_key, preds in test_preds.items():
        pred_frame[f"y_pred_{model_key}"] = preds

    return records, comparison, holdout_df, pred_frame


def train_yards_gained(
    *,
    experiment_id: str | None = None,
    play_type_experiment_id: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Run holdout evaluation and write yards-gained experiment artifacts."""
    exp_id = experiment_id or allocate_experiment_id()
    update_experiment_config(
        exp_id,
        {
            "seed": SEED,
            "play_type_experiment": play_type_experiment_id,
        },
    )

    _, comparison, holdout_df, pred_frame = run_yards_gained_holdout(
        play_type_experiment_id=play_type_experiment_id,
    )

    out_dir = ensure_artifacts_dir("yards_gained", experiment_id=exp_id)
    holdout_df.to_csv(out_dir / HOLDOUT_RESULTS_FILENAME, index=False)
    comparison.to_csv(out_dir / MODEL_COMPARISON_FILENAME, index=False)
    pred_frame.to_parquet(out_dir / TEST_PREDICTIONS_FILENAME, index=False)

    set_active_experiment("yards_gained", exp_id)
    update_experiment_config(
        exp_id,
        {
            "tasks": {
                "yards_gained": build_task_result_config(
                    "yards_gained",
                    comparison,
                    metric="rmse",
                    higher_is_better=False,
                ),
            }
        },
    )

    return comparison, exp_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run yards-gained holdout evaluation")
    parser.add_argument(
        "--experiment",
        help="Experiment id (default: next exp_NNN)",
    )
    parser.add_argument(
        "--play-type-experiment",
        help="Play-type experiment id for OOF proba (default: active)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    comparison_df, exp_id = train_yards_gained(
        experiment_id=args.experiment,
        play_type_experiment_id=args.play_type_experiment,
    )
    out_path = task_experiment_dir(exp_id, "yards_gained") / MODEL_COMPARISON_FILENAME
    print(f"Yards-gained holdout complete → {out_path}")
    best_model = select_best_model(
        comparison_df,
        "rmse",
        higher_is_better=False,
    )
    print(f"  best model: {best_model}")
    print(comparison_df.to_string(index=False))
