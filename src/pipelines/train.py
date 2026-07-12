"""
Play-type classification CV pipeline.

Wires data loaders, model builders, and shared evaluation into artifacts.

Usage (from project root):
    python3 -m src.pipelines.train
    python3 -m src.pipelines.train --experiment log_loss_v2
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.data.loaders import load_play_type_dataset
from src.data.schema import N_FOLDS, SEED
from src.evaluation.cross_validation import cross_validate_classifier, stratified_folds
from src.evaluation.feature_importance import export_feature_importances
from src.evaluation.model_selection import summarize_cv_results
from src.models import iter_classifier_builders
from src.utils.experiment_profile import get_active_profile_or_none
from src.utils.experiments import (
    allocate_experiment_id,
    build_result_config,
    set_active_experiment,
    update_experiment_config,
)
from src.utils.io import ensure_artifacts_dir, write_cv_results

__all__ = ["run_play_type_cross_validation", "train_play_type"]

MODEL_COMPARISON_FILENAME = "model_comparison.csv"


def _numeric_impute_columns(X: pd.DataFrame) -> list[str]:
    return X.select_dtypes(include=np.number).columns.tolist()


def _resolve_run_options(
    *,
    n_folds: int | None,
    seed: int | None,
) -> tuple[int, int]:
    profile = get_active_profile_or_none()
    resolved_n_folds = n_folds
    resolved_seed = seed
    if profile is not None:
        if resolved_n_folds is None:
            resolved_n_folds = profile.n_folds
        if resolved_seed is None:
            resolved_seed = profile.seed
    return resolved_n_folds or N_FOLDS, resolved_seed or SEED


def run_play_type_cross_validation(
    *,
    n_folds: int | None = None,
    seed: int | None = None,
) -> tuple[
    list[dict],
    dict[str, np.ndarray],
    pd.DataFrame,
]:
    """
    Run stratified CV for each classifier listed in the active profile.

    Returns fold-level records, OOF probability arrays per model, and a
    comparison summary (mean ± std per metric).
    """
    folds_n, folds_seed = _resolve_run_options(n_folds=n_folds, seed=seed)
    X, y = load_play_type_dataset()
    folds = stratified_folds(y, folds_n, folds_seed)
    impute_cols = _numeric_impute_columns(X)

    records: list[dict] = []
    oof_by_model: dict[str, np.ndarray] = {}

    for model_key, builder in iter_classifier_builders():
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
    """Run play-type CV and write artifacts under ``experiments/<id>/``."""
    profile = get_active_profile_or_none()
    exp_id = experiment_id or allocate_experiment_id()
    config_patch: dict[str, object] = {
        "seed": profile.seed if profile else SEED,
        "n_folds": profile.n_folds if profile else N_FOLDS,
    }
    if profile is not None:
        config_patch["name"] = profile.name
        if profile.description:
            config_patch["description"] = profile.description
        if profile.source_path is not None:
            config_patch["profile"] = str(profile.source_path)
    update_experiment_config(exp_id, config_patch)

    records, _, comparison = run_play_type_cross_validation()

    write_cv_results(records, experiment_id=exp_id)
    out_dir = ensure_artifacts_dir(experiment_id=exp_id)
    comparison_path = out_dir / MODEL_COMPARISON_FILENAME
    comparison.to_csv(comparison_path, index=False)

    export_feature_importances(experiment_id=exp_id)

    set_active_experiment(exp_id)
    models_config = profile.models_config() if profile else None
    update_experiment_config(
        exp_id,
        build_result_config(
            comparison,
            metric="roc_auc",
            higher_is_better=True,
            experiment_id=exp_id,
            models_config=models_config,
        ),
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
    from src.utils.experiment_profile import load_experiment_profile, use_experiment_profile
    from src.utils.experiment_profile import DEFAULT_PROFILE_PATH

    args = _parse_args()
    with use_experiment_profile(load_experiment_profile(DEFAULT_PROFILE_PATH)):
        comparison_df, exp_id = train_play_type(experiment_id=args.experiment)
    print(f"Play-type CV complete → experiments/{exp_id}/{MODEL_COMPARISON_FILENAME}")
    print(comparison_df.to_string(index=False))
