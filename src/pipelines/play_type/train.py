"""
Play-type classification CV pipeline.

Wires data loaders, model builders, and shared evaluation into artifacts.

Usage (from project root):
    python3 -m src.pipelines.play_type.train
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loaders import load_play_type_dataset
from src.data.schema import N_FOLDS, PLAY_TYPE_ARTIFACTS_DIR, SEED
from src.evaluation.cross_validation import cross_validate_classifier, stratified_folds
from src.evaluation.model_selection import summarize_cv_results
from src.models import CLASSIFIER_BUILDERS
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


def train_play_type() -> pd.DataFrame:
    """Run play-type CV, write ``cv_results.csv`` and ``model_comparison.csv``."""
    records, _, comparison = run_play_type_cross_validation()

    write_cv_results(records, "play_type")
    out_dir = ensure_artifacts_dir("play_type")
    comparison_path = out_dir / MODEL_COMPARISON_FILENAME
    comparison.to_csv(comparison_path, index=False)

    return comparison


if __name__ == "__main__":
    comparison_df = train_play_type()
    out_path = PLAY_TYPE_ARTIFACTS_DIR / MODEL_COMPARISON_FILENAME
    print(f"Play-type CV complete → {out_path}")
    print(comparison_df.to_string(index=False))
