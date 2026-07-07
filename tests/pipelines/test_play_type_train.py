"""Play-type pipeline integration smoke tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import SEED
from src.evaluation.cross_validation import cross_validate_classifier, stratified_folds
from src.evaluation.model_selection import summarize_cv_results
from src.models import CLASSIFIER_BUILDERS
from src.pipelines.play_type.train import run_play_type_cross_validation


def _run_cv_on_subsample(X: pd.DataFrame, y: pd.Series, *, n_folds: int = 2):
    folds = stratified_folds(y, n_folds=n_folds, seed=SEED)
    impute_cols = X.select_dtypes(include=np.number).columns.tolist()
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


def test_play_type_cv_smoke_on_subsample(play_type_subsample, monkeypatch) -> None:
    X, y = play_type_subsample
    monkeypatch.setattr(
        "src.pipelines.play_type.train.load_play_type_dataset",
        lambda: (X, y),
    )

    records, oof_by_model, comparison = run_play_type_cross_validation(n_folds=2)

    assert len(comparison) == len(CLASSIFIER_BUILDERS)
    assert len(records) == len(CLASSIFIER_BUILDERS) * 2
    for oof in oof_by_model.values():
        assert len(oof) == len(y)
        assert not np.isnan(oof).any()

    baseline_auc = comparison.loc[
        comparison["model"] == "baseline", "roc_auc_mean"
    ].iloc[0]
    assert abs(baseline_auc - 0.5) < 0.05


def test_play_type_cv_reproducible_on_subsample(play_type_subsample) -> None:
    X, y = play_type_subsample
    _, _, comparison_a = _run_cv_on_subsample(X, y)
    _, _, comparison_b = _run_cv_on_subsample(X, y)
    pd.testing.assert_frame_equal(
        comparison_a.sort_values("model").reset_index(drop=True),
        comparison_b.sort_values("model").reset_index(drop=True),
    )
