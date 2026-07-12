"""Cross-validation smoke tests."""
from __future__ import annotations

import numpy as np

from src.data.schema import SEED
from src.evaluation.cross_validation import (
    cross_validate_classifier,
    stratified_folds,
)
from src.models import CLASSIFIER_BUILDERS


def test_stratified_folds_deterministic(play_type_subsample) -> None:
    _, y = play_type_subsample
    folds_a = stratified_folds(y, n_folds=2, seed=SEED)
    folds_b = stratified_folds(y, n_folds=2, seed=SEED)
    assert len(folds_a) == 2
    for (train_a, val_a), (train_b, val_b) in zip(folds_a, folds_b):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(val_a, val_b)


def test_classifier_cv_two_fold_smoke(play_type_subsample) -> None:
    X, y = play_type_subsample
    folds = stratified_folds(y, n_folds=2, seed=SEED)
    impute_cols = X.select_dtypes(include=np.number).columns.tolist()

    result = cross_validate_classifier(
        CLASSIFIER_BUILDERS["baseline"](),
        X,
        y,
        folds,
        impute_numeric_cols=impute_cols,
    )

    assert len(result.fold_metrics) == 2
    assert len(result.oof) == len(y)
    assert not np.isnan(result.oof).any()
    assert 0.0 <= result.oof.min() <= result.oof.max() <= 1.0
