"""
Generic cross-validation loops for classifiers and regressors.

Fold generation matches selection (``StratifiedKFold`` with ``SEED=42``).
Median imputation is fit on the train fold only.

Reference: ``docs/modeling_plan.md`` Phase 2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

from src.evaluation.metrics import classification_metrics, regression_metrics

__all__ = [
    "CrossValidationResult",
    "cross_validate_classifier",
    "cross_validate_regressor",
    "stratified_folds",
]


@dataclass(frozen=True)
class CrossValidationResult:
    """Per-fold metric dicts plus out-of-fold predictions."""

    fold_metrics: list[dict[str, float]]
    oof: np.ndarray


def stratified_folds(
    y: pd.Series | np.ndarray,
    n_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Deterministic stratified train/validation index pairs.

    Uses the same ``StratifiedKFold`` settings as selection embedded CV.
    """
    y_arr = np.asarray(y)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    # X is unused by StratifiedKFold.split; pass a dummy array of matching length.
    dummy_X = np.zeros(len(y_arr))
    return [(train_idx, val_idx) for train_idx, val_idx in cv.split(dummy_X, y_arr)]


def _impute_columns_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median-impute *columns* — fit on train, transform test."""
    if not columns:
        return X_train, X_test

    train = X_train.copy()
    test = X_test.copy()
    imp = SimpleImputer(strategy="median")
    train[columns] = imp.fit_transform(train[columns])
    test[columns] = imp.transform(test[columns])
    return train, test


def cross_validate_classifier(
    model: ClassifierMixin,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    impute_numeric_cols: list[str],
) -> CrossValidationResult:
    """
    Run stratified CV for a classifier; return per-fold metrics and OOF proba.

    OOF array holds predicted P(positive) for each row (column index 1).
    """
    y_arr = np.asarray(y)
    oof = np.full(len(y_arr), np.nan, dtype=float)
    fold_metrics: list[dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y_arr[train_idx]
        y_val = y_arr[val_idx]

        X_train_imp, X_val_imp = _impute_columns_train_test(
            X_train, X_val, impute_numeric_cols
        )

        fold_model = _clone_estimator(model)
        fold_model.fit(X_train_imp, y_train)
        val_proba = fold_model.predict_proba(X_val_imp)[:, 1]
        oof[val_idx] = val_proba

        metrics = classification_metrics(y_val, val_proba)
        fold_metrics.append({"fold": float(fold_idx), **metrics})

    if np.isnan(oof).any():
        raise RuntimeError("cross_validate_classifier produced NaN OOF predictions")

    return CrossValidationResult(fold_metrics=fold_metrics, oof=oof)


def cross_validate_regressor(
    model: RegressorMixin,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    impute_numeric_cols: list[str],
) -> CrossValidationResult:
    """Run CV for a regressor; return per-fold metrics and OOF predictions."""
    y_arr = np.asarray(y, dtype=float)
    oof = np.full(len(y_arr), np.nan, dtype=float)
    fold_metrics: list[dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y_arr[train_idx]
        y_val = y_arr[val_idx]

        X_train_imp, X_val_imp = _impute_columns_train_test(
            X_train, X_val, impute_numeric_cols
        )

        fold_model = _clone_estimator(model)
        fold_model.fit(X_train_imp, y_train)
        val_pred = fold_model.predict(X_val_imp)
        oof[val_idx] = val_pred

        metrics = regression_metrics(y_val, val_pred)
        fold_metrics.append({"fold": float(fold_idx), **metrics})

    if np.isnan(oof).any():
        raise RuntimeError("cross_validate_regressor produced NaN OOF predictions")

    return CrossValidationResult(fold_metrics=fold_metrics, oof=oof)


def _clone_estimator(model: ClassifierMixin | RegressorMixin):
    """Fresh estimator instance for each fold (sklearn clone when available)."""
    from sklearn.base import clone

    return clone(model)
