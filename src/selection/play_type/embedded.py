"""
Task 1 — embedded feature selection via cross-validated LightGBM gain importance.

Also owns out-of-fold ``pred_pass_proba`` generation for Task 2.

Usage (from project root):
    python3 -m src.selection.play_type.embedded
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.selection.play_type.univariate import PlayTypeUnivariateResults
from src.selection.shared.common import (
    LGBM_PARAMS,
    N_FOLDS,
    _accumulate_fold_gain,
    _importance_dataframe,
    _median_impute_train_test,
    binary_play_type,
    chi2_significant_categoricals,
    ensure_artifacts_dir,
    load_features_full,
    passes_univariate_gate,
)
from src.selection.shared.feature_schema import (
    DROP_ALWAYS,
    SEED,
    TARGET_CLF,
    TASK1_EMBEDDED_CSV,
    TASK2_GENERATED_FEATURES,
    validate_feature_schema,
)


@dataclass(frozen=True)
class PlayTypeEmbeddedResults:
    """Container for Task 1 embedded selection outputs."""

    task1: pd.DataFrame
    oof_pass_proba: pd.Series


def _assert_no_excluded(columns: list[str], context: str) -> None:
    leaked = sorted(set(columns) & set(DROP_ALWAYS))
    if leaked:
        raise ValueError(f"{context}: excluded columns in feature matrix: {leaked}")


def build_task1_feature_matrix(
    df: pd.DataFrame,
    numeric_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    """Raw numerics plus one-hot encoded categoricals (impute inside CV)."""
    _assert_no_excluded(numeric_cols + cat_cols, "task1 feature matrix")

    parts = [df[numeric_cols].copy()]
    if cat_cols:
        cats = pd.get_dummies(
            df[cat_cols], columns=cat_cols, prefix=cat_cols, dtype=float
        )
        parts.append(cats)
    return pd.concat(parts, axis=1)


def cross_fitted_pass_proba(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_folds: int = N_FOLDS,
) -> pd.Series:
    """
    Out-of-fold predicted P(pass) using stratified CV.

    Each row's probability comes from a classifier that did not see that row
    during training (leakage guardrail for Task 2).
    """
    oof = np.full(len(y), np.nan)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    for train_idx, val_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]

        X_train_imp, X_val_imp = _median_impute_train_test(X_train, X_val)

        clf = lgb.LGBMClassifier(**LGBM_PARAMS)
        clf.fit(X_train_imp, y_train)
        oof[val_idx] = clf.predict_proba(X_val_imp)[:, 1]

    if np.isnan(oof).any():
        raise RuntimeError("cross_fitted_pass_proba produced NaN predictions")

    return pd.Series(oof, index=y.index, name=TASK2_GENERATED_FEATURES[0])


def task1_embedded_importance(
    df: pd.DataFrame,
    univariate_results: PlayTypeUnivariateResults,
) -> pd.DataFrame:
    """5-fold stratified CV LightGBM classifier; average gain importance."""
    numeric_cols = passes_univariate_gate(
        univariate_results.task1_numeric, mi_col="mi_clf"
    )
    cat_cols = chi2_significant_categoricals(univariate_results.chi2)

    X = build_task1_feature_matrix(df, numeric_cols, cat_cols)
    y = binary_play_type(df[TARGET_CLF])

    accum: dict[str, float] = defaultdict(float)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for train_idx, val_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]

        X_train_imp, _ = _median_impute_train_test(X_train, X_val)

        clf = lgb.LGBMClassifier(**LGBM_PARAMS)
        clf.fit(X_train_imp, y_train)
        _accumulate_fold_gain(clf, X.columns.tolist(), accum)

    for name in accum:
        accum[name] /= N_FOLDS

    return _importance_dataframe(dict(accum))


def run_task1_embedded(
    df: pd.DataFrame,
    univariate_results: PlayTypeUnivariateResults,
) -> PlayTypeEmbeddedResults:
    """
    Run Task 1 embedded selection and compute OOF pass probability.

    OOF proba uses Task 1 **univariate-passing** features (not final embedded
    keep lists). Task 2 embedded importance therefore uses a slightly different
    OOF vector than the modeling parquet (which recomputes OOF from final lists).
    """
    validate_feature_schema(df.columns.tolist())

    numeric_cols = passes_univariate_gate(
        univariate_results.task1_numeric, mi_col="mi_clf"
    )
    cat_cols = chi2_significant_categoricals(univariate_results.chi2)
    X_task1 = build_task1_feature_matrix(df, numeric_cols, cat_cols)
    y_clf = binary_play_type(df[TARGET_CLF])
    oof_pass_proba = cross_fitted_pass_proba(X_task1, y_clf)

    task1_df = task1_embedded_importance(df, univariate_results)

    return PlayTypeEmbeddedResults(task1=task1_df, oof_pass_proba=oof_pass_proba)


def write_task1_embedded_artifacts(results: PlayTypeEmbeddedResults) -> None:
    """Write Task 1 embedded importance CSV."""
    ensure_artifacts_dir()
    results.task1.to_csv(TASK1_EMBEDDED_CSV, index=False)


def main() -> PlayTypeEmbeddedResults:
    from src.selection.play_type.univariate import run_task1_univariate

    df = load_features_full()
    print(f"[INFO] Loaded features_full: {df.shape}")

    print("[INFO] Running Task 1 univariate selection...")
    univariate_results = run_task1_univariate(df)

    print("[INFO] Running Task 1 embedded selection...")
    results = run_task1_embedded(df, univariate_results)
    write_task1_embedded_artifacts(results)

    t1_norm_sum = results.task1["embedded_importance_norm"].sum()
    print(f"[DONE] task1 embedded → {TASK1_EMBEDDED_CSV} (norm sum={t1_norm_sum:.4f})")
    print(
        f"[INFO] OOF pred_pass_proba: mean={results.oof_pass_proba.mean():.4f}, "
        f"min={results.oof_pass_proba.min():.4f}, max={results.oof_pass_proba.max():.4f}"
    )

    print("\nTop 10 Task 1 (embedded_importance_norm):")
    print(
        results.task1.head(10).to_string(index=False, float_format="{:.4f}".format)
    )

    return results


if __name__ == "__main__":
    main()
