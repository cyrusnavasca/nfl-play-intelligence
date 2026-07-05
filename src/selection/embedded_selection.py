"""
Phase 3 — embedded feature selection via cross-validated LightGBM gain importance.

Refines univariate-passing candidates using model-native feature importance.
Task 2 uses out-of-fold ``pred_pass_proba`` (never in-sample).

Usage (from project root):
    python3 -m src.selection.embedded_selection
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold

from src.selection.feature_schema import (
    ARTIFACTS_DIR,
    CAT_FEATURES,
    DROP_ALWAYS,
    FEATURES_FULL_PATH,
    SEED,
    TARGET_CLF,
    TARGET_REG,
    TASK1_EMBEDDED_CSV,
    TASK2_EMBEDDED_CSV,
    TASK2_GENERATED_FEATURES,
    validate_feature_schema,
)
from src.selection.univariate_selection import (
    UnivariateSelectionResults,
    binary_play_type,
    chi2_significant_features,
    load_features_full,
    passes_univariate_gate,
    run_univariate_selection,
)

N_FOLDS = 5

LGBM_PARAMS: dict[str, object] = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "verbosity": -1,
    "importance_type": "gain",
}


@dataclass(frozen=True)
class EmbeddedSelectionResults:
    """Container for Phase 3 embedded selection importance tables."""

    task1: pd.DataFrame
    task2: pd.DataFrame
    oof_pass_proba: pd.Series


def _assert_no_excluded(columns: list[str], context: str) -> None:
    leaked = sorted(set(columns) & set(DROP_ALWAYS))
    if leaked:
        raise ValueError(f"{context}: excluded columns in feature matrix: {leaked}")


def chi2_significant_categoricals(chi2_df: pd.DataFrame) -> list[str]:
    """χ²-significant raw categoricals only (exclude binary/discrete numerics)."""
    passed = set(chi2_significant_features(chi2_df))
    return [col for col in CAT_FEATURES if col in passed]


def _median_impute_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    imp = SimpleImputer(strategy="median")
    train_vals = imp.fit_transform(X_train)
    test_vals = imp.transform(X_test)
    return (
        pd.DataFrame(train_vals, columns=X_train.columns, index=X_train.index),
        pd.DataFrame(test_vals, columns=X_test.columns, index=X_test.index),
    )


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


def build_task2_feature_matrix(
    df: pd.DataFrame,
    numeric_cols: list[str],
    pred_pass_proba: pd.Series,
) -> pd.DataFrame:
    """Raw numerics plus out-of-fold pass probability (impute inside CV)."""
    _assert_no_excluded(numeric_cols, "task2 feature matrix")

    proba_col = TASK2_GENERATED_FEATURES[0]
    X = df[numeric_cols].copy()
    X[proba_col] = pred_pass_proba.values
    return X


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


def _accumulate_fold_gain(
    model: lgb.LGBMClassifier | lgb.LGBMRegressor,
    feature_names: list[str],
    accum: dict[str, float],
) -> None:
    gains = model.booster_.feature_importance(importance_type="gain")
    for name, gain in zip(feature_names, gains):
        accum[name] += float(gain)


def _importance_dataframe(accum: dict[str, float]) -> pd.DataFrame:
    total = sum(accum.values())
    rows = [
        {
            "feature": name,
            "embedded_importance": gain,
            "embedded_importance_norm": gain / total if total > 0 else 0.0,
        }
        for name, gain in accum.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("embedded_importance_norm", ascending=False)
        .reset_index(drop=True)
    )


def task1_embedded_importance(
    df: pd.DataFrame,
    univariate_results: UnivariateSelectionResults,
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


def task2_embedded_importance(
    df: pd.DataFrame,
    univariate_results: UnivariateSelectionResults,
    oof_pass_proba: pd.Series,
) -> pd.DataFrame:
    """5-fold CV LightGBM regressor; average gain importance."""
    numeric_cols = passes_univariate_gate(
        univariate_results.task2_numeric, mi_col="mi_reg"
    )

    X = build_task2_feature_matrix(df, numeric_cols, oof_pass_proba)
    y = df[TARGET_REG].fillna(df[TARGET_REG].median())

    accum: dict[str, float] = defaultdict(float)
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for train_idx, val_idx in cv.split(X):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]

        X_train_imp, _ = _median_impute_train_test(X_train, X_val)

        reg = lgb.LGBMRegressor(**LGBM_PARAMS)
        reg.fit(X_train_imp, y_train)
        _accumulate_fold_gain(reg, X.columns.tolist(), accum)

    for name in accum:
        accum[name] /= N_FOLDS

    return _importance_dataframe(dict(accum))


def run_embedded_selection(
    df: pd.DataFrame,
    univariate_results: UnivariateSelectionResults | None = None,
) -> EmbeddedSelectionResults:
    """Run Phase 3 embedded selection for both tasks."""
    validate_feature_schema(df.columns.tolist())

    if univariate_results is None:
        univariate_results = run_univariate_selection(df)

    task1_numeric = passes_univariate_gate(
        univariate_results.task1_numeric, mi_col="mi_clf"
    )
    task1_cats = chi2_significant_categoricals(univariate_results.chi2)
    X_task1 = build_task1_feature_matrix(df, task1_numeric, task1_cats)
    y_clf = binary_play_type(df[TARGET_CLF])
    oof_pass_proba = cross_fitted_pass_proba(X_task1, y_clf)

    task1_df = task1_embedded_importance(df, univariate_results)
    task2_df = task2_embedded_importance(df, univariate_results, oof_pass_proba)

    return EmbeddedSelectionResults(
        task1=task1_df,
        task2=task2_df,
        oof_pass_proba=oof_pass_proba,
    )


def write_embedded_selection_artifacts(results: EmbeddedSelectionResults) -> None:
    """Write Phase 3 embedded selection importance CSVs."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    results.task1.to_csv(TASK1_EMBEDDED_CSV, index=False)
    results.task2.to_csv(TASK2_EMBEDDED_CSV, index=False)


def main() -> EmbeddedSelectionResults:
    df = load_features_full()
    print(f"[INFO] Loaded features_full: {df.shape}")

    print("[INFO] Running univariate selection (Phase 2)...")
    univariate_results = run_univariate_selection(df)

    print("[INFO] Running embedded selection (Phase 3)...")
    results = run_embedded_selection(df, univariate_results)
    write_embedded_selection_artifacts(results)

    t1_norm_sum = results.task1["embedded_importance_norm"].sum()
    t2_norm_sum = results.task2["embedded_importance_norm"].sum()
    print(f"[DONE] task1 embedded → {TASK1_EMBEDDED_CSV} (norm sum={t1_norm_sum:.4f})")
    print(f"[DONE] task2 embedded → {TASK2_EMBEDDED_CSV} (norm sum={t2_norm_sum:.4f})")
    print(
        f"[INFO] OOF pred_pass_proba: mean={results.oof_pass_proba.mean():.4f}, "
        f"min={results.oof_pass_proba.min():.4f}, max={results.oof_pass_proba.max():.4f}"
    )

    print("\nTop 10 Task 1 (embedded_importance_norm):")
    print(
        results.task1.head(10).to_string(index=False, float_format="{:.4f}".format)
    )
    print("\nTop 10 Task 2 (embedded_importance_norm):")
    print(
        results.task2.head(10).to_string(index=False, float_format="{:.4f}".format)
    )

    return results


if __name__ == "__main__":
    main()
