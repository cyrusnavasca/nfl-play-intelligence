"""
Task 2 — embedded feature selection via cross-validated LightGBM gain importance.

Consumes out-of-fold ``pred_pass_proba`` from ``play_type.embedded`` — never
computes OOF proba locally.

Usage (from project root):
    python3 -m src.selection.yards_gained.embedded
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import KFold

from src.selection.shared.common import (
    LGBM_PARAMS,
    N_FOLDS,
    _accumulate_fold_gain,
    _importance_dataframe,
    _median_impute_train_test,
    ensure_artifacts_dir,
    load_features_full,
    passes_univariate_gate,
)
from src.selection.shared.feature_schema import (
    DROP_ALWAYS,
    SEED,
    TARGET_REG,
    TASK2_EMBEDDED_CSV,
    TASK2_GENERATED_FEATURES,
    validate_feature_schema,
)
from src.selection.yards_gained.univariate import YardsGainedUnivariateResults


@dataclass(frozen=True)
class YardsGainedEmbeddedResults:
    """Container for Task 2 embedded selection table."""

    task2: pd.DataFrame


def _assert_no_excluded(columns: list[str], context: str) -> None:
    leaked = sorted(set(columns) & set(DROP_ALWAYS))
    if leaked:
        raise ValueError(f"{context}: excluded columns in feature matrix: {leaked}")


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


def task2_embedded_importance(
    df: pd.DataFrame,
    univariate_results: YardsGainedUnivariateResults,
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


def run_task2_embedded(
    df: pd.DataFrame,
    univariate_results: YardsGainedUnivariateResults,
    oof_pass_proba: pd.Series,
) -> YardsGainedEmbeddedResults:
    """Run Task 2 embedded selection using supplied OOF pass probability."""
    validate_feature_schema(df.columns.tolist())

    task2_df = task2_embedded_importance(df, univariate_results, oof_pass_proba)
    return YardsGainedEmbeddedResults(task2=task2_df)


def write_task2_embedded_artifacts(results: YardsGainedEmbeddedResults) -> None:
    """Write Task 2 embedded importance CSV."""
    ensure_artifacts_dir()
    results.task2.to_csv(TASK2_EMBEDDED_CSV, index=False)


def main() -> YardsGainedEmbeddedResults:
    from src.selection.play_type.embedded import run_task1_embedded
    from src.selection.play_type.univariate import run_task1_univariate
    from src.selection.yards_gained.univariate import run_task2_univariate

    df = load_features_full()
    print(f"[INFO] Loaded features_full: {df.shape}")

    print("[INFO] Running Task 1 univariate + embedded (for OOF proba)...")
    task1_univariate = run_task1_univariate(df)
    task1_embedded = run_task1_embedded(df, task1_univariate)

    print("[INFO] Running Task 2 univariate selection...")
    task2_univariate = run_task2_univariate(df)

    print("[INFO] Running Task 2 embedded selection...")
    results = run_task2_embedded(df, task2_univariate, task1_embedded.oof_pass_proba)
    write_task2_embedded_artifacts(results)

    t2_norm_sum = results.task2["embedded_importance_norm"].sum()
    print(f"[DONE] task2 embedded → {TASK2_EMBEDDED_CSV} (norm sum={t2_norm_sum:.4f})")

    print("\nTop 10 Task 2 (embedded_importance_norm):")
    print(
        results.task2.head(10).to_string(index=False, float_format="{:.4f}".format)
    )

    return results


if __name__ == "__main__":
    main()
