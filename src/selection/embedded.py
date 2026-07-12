"""
Embedded feature selection via cross-validated LightGBM gain importance.

Consumes the manual feature lists from the feature config (no filter stage) and
produces mean per-fold gain importance for each numeric feature and one-hot
categorical dummy.

Usage (from project root):
    python3 -m src.selection.embedded
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.selection.shared.common import (
    LGBM_PARAMS,
    N_FOLDS,
    _accumulate_fold_gain,
    _importance_dataframe,
    _median_impute_train_test,
    binary_play_type,
    ensure_artifacts_dir,
    load_features_full,
)
from src.selection.shared.feature_schema import (
    DROP_ALWAYS,
    SEED,
    TARGET_CLF,
    TASK1_EMBEDDED_CSV,
)


@dataclass(frozen=True)
class PlayTypeEmbeddedResults:
    """Container for embedded selection outputs."""

    task1: pd.DataFrame


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
    _assert_no_excluded(numeric_cols + cat_cols, "feature matrix")

    parts = [df[numeric_cols].copy()]
    if cat_cols:
        cats = pd.get_dummies(
            df[cat_cols], columns=cat_cols, prefix=cat_cols, dtype=float
        )
        parts.append(cats)
    return pd.concat(parts, axis=1)


def embedded_importance(
    df: pd.DataFrame,
    numeric_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    """5-fold stratified CV LightGBM classifier; average gain importance."""
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


def run_embedded(
    df: pd.DataFrame,
    numeric_cols: list[str],
    cat_cols: list[str],
) -> PlayTypeEmbeddedResults:
    """Run embedded selection on the manually selected features."""
    task1_df = embedded_importance(df, numeric_cols, cat_cols)
    return PlayTypeEmbeddedResults(task1=task1_df)


def write_embedded_artifacts(results: PlayTypeEmbeddedResults) -> None:
    """Write embedded importance CSV."""
    ensure_artifacts_dir()
    results.task1.to_csv(TASK1_EMBEDDED_CSV, index=False)


def main() -> PlayTypeEmbeddedResults:
    from src.selection.feature_config import load_feature_config

    df = load_features_full()
    print(f"[INFO] Loaded features_full: {df.shape}")

    cfg = load_feature_config()
    print(
        f"[INFO] Feature config '{cfg.name}': "
        f"{len(cfg.numeric)} numeric + {len(cfg.categorical)} categorical"
    )

    results = run_embedded(df, cfg.numeric, cfg.categorical)
    write_embedded_artifacts(results)

    norm_sum = results.task1["embedded_importance_norm"].sum()
    print(f"[DONE] embedded importance → {TASK1_EMBEDDED_CSV} (norm sum={norm_sum:.4f})")

    print("\nTop 10 (embedded_importance_norm):")
    print(results.task1.head(10).to_string(index=False, float_format="{:.4f}".format))

    return results


if __name__ == "__main__":
    main()
