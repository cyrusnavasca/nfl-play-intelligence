"""
Phase 5 — build modeling-ready parquet datasets from threshold selection output.

Materializes Task 1 (play-type classifier) and Task 2 (yards regressor) datasets
with one-hot encoded categoricals, median-imputed numerics, and out-of-fold
``pred_pass_proba`` for Task 2.

Usage (from project root):
    python3 -m src.selection.build_datasets
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
from sklearn.impute import SimpleImputer

from src.selection.embedded_selection import (
    build_task1_feature_matrix,
    cross_fitted_pass_proba,
)
from src.selection.feature_schema import (
    DROP_ALWAYS,
    FEATURE_SELECTION_MANIFEST_PATH,
    FEATURES_FULL_PATH,
    PROCESSED_DIR,
    TARGET_CLF,
    TARGET_REG,
    PLAY_TYPE_MODELING_PARQUET_PATH,
    TASK2_GENERATED_FEATURES,
    YARDS_GAINED_MODELING_PARQUET_PATH,
    validate_feature_schema,
)
from src.selection.threshold_selection import (
    ThresholdSelectionResults,
    run_threshold_selection,
    validate_selection_manifest,
)
from src.selection.univariate_selection import (
    binary_play_type,
    load_features_full,
    run_univariate_selection,
)
from src.selection.embedded_selection import run_embedded_selection


@dataclass(frozen=True)
class BuildDatasetsResults:
    """Container for Phase 5 modeling datasets."""

    task1: pd.DataFrame
    task2: pd.DataFrame


def load_selection_manifest(
    path=FEATURE_SELECTION_MANIFEST_PATH,
) -> dict[str, object]:
    """Load the Phase 4 selection manifest from disk."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def median_impute_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Median-impute *columns* in place; leaves other columns unchanged."""
    if not columns:
        return df

    out = df.copy()
    imp = SimpleImputer(strategy="median")
    out[columns] = imp.fit_transform(out[columns])
    return out


def build_task1_dataset(
    df: pd.DataFrame,
    selection_manifest: dict[str, object],
) -> pd.DataFrame:
    """
    Task 1 modeling frame: selected numerics + one-hot categoricals + target.

    Categorical encoding matches Phase 3 (``pd.get_dummies`` with column prefix).
    ``play_type`` is stored as pass=1 / run=0 for direct classifier training.
    """
    task1_final = selection_manifest["task1"]["final"]  # type: ignore[index]
    numeric_cols: list[str] = list(task1_final["numeric"])
    cat_cols: list[str] = list(task1_final["categorical"])

    features = build_task1_feature_matrix(df, numeric_cols, cat_cols)
    features = median_impute_columns(features, numeric_cols)

    out = features.copy()
    out[TARGET_CLF] = binary_play_type(df[TARGET_CLF])
    return out


def build_task2_dataset(
    df: pd.DataFrame,
    selection_manifest: dict[str, object],
    oof_pass_proba: pd.Series,
) -> pd.DataFrame:
    """
    Task 2 modeling frame: selected numerics + OOF pass probability + target.

    ``pred_pass_proba`` is never median-imputed (no NaNs expected from OOF CV).
    """
    task2_final = selection_manifest["task2"]["final"]  # type: ignore[index]
    numeric_cols: list[str] = list(task2_final["numeric"])
    proba_col = TASK2_GENERATED_FEATURES[0]

    if len(oof_pass_proba) != len(df):
        raise ValueError(
            f"oof_pass_proba length ({len(oof_pass_proba)}) != df rows ({len(df)})"
        )

    features = df[numeric_cols].copy()
    features = median_impute_columns(features, numeric_cols)
    features[proba_col] = oof_pass_proba.values
    features[TARGET_REG] = df[TARGET_REG].values
    return features


def _assert_no_excluded(columns: list[str], context: str) -> None:
    leaked = sorted(set(columns) & set(DROP_ALWAYS))
    if leaked:
        raise ValueError(f"{context}: excluded columns present: {leaked}")


def validate_task1_dataset(
    dataset: pd.DataFrame,
    selection_manifest: dict[str, object],
    source_df: pd.DataFrame,
    *,
    expected_rows: int,
) -> None:
    """Ensure Task 1 parquet matches manifest and leakage rules."""
    task1_final = selection_manifest["task1"]["final"]  # type: ignore[index]
    numeric_cols: list[str] = list(task1_final["numeric"])
    cat_cols: list[str] = list(task1_final["categorical"])

    expected_features = build_task1_feature_matrix(
        source_df, numeric_cols, cat_cols
    ).columns.tolist()
    expected_columns = expected_features + [TARGET_CLF]

    if list(dataset.columns) != expected_columns:
        missing = sorted(set(expected_columns) - set(dataset.columns))
        extra = sorted(set(dataset.columns) - set(expected_columns))
        raise ValueError(
            "task1 columns mismatch manifest encoding: "
            f"missing={missing}, extra={extra}"
        )

    _assert_no_excluded(dataset.columns.tolist(), "task1 dataset")
    if TARGET_REG in dataset.columns:
        raise ValueError("task1 dataset must not include yards_gained")

    for col in cat_cols:
        if col in dataset.columns:
            raise ValueError(f"task1 raw categorical not encoded: {col}")

    if len(dataset) != expected_rows:
        raise ValueError(
            f"task1 row count {len(dataset)} != expected {expected_rows}"
        )


def validate_task2_dataset(
    dataset: pd.DataFrame,
    selection_manifest: dict[str, object],
    *,
    expected_rows: int,
) -> None:
    """Ensure Task 2 parquet matches manifest and leakage rules."""
    task2_final = selection_manifest["task2"]["final"]  # type: ignore[index]
    numeric_cols: list[str] = list(task2_final["numeric"])
    generated_cols: list[str] = list(task2_final["generated"])
    expected_columns = numeric_cols + generated_cols + [TARGET_REG]

    if list(dataset.columns) != expected_columns:
        missing = sorted(set(expected_columns) - set(dataset.columns))
        extra = sorted(set(dataset.columns) - set(expected_columns))
        raise ValueError(
            "task2 columns mismatch manifest: "
            f"missing={missing}, extra={extra}"
        )

    _assert_no_excluded(dataset.columns.tolist(), "task2 dataset")
    if TARGET_CLF in dataset.columns:
        raise ValueError("task2 dataset must not include play_type")

    proba_col = TASK2_GENERATED_FEATURES[0]
    if dataset[proba_col].isna().any():
        raise ValueError(f"task2 {proba_col} contains NaN values")

    if len(dataset) != expected_rows:
        raise ValueError(
            f"task2 row count {len(dataset)} != expected {expected_rows}"
        )


def run_build_datasets(
    df: pd.DataFrame,
    threshold_results: ThresholdSelectionResults,
) -> BuildDatasetsResults:
    """Build and validate both modeling datasets from Phase 4 output."""
    validate_feature_schema(df.columns.tolist())
    manifest = threshold_results.selection_manifest
    validate_selection_manifest(manifest)

    task1 = build_task1_dataset(df, manifest)
    task2 = build_task2_dataset(df, manifest, threshold_results.oof_pass_proba)

    expected_rows = len(df)
    validate_task1_dataset(task1, manifest, df, expected_rows=expected_rows)
    validate_task2_dataset(task2, manifest, expected_rows=expected_rows)

    return BuildDatasetsResults(task1=task1, task2=task2)


def write_modeling_datasets(results: BuildDatasetsResults) -> None:
    """Write play_type_modeling.parquet and yards_gained_modeling.parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    results.task1.to_parquet(PLAY_TYPE_MODELING_PARQUET_PATH, index=False)
    results.task2.to_parquet(YARDS_GAINED_MODELING_PARQUET_PATH, index=False)


def _oof_from_manifest(df: pd.DataFrame, manifest: dict[str, object]) -> pd.Series:
    """Recompute OOF pass probability from manifest Task 1 final feature lists."""
    task1_final = manifest["task1"]["final"]  # type: ignore[index]
    X = build_task1_feature_matrix(
        df,
        list(task1_final["numeric"]),
        list(task1_final["categorical"]),
    )
    y = binary_play_type(df[TARGET_CLF])
    return cross_fitted_pass_proba(X, y)


def main() -> BuildDatasetsResults:
    df = load_features_full()
    print(f"[INFO] Loaded features_full: {df.shape}")

    if FEATURE_SELECTION_MANIFEST_PATH.exists():
        print(f"[INFO] Loading manifest → {FEATURE_SELECTION_MANIFEST_PATH}")
        manifest = load_selection_manifest()
        validate_selection_manifest(manifest)
        print("[INFO] Computing OOF pred_pass_proba from manifest Task 1 features...")
        oof_pass_proba = _oof_from_manifest(df, manifest)
        threshold_results = ThresholdSelectionResults(
            selection_manifest=manifest,
            cross_task_summary=pd.DataFrame(),
            oof_pass_proba=oof_pass_proba,
        )
    else:
        print("[INFO] No manifest found — running Phases 2–4 inline...")
        univariate_results = run_univariate_selection(df)
        embedded_results = run_embedded_selection(df, univariate_results)
        threshold_results = run_threshold_selection(
            df, univariate_results, embedded_results
        )

    print("[INFO] Building modeling datasets (Phase 5)...")
    results = run_build_datasets(df, threshold_results)
    write_modeling_datasets(results)

    t1 = results.task1
    t2 = results.task2
    manifest = threshold_results.selection_manifest
    t1_final = manifest["task1"]["final"]  # type: ignore[index]
    t2_final = manifest["task2"]["final"]  # type: ignore[index]

    print(f"[DONE] play_type_modeling → {PLAY_TYPE_MODELING_PARQUET_PATH}")
    print(f"       shape={t1.shape}  |  target={TARGET_CLF}")
    print(
        f"       manifest: {len(t1_final['numeric'])} numeric + "
        f"{len(t1_final['categorical'])} categorical → "
        f"{len(t1.columns) - 1} model columns"
    )
    print(f"       columns: {t1.columns.tolist()}")

    print(f"\n[DONE] yards_gained_modeling → {YARDS_GAINED_MODELING_PARQUET_PATH}")
    print(f"       shape={t2.shape}  |  target={TARGET_REG}")
    print(
        f"       manifest: {len(t2_final['numeric'])} numeric + "
        f"{len(t2_final['generated'])} generated"
    )
    print(f"       columns: {t2.columns.tolist()}")

    return results


if __name__ == "__main__":
    main()
