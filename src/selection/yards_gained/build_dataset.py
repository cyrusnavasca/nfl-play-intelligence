"""
Task 2 — build modeling-ready yards-gained parquet dataset.

Usage (from project root):
    python3 -m src.selection.yards_gained.build_dataset
"""
from __future__ import annotations

import pandas as pd

from src.selection.play_type.embedded import (
    build_task1_feature_matrix,
    cross_fitted_pass_proba,
)
from src.selection.shared.common import (
    _assert_no_excluded,
    binary_play_type,
    load_features_full,
    median_impute_columns,
)
from src.selection.shared.feature_schema import (
    PROCESSED_DIR,
    TARGET_CLF,
    TARGET_REG,
    TASK2_GENERATED_FEATURES,
    YARDS_GAINED_MODELING_PARQUET_PATH,
    validate_feature_schema,
)
from src.selection.shared.manifest import load_selection_manifest, validate_selection_manifest


def oof_pass_proba_from_manifest(
    df: pd.DataFrame,
    manifest: dict[str, object],
) -> pd.Series:
    """Recompute OOF pass probability from manifest Task 1 final feature lists."""
    task1_final = manifest["task1"]["final"]  # type: ignore[index]
    X = build_task1_feature_matrix(
        df,
        list(task1_final["numeric"]),
        list(task1_final["categorical"]),
    )
    y = binary_play_type(df[TARGET_CLF])
    return cross_fitted_pass_proba(X, y)


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


def _oof_from_manifest(df: pd.DataFrame, manifest: dict[str, object]) -> pd.Series:
    """Alias for backward compatibility."""
    return oof_pass_proba_from_manifest(df, manifest)


def main() -> pd.DataFrame:
    df = load_features_full()
    print(f"[INFO] Loaded features_full: {df.shape}")

    manifest = load_selection_manifest()
    validate_selection_manifest(manifest)
    validate_feature_schema(df.columns.tolist())

    print("[INFO] Computing OOF pred_pass_proba from manifest Task 1 features...")
    oof_pass_proba = oof_pass_proba_from_manifest(df, manifest)

    dataset = build_task2_dataset(df, manifest, oof_pass_proba)
    validate_task2_dataset(dataset, manifest, expected_rows=len(df))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(YARDS_GAINED_MODELING_PARQUET_PATH, index=False)

    t2_final = manifest["task2"]["final"]  # type: ignore[index]
    print(f"[DONE] yards_gained_modeling → {YARDS_GAINED_MODELING_PARQUET_PATH}")
    print(f"       shape={dataset.shape}  |  target={TARGET_REG}")
    print(
        f"       manifest: {len(t2_final['numeric'])} numeric + "
        f"{len(t2_final['generated'])} generated"
    )

    return dataset


if __name__ == "__main__":
    main()
