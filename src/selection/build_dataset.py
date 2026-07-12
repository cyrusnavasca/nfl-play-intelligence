"""
Task 1 — build modeling-ready play-type parquet dataset.

Usage (from project root):
    python3 -m src.selection.build_dataset
"""
from __future__ import annotations

import pandas as pd

from src.selection.embedded import build_task1_feature_matrix
from src.selection.shared.common import (
    _assert_no_excluded,
    binary_play_type,
    load_features_full,
    median_impute_columns,
)
from src.selection.shared.feature_schema import (
    PLAY_TYPE_MODELING_PARQUET_PATH,
    PROCESSED_DIR,
    TARGET_CLF,
    validate_feature_schema,
)
from src.selection.shared.manifest import load_selection_manifest, validate_selection_manifest


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

    for col in cat_cols:
        if col in dataset.columns:
            raise ValueError(f"task1 raw categorical not encoded: {col}")

    if len(dataset) != expected_rows:
        raise ValueError(
            f"task1 row count {len(dataset)} != expected {expected_rows}"
        )


def main() -> pd.DataFrame:
    df = load_features_full()
    print(f"[INFO] Loaded features_full: {df.shape}")

    manifest = load_selection_manifest()
    validate_selection_manifest(manifest)
    validate_feature_schema(df.columns.tolist())

    dataset = build_task1_dataset(df, manifest)
    validate_task1_dataset(dataset, manifest, df, expected_rows=len(df))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(PLAY_TYPE_MODELING_PARQUET_PATH, index=False)

    t1_final = manifest["task1"]["final"]  # type: ignore[index]
    print(f"[DONE] play_type_modeling → {PLAY_TYPE_MODELING_PARQUET_PATH}")
    print(f"       shape={dataset.shape}  |  target={TARGET_CLF}")
    print(
        f"       manifest: {len(t1_final['numeric'])} numeric + "
        f"{len(t1_final['categorical'])} categorical → "
        f"{len(dataset.columns) - 1} model columns"
    )

    return dataset


if __name__ == "__main__":
    main()
