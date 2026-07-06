"""
Modeling schema and data contract.

Single source of truth for modeling paths, targets, artifact locations, and
model registry keys. Downstream modeling modules must import column-related
constants from here — no hardcoded column names elsewhere.

Reference: ``docs/modeling_plan.md`` Phase 1 (migrated to ``src/data/`` in Phase 2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from src.selection.shared.feature_schema import (
    DROP_ALWAYS,
    PLAY_TYPE_MODELING_PARQUET_PATH,
    TARGET_CLF,
    TARGET_REG,
    TASK2_GENERATED_FEATURES,
    YARDS_GAINED_MODELING_PARQUET_PATH,
    SEED,
)

# Matches ``src.selection.shared.common.N_FOLDS`` (defined there to avoid
# importing lightgbm when loading schema-only modules).
N_FOLDS = 5

# Re-export selection constants used across modeling.
__all__ = [
    "DROP_ALWAYS",
    "EXPECTED_PLAY_TYPE_FEATURE_COUNT",
    "EXPECTED_YARDS_NUMERIC_COUNT",
    "BEST_MODEL_DIR",
    "EXPERIMENTS_DIR",
    "MODELING_ARTIFACTS_DIR",
    "MODEL_REGISTRY_KEYS",
    "ModelingParquetValidationError",
    "ModelingTask",
    "N_FOLDS",
    "PLAY_TYPE_ARTIFACTS_DIR",
    "PLAY_TYPE_MODELING_PARQUET_PATH",
    "SEED",
    "TARGET_CLF",
    "TARGET_REG",
    "TASK2_GENERATED_FEATURES",
    "YARDS_GAINED_ARTIFACTS_DIR",
    "YARDS_GAINED_MODELING_PARQUET_PATH",
    "validate_modeling_parquet",
]

# ---------------------------------------------------------------------------
# Modeling parquet column counts (from feature_selection_manifest.json final)
# ---------------------------------------------------------------------------

EXPECTED_PLAY_TYPE_FEATURE_COUNT = 34
EXPECTED_YARDS_NUMERIC_COUNT = 21

# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------

MODELING_ARTIFACTS_DIR = Path("artifacts/modeling")
EXPERIMENTS_DIR = MODELING_ARTIFACTS_DIR / "experiments"
BEST_MODEL_DIR = MODELING_ARTIFACTS_DIR / "best_model"
PLAY_TYPE_ARTIFACTS_DIR = MODELING_ARTIFACTS_DIR / "play_type"
YARDS_GAINED_ARTIFACTS_DIR = MODELING_ARTIFACTS_DIR / "yards_gained"

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY_KEYS: tuple[str, ...] = ("baseline", "random_forest", "xgboost")

ModelingTask = Literal["play_type", "yards_gained"]


class ModelingParquetValidationError(ValueError):
    """Raised when a modeling parquet does not match the data contract."""


def validate_modeling_parquet(
    path: Path | str,
    task: ModelingTask,
) -> pd.DataFrame:
    """
    Load and validate a modeling parquet for *task*.

    Checks:
      - row count > 0
      - expected target column present
      - no ``DROP_ALWAYS`` columns in the frame
      - task-specific feature layout and leakage rules
    """
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise ModelingParquetValidationError(f"{task}: parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    if len(df) == 0:
        raise ModelingParquetValidationError(f"{task}: parquet has zero rows")

    leaked = sorted(set(df.columns) & set(DROP_ALWAYS))
    if leaked:
        raise ModelingParquetValidationError(
            f"{task}: DROP_ALWAYS columns present: {leaked}"
        )

    if task == "play_type":
        _validate_play_type_parquet(df)
    elif task == "yards_gained":
        _validate_yards_gained_parquet(df)
    else:
        raise ModelingParquetValidationError(f"unknown modeling task: {task!r}")

    return df


def _validate_play_type_parquet(df: pd.DataFrame) -> None:
    if TARGET_CLF not in df.columns:
        raise ModelingParquetValidationError(
            f"play_type: missing target column {TARGET_CLF!r}"
        )
    if TARGET_REG in df.columns:
        raise ModelingParquetValidationError(
            f"play_type: must not include {TARGET_REG!r}"
        )

    feature_cols = [col for col in df.columns if col != TARGET_CLF]
    if len(feature_cols) != EXPECTED_PLAY_TYPE_FEATURE_COUNT:
        raise ModelingParquetValidationError(
            f"play_type: expected {EXPECTED_PLAY_TYPE_FEATURE_COUNT} feature columns, "
            f"got {len(feature_cols)}"
        )


def _validate_yards_gained_parquet(df: pd.DataFrame) -> None:
    if TARGET_REG not in df.columns:
        raise ModelingParquetValidationError(
            f"yards_gained: missing target column {TARGET_REG!r}"
        )
    if TARGET_CLF in df.columns:
        raise ModelingParquetValidationError(
            f"yards_gained: must not include raw {TARGET_CLF!r}"
        )

    generated = list(TASK2_GENERATED_FEATURES)
    missing_generated = [col for col in generated if col not in df.columns]
    if missing_generated:
        raise ModelingParquetValidationError(
            f"yards_gained: missing generated features: {missing_generated}"
        )

    feature_cols = [
        col
        for col in df.columns
        if col not in {TARGET_REG, *generated}
    ]
    if len(feature_cols) != EXPECTED_YARDS_NUMERIC_COUNT:
        raise ModelingParquetValidationError(
            f"yards_gained: expected {EXPECTED_YARDS_NUMERIC_COUNT} numeric columns, "
            f"got {len(feature_cols)}"
        )

    expected_total_cols = (
        EXPECTED_YARDS_NUMERIC_COUNT + len(generated) + 1
    )
    if len(df.columns) != expected_total_cols:
        raise ModelingParquetValidationError(
            f"yards_gained: expected {expected_total_cols} total columns, "
            f"got {len(df.columns)}"
        )


if __name__ == "__main__":
    play_type_df = validate_modeling_parquet(
        PLAY_TYPE_MODELING_PARQUET_PATH, "play_type"
    )
    yards_df = validate_modeling_parquet(
        YARDS_GAINED_MODELING_PARQUET_PATH, "yards_gained"
    )

    print("Modeling schema OK")
    print(
        f"  play_type:     {len(play_type_df):,} rows × {len(play_type_df.columns)} cols "
        f"({EXPECTED_PLAY_TYPE_FEATURE_COUNT} features + {TARGET_CLF})"
    )
    print(
        f"  yards_gained:  {len(yards_df):,} rows × {len(yards_df.columns)} cols "
        f"({EXPECTED_YARDS_NUMERIC_COUNT} numeric + "
        f"{', '.join(TASK2_GENERATED_FEATURES)} + {TARGET_REG})"
    )
    print(f"  artifacts:     {MODELING_ARTIFACTS_DIR}")
    print(f"  experiments:   {EXPERIMENTS_DIR}")
    print(f"  model keys:    {', '.join(MODEL_REGISTRY_KEYS)}")
    print(f"  CV:            N_FOLDS={N_FOLDS}, SEED={SEED}")
