"""
Modeling schema and data contract.

Single source of truth for the modeling parquet path, target, artifact
locations, and model registry keys. Downstream modeling modules must import
column-related constants from here — no hardcoded column names elsewhere.

Reference: ``docs/modeling_plan.md`` Phase 1 (migrated to ``src/data/`` in Phase 2).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.selection.shared.feature_schema import (
    DROP_ALWAYS,
    PLAY_TYPE_MODELING_PARQUET_PATH,
    TARGET_CLF,
    SEED,
)

# Matches ``src.selection.shared.common.N_FOLDS`` (defined there to avoid
# importing lightgbm when loading schema-only modules).
N_FOLDS = 5

# Re-export selection constants used across modeling.
__all__ = [
    "DROP_ALWAYS",
    "EXPECTED_FEATURE_COUNT",
    "BEST_MODEL_DIR",
    "EXPERIMENTS_DIR",
    "MODELING_ARTIFACTS_DIR",
    "MODEL_REGISTRY_KEYS",
    "ModelingParquetValidationError",
    "N_FOLDS",
    "PLAY_TYPE_MODELING_PARQUET_PATH",
    "SEED",
    "TARGET_CLF",
    "validate_modeling_parquet",
]

# ---------------------------------------------------------------------------
# Modeling parquet column count (from feature_selection_manifest.json final)
# ---------------------------------------------------------------------------

EXPECTED_FEATURE_COUNT = 34

# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------

MODELING_ARTIFACTS_DIR = Path("artifacts/modeling")
EXPERIMENTS_DIR = MODELING_ARTIFACTS_DIR / "experiments"
BEST_MODEL_DIR = MODELING_ARTIFACTS_DIR / "best_model"

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY_KEYS: tuple[str, ...] = ("baseline", "random_forest", "xgboost")


class ModelingParquetValidationError(ValueError):
    """Raised when the modeling parquet does not match the data contract."""


def validate_modeling_parquet(path: Path | str) -> pd.DataFrame:
    """
    Load and validate the play-type modeling parquet.

    Checks:
      - row count > 0
      - target column present
      - no ``DROP_ALWAYS`` columns in the frame
      - expected feature count
    """
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise ModelingParquetValidationError(f"parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    if len(df) == 0:
        raise ModelingParquetValidationError("parquet has zero rows")

    leaked = sorted(set(df.columns) & set(DROP_ALWAYS))
    if leaked:
        raise ModelingParquetValidationError(f"DROP_ALWAYS columns present: {leaked}")

    if TARGET_CLF not in df.columns:
        raise ModelingParquetValidationError(f"missing target column {TARGET_CLF!r}")

    feature_cols = [col for col in df.columns if col != TARGET_CLF]
    if len(feature_cols) != EXPECTED_FEATURE_COUNT:
        raise ModelingParquetValidationError(
            f"expected {EXPECTED_FEATURE_COUNT} feature columns, got {len(feature_cols)}"
        )

    return df


if __name__ == "__main__":
    df = validate_modeling_parquet(PLAY_TYPE_MODELING_PARQUET_PATH)

    print("Modeling schema OK")
    print(
        f"  play_type:     {len(df):,} rows × {len(df.columns)} cols "
        f"({EXPECTED_FEATURE_COUNT} features + {TARGET_CLF})"
    )
    print(f"  artifacts:     {MODELING_ARTIFACTS_DIR}")
    print(f"  experiments:   {EXPERIMENTS_DIR}")
    print(f"  model keys:    {', '.join(MODEL_REGISTRY_KEYS)}")
    print(f"  CV:            N_FOLDS={N_FOLDS}, SEED={SEED}")
