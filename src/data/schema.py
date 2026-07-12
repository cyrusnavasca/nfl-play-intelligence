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

import json

from src.selection.shared.feature_schema import (
    DROP_ALWAYS,
    FEATURE_SELECTION_MANIFEST_PATH,
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


def _load_manifest_final(path=FEATURE_SELECTION_MANIFEST_PATH) -> dict | None:
    """Return the manifest's ``task1.final`` lists, or None if unavailable."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        return manifest["task1"]["final"]
    except (KeyError, ValueError, OSError):
        return None


def validate_modeling_parquet(path: Path | str) -> pd.DataFrame:
    """
    Load and validate the play-type modeling parquet.

    Checks:
      - row count > 0
      - target column present
      - at least one feature column
      - no ``DROP_ALWAYS`` columns in the frame
      - if the selection manifest is present, every final numeric feature is a
        column and every final categorical has at least one one-hot column
        (the exact feature count is set by the manual feature config, not fixed)
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
    if not feature_cols:
        raise ModelingParquetValidationError("parquet has no feature columns")

    final = _load_manifest_final()
    if final is not None:
        cols = set(feature_cols)
        missing_numeric = [c for c in final["numeric"] if c not in cols]
        if missing_numeric:
            raise ModelingParquetValidationError(
                f"final numeric features missing from parquet: {missing_numeric}"
            )
        missing_cat = [
            cat
            for cat in final["categorical"]
            if not any(c.startswith(f"{cat}_") for c in cols)
        ]
        if missing_cat:
            raise ModelingParquetValidationError(
                f"final categorical features have no one-hot columns: {missing_cat}"
            )

    return df


if __name__ == "__main__":
    df = validate_modeling_parquet(PLAY_TYPE_MODELING_PARQUET_PATH)

    n_features = len(df.columns) - 1
    print("Modeling schema OK")
    print(
        f"  play_type:     {len(df):,} rows × {len(df.columns)} cols "
        f"({n_features} features + {TARGET_CLF})"
    )
    print(f"  artifacts:     {MODELING_ARTIFACTS_DIR}")
    print(f"  experiments:   {EXPERIMENTS_DIR}")
    print(f"  model keys:    {', '.join(MODEL_REGISTRY_KEYS)}")
    print(f"  CV:            N_FOLDS={N_FOLDS}, SEED={SEED}")
