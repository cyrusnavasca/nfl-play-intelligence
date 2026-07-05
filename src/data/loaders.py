"""
Modeling dataset loaders.

Loads validated modeling parquets and splits features from targets.
"""
from __future__ import annotations

import pandas as pd

from src.data.schema import (
    PLAY_TYPE_MODELING_PARQUET_PATH,
    TARGET_CLF,
    TARGET_REG,
    TASK2_GENERATED_FEATURES,
    YARDS_GAINED_MODELING_PARQUET_PATH,
    validate_modeling_parquet,
)

__all__ = [
    "load_play_type_dataset",
    "load_yards_gained_dataset",
    "yards_numeric_columns",
]


def load_play_type_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load play-type modeling parquet; return ``(X, y)`` with binary target."""
    df = validate_modeling_parquet(PLAY_TYPE_MODELING_PARQUET_PATH, "play_type")
    y = df[TARGET_CLF]
    X = df.drop(columns=[TARGET_CLF])
    return X, y


def load_yards_gained_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load yards-gained modeling parquet; return ``(X, y)``."""
    df = validate_modeling_parquet(YARDS_GAINED_MODELING_PARQUET_PATH, "yards_gained")
    y = df[TARGET_REG]
    X = df.drop(columns=[TARGET_REG])
    return X, y


def yards_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns excluding target and ``pred_pass_proba``."""
    generated = set(TASK2_GENERATED_FEATURES)
    return [
        col
        for col in df.columns
        if col not in generated and col != TARGET_REG
    ]
