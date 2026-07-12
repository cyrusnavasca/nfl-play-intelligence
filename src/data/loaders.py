"""
Modeling dataset loaders.

Loads validated modeling parquets and splits features from targets.
"""
from __future__ import annotations

import pandas as pd

from src.data.schema import (
    PLAY_TYPE_MODELING_PARQUET_PATH,
    TARGET_CLF,
    validate_modeling_parquet,
)

__all__ = [
    "load_play_type_dataset",
]


def load_play_type_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load play-type modeling parquet; return ``(X, y)`` with binary target."""
    df = validate_modeling_parquet(PLAY_TYPE_MODELING_PARQUET_PATH)
    y = df[TARGET_CLF]
    X = df.drop(columns=[TARGET_CLF])
    return X, y
