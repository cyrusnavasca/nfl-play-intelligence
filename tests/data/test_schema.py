"""Data contract validation for the play-type modeling parquet."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.loaders import load_play_type_dataset
from src.data.schema import (
    DROP_ALWAYS,
    EXPECTED_FEATURE_COUNT,
    PLAY_TYPE_MODELING_PARQUET_PATH,
    TARGET_CLF,
    validate_modeling_parquet,
)
from src.selection.shared.feature_schema import (
    PLAY_TYPE_MODELING_PARQUET_PATH as SELECTION_PLAY_TYPE_PATH,
)


def test_parquet_path_matches_selection_schema() -> None:
    assert PLAY_TYPE_MODELING_PARQUET_PATH == SELECTION_PLAY_TYPE_PATH
    assert PLAY_TYPE_MODELING_PARQUET_PATH.exists(), (
        f"missing parquet: {PLAY_TYPE_MODELING_PARQUET_PATH}"
    )


def test_play_type_feature_count() -> None:
    X, y = load_play_type_dataset()
    assert len(X.columns) == EXPECTED_FEATURE_COUNT
    assert y.name == TARGET_CLF
    assert "yards_gained" not in X.columns


def test_drop_always_columns_absent() -> None:
    df = validate_modeling_parquet(PLAY_TYPE_MODELING_PARQUET_PATH)
    leaked = set(df.columns) & set(DROP_ALWAYS)
    assert not leaked, f"DROP_ALWAYS columns present: {sorted(leaked)}"


def test_validate_rejects_drop_always_columns(tmp_path) -> None:
    df = pd.read_parquet(PLAY_TYPE_MODELING_PARQUET_PATH)
    df["epa"] = 0.0
    bad_path = tmp_path / "bad_play_type.parquet"
    df.to_parquet(bad_path, index=False)
    with pytest.raises(Exception, match="DROP_ALWAYS"):
        validate_modeling_parquet(bad_path)
