"""Data contract validation for modeling parquets."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.loaders import load_play_type_dataset, load_yards_gained_dataset
from src.data.schema import (
    DROP_ALWAYS,
    EXPECTED_PLAY_TYPE_FEATURE_COUNT,
    EXPECTED_YARDS_NUMERIC_COUNT,
    PLAY_TYPE_MODELING_PARQUET_PATH,
    TARGET_CLF,
    TARGET_REG,
    TASK2_GENERATED_FEATURES,
    YARDS_GAINED_MODELING_PARQUET_PATH,
    validate_modeling_parquet,
)
from src.selection.shared.feature_schema import (
    PLAY_TYPE_MODELING_PARQUET_PATH as SELECTION_PLAY_TYPE_PATH,
    YARDS_GAINED_MODELING_PARQUET_PATH as SELECTION_YARDS_PATH,
)


@pytest.mark.parametrize(
    ("schema_path", "selection_path"),
    [
        (PLAY_TYPE_MODELING_PARQUET_PATH, SELECTION_PLAY_TYPE_PATH),
        (YARDS_GAINED_MODELING_PARQUET_PATH, SELECTION_YARDS_PATH),
    ],
)
def test_parquet_paths_match_selection_schema(schema_path, selection_path) -> None:
    assert schema_path == selection_path
    assert schema_path.exists(), f"missing parquet: {schema_path}"


def test_play_type_feature_count() -> None:
    X, y = load_play_type_dataset()
    assert len(X.columns) == EXPECTED_PLAY_TYPE_FEATURE_COUNT
    assert y.name == TARGET_CLF
    assert TARGET_REG not in X.columns


def test_yards_gained_feature_count() -> None:
    X, y = load_yards_gained_dataset()
    assert len(X.columns) == EXPECTED_YARDS_NUMERIC_COUNT + len(TASK2_GENERATED_FEATURES)
    assert set(TASK2_GENERATED_FEATURES) <= set(X.columns)
    assert TARGET_CLF not in X.columns
    assert y.name == TARGET_REG


def test_modeling_parquets_share_row_count() -> None:
    play_df = validate_modeling_parquet(PLAY_TYPE_MODELING_PARQUET_PATH, "play_type")
    yards_df = validate_modeling_parquet(YARDS_GAINED_MODELING_PARQUET_PATH, "yards_gained")
    assert len(play_df) == len(yards_df)


def test_drop_always_columns_absent() -> None:
    for task, path in (
        ("play_type", PLAY_TYPE_MODELING_PARQUET_PATH),
        ("yards_gained", YARDS_GAINED_MODELING_PARQUET_PATH),
    ):
        df = validate_modeling_parquet(path, task)
        leaked = set(df.columns) & set(DROP_ALWAYS)
        assert not leaked, f"{task}: DROP_ALWAYS columns present: {sorted(leaked)}"


def test_validate_rejects_drop_always_columns(tmp_path) -> None:
    df = pd.read_parquet(PLAY_TYPE_MODELING_PARQUET_PATH)
    df["epa"] = 0.0
    bad_path = tmp_path / "bad_play_type.parquet"
    df.to_parquet(bad_path, index=False)
    with pytest.raises(Exception, match="DROP_ALWAYS"):
        validate_modeling_parquet(bad_path, "play_type")
