"""Shared fixtures for modeling validation tests."""
from __future__ import annotations

import pytest

from src.data.loaders import load_play_type_dataset, load_yards_gained_dataset

SUBSAMPLE_ROWS = 1_000


@pytest.fixture(scope="session")
def play_type_subsample() -> tuple:
    """First 1k rows of the play-type modeling frame."""
    X, y = load_play_type_dataset()
    return (
        X.iloc[:SUBSAMPLE_ROWS].reset_index(drop=True),
        y.iloc[:SUBSAMPLE_ROWS].reset_index(drop=True),
    )


@pytest.fixture(scope="session")
def yards_subsample() -> tuple:
    """First 1k rows of the yards-gained modeling frame."""
    X, y = load_yards_gained_dataset()
    return (
        X.iloc[:SUBSAMPLE_ROWS].reset_index(drop=True),
        y.iloc[:SUBSAMPLE_ROWS].reset_index(drop=True),
    )
