"""Target transform unit tests."""
from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.target_transform import (
    TARGET_TRANSFORM_KEYS,
    build_target_transform,
)


@pytest.fixture
def yards_sample() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(loc=5.5, scale=8.6, size=500)


@pytest.mark.parametrize("name", TARGET_TRANSFORM_KEYS)
def test_roundtrip_preserves_yards(name: str, yards_sample: np.ndarray) -> None:
    transform = build_target_transform(name)
    y_model = transform.fit_transform(yards_sample)
    y_back = transform.inverse_transform(y_model)
    np.testing.assert_allclose(y_back, yards_sample, rtol=1e-5, atol=1e-4)


def test_log_handles_negative_yards() -> None:
    y = np.array([-34.0, -7.0, 0.0, 5.0, 45.0, 99.0])
    transform = build_target_transform("log")
    y_model = transform.fit_transform(y)
    y_back = transform.inverse_transform(y_model)
    np.testing.assert_allclose(y_back, y, rtol=1e-6, atol=1e-5)


def test_signed_log_handles_negative_yards() -> None:
    y = np.array([-34.0, -7.0, 0.0, 5.0, 45.0, 99.0])
    transform = build_target_transform("signed_log")
    y_model = transform.fit_transform(y)
    y_back = transform.inverse_transform(y_model)
    np.testing.assert_allclose(y_back, y, rtol=1e-6, atol=1e-5)


def test_build_rejects_unknown_transform() -> None:
    with pytest.raises(ValueError, match="unknown target transform"):
        build_target_transform("sqrt")
