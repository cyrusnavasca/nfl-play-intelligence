"""Train/test split validation."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.schema import SEED, TARGET_REG
from src.preprocessing.split_data import save_split_indices, train_test_split_frame


def test_train_test_split_is_deterministic(yards_subsample) -> None:
    X, y = yards_subsample
    frame = X.copy()
    frame[TARGET_REG] = y

    split_a = train_test_split_frame(frame, TARGET_REG, seed=SEED)
    split_b = train_test_split_frame(frame, TARGET_REG, seed=SEED)

    X_train_a, X_test_a, y_train_a, y_test_a = split_a
    X_train_b, X_test_b, y_train_b, y_test_b = split_b
    pd.testing.assert_frame_equal(X_train_a, X_train_b)
    pd.testing.assert_frame_equal(X_test_a, X_test_b)
    pd.testing.assert_series_equal(y_train_a, y_train_b)
    pd.testing.assert_series_equal(y_test_a, y_test_b)


def test_train_test_split_80_20_proportions(yards_subsample) -> None:
    X, y = yards_subsample
    frame = X.copy()
    frame[TARGET_REG] = y

    X_train, X_test, y_train, y_test = train_test_split_frame(frame, TARGET_REG)
    n = len(frame)
    assert len(y_train) == int(n * 0.8)
    assert len(y_test) == n - len(y_train)
    assert X_train.shape[1] == X.shape[1]


def test_save_split_indices_roundtrip(tmp_path, yards_subsample) -> None:
    X, y = yards_subsample
    frame = X.copy()
    frame[TARGET_REG] = y

    indices = np.arange(len(frame))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=SEED, shuffle=True
    )

    out_path = save_split_indices(train_idx, test_idx, tmp_path / "split.json")
    payload = json.loads(out_path.read_text())
    assert payload["n_train"] == len(train_idx)
    assert payload["n_test"] == len(test_idx)
    assert payload["train_idx"][0] == int(train_idx[0])
