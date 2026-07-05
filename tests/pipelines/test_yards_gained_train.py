"""Yards-gained holdout pipeline smoke tests."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loaders import yards_numeric_columns
from src.data.schema import TASK2_GENERATED_FEATURES
from src.models import REGRESSOR_BUILDERS
from src.pipelines.yards_gained import train as yards_train
from src.pipelines.yards_gained.train import (
    build_augmented_yards_frame,
    run_yards_gained_holdout,
)


def test_yards_train_has_no_cv_fold_loop() -> None:
    source = inspect.getsource(yards_train)
    assert "stratified_folds" not in source
    assert "cross_validate_regressor" not in source
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("for ") and "fold" in stripped:
            raise AssertionError(f"unexpected fold loop in yards_gained/train.py: {stripped}")


def test_oof_proba_appended_before_split(yards_subsample, monkeypatch) -> None:
    X_raw, y = yards_subsample
    numeric_cols = yards_numeric_columns(X_raw)
    oof_proba = pd.Series(np.linspace(0.1, 0.9, len(y)))

    monkeypatch.setattr(
        "src.pipelines.yards_gained.train.load_yards_gained_dataset",
        lambda: (X_raw, y),
    )
    monkeypatch.setattr(
        "src.pipelines.yards_gained.train.get_best_oof_proba",
        lambda: oof_proba,
    )

    X, y_out = build_augmented_yards_frame()
    proba_col = TASK2_GENERATED_FEATURES[0]

    assert proba_col in X.columns
    assert len(X.columns) == len(numeric_cols) + 1
    pd.testing.assert_series_equal(X[proba_col], oof_proba, check_names=False)
    pd.testing.assert_series_equal(y_out, y)


def test_holdout_smoke_on_subsample(yards_subsample, monkeypatch) -> None:
    X_raw, y = yards_subsample
    oof_proba = pd.Series(np.linspace(0.2, 0.8, len(y)))

    monkeypatch.setattr(
        "src.pipelines.yards_gained.train.load_yards_gained_dataset",
        lambda: (X_raw, y),
    )
    monkeypatch.setattr(
        "src.pipelines.yards_gained.train.get_best_oof_proba",
        lambda: oof_proba,
    )

    records, comparison, holdout_df, pred_frame = run_yards_gained_holdout()

    assert len(comparison) == len(REGRESSOR_BUILDERS)
    assert len(holdout_df) == len(REGRESSOR_BUILDERS)
    assert len(pred_frame) == int(len(y) * 0.2)
    assert pred_frame.isna().sum().sum() == 0

    baseline_rmse = holdout_df.loc[
        holdout_df["model"] == "baseline", "rmse"
    ].iloc[0]
    yards_std = float(y.std(ddof=0))
    assert abs(baseline_rmse - yards_std) < 1.0

    for record in records:
        assert record["model"] in REGRESSOR_BUILDERS


def test_oof_parquet_has_zero_nans_when_present() -> None:
    oof_path = Path("artifacts/modeling/play_type/oof_predictions.parquet")
    if not oof_path.exists():
        return
    oof_df = pd.read_parquet(oof_path)
    assert oof_df.isna().sum().sum() == 0
    assert len(oof_df) == 276_286
