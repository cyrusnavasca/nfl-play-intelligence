"""Yards-gained holdout with Yeo-Johnson target transform."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipelines.yards_gained.train import run_yards_gained_holdout
from src.utils.experiment_profile import (
    ExperimentProfile,
    TaskProfile,
    use_experiment_profile,
)


def test_holdout_with_yeo_johnson_transform(yards_subsample, monkeypatch) -> None:
    X_raw, y = yards_subsample
    oof_proba = pd.Series(np.linspace(0.2, 0.8, len(y)))

    monkeypatch.setattr(
        "src.pipelines.yards_gained.train.load_yards_gained_dataset",
        lambda: (X_raw, y),
    )
    monkeypatch.setattr(
        "src.pipelines.yards_gained.train.get_best_oof_proba",
        lambda **_: oof_proba,
    )

    profile = ExperimentProfile(
        name="yeo_test",
        description=None,
        seed=42,
        n_folds=5,
        persist_best=False,
        play_type_experiment=None,
        tasks={
            "yards_gained": TaskProfile(
                models={
                    "baseline": {"strategy": "mean"},
                    "xgboost": {
                        "n_estimators": 50,
                        "max_depth": 4,
                        "learning_rate": 0.1,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "random_state": 42,
                        "verbosity": 0,
                    },
                },
                target_transform="yeo_johnson",
            )
        },
    )

    with use_experiment_profile(profile):
        _, comparison, holdout_df, pred_frame, transform = run_yards_gained_holdout()

    assert transform.name == "yeo_johnson"
    assert len(comparison) == 2
    assert pred_frame.isna().sum().sum() == 0
    assert holdout_df["rmse"].notna().all()
