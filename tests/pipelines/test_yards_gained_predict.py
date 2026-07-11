"""Yards-gained refit pipeline and artifact tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inference.predict import predict_yards
from src.pipelines.yards_gained.predict import (
    METADATA_FILENAME,
    TARGET_TRANSFORM_FILENAME,
    refit_best_regressor,
)
from src.utils.io import FEATURE_IMPUTER_FILENAME
from src.pipelines.yards_gained.train import (
    HOLDOUT_RESULTS_FILENAME,
    MODEL_COMPARISON_FILENAME,
    train_yards_gained,
)
from src.utils.experiments import task_experiment_dir


def test_refit_writes_model_transform_and_imputer(
    yards_subsample,
    experiment_dirs,
    monkeypatch,
) -> None:
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
    monkeypatch.setattr(
        "src.pipelines.yards_gained.predict.build_augmented_yards_frame",
        lambda **_: (X_raw.assign(pred_pass_proba=oof_proba), y),
    )

    _, exp_id = train_yards_gained(experiment_id="exp_test")
    best_model_path = refit_best_regressor(experiment_id=exp_id, promote=False)

    artifacts_dir = task_experiment_dir(exp_id, "yards_gained")
    assert (artifacts_dir / TARGET_TRANSFORM_FILENAME).exists()
    assert (artifacts_dir / FEATURE_IMPUTER_FILENAME).exists()
    assert (artifacts_dir / METADATA_FILENAME).exists()
    assert best_model_path.exists()

    best_dir = best_model_path.parent
    assert (best_dir / TARGET_TRANSFORM_FILENAME).exists()
    assert (best_dir / FEATURE_IMPUTER_FILENAME).exists()


def test_refit_requires_saved_target_transform(
    yards_subsample,
    experiment_dirs,
    monkeypatch,
) -> None:
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

    _, exp_id = train_yards_gained(experiment_id="exp_no_transform")
    artifacts_dir = task_experiment_dir(exp_id, "yards_gained")
    (artifacts_dir / TARGET_TRANSFORM_FILENAME).unlink()

    with pytest.raises(FileNotFoundError, match="Missing target transform"):
        refit_best_regressor(experiment_id=exp_id, promote=False)


def test_refit_artifacts_support_predict_yards(
    yards_subsample,
    experiment_dirs,
    monkeypatch,
) -> None:
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
    monkeypatch.setattr(
        "src.pipelines.yards_gained.predict.build_augmented_yards_frame",
        lambda **_: (X_raw.assign(pred_pass_proba=oof_proba), y),
    )

    _, exp_id = train_yards_gained(experiment_id="exp_predict")
    best_model_path = refit_best_regressor(experiment_id=exp_id, promote=False)
    best_dir = best_model_path.parent

    X_aug = X_raw.copy()
    X_aug["pred_pass_proba"] = oof_proba.values
    preds = predict_yards(
        X_aug,
        best_dir / "model.joblib",
        best_dir / TARGET_TRANSFORM_FILENAME,
        imputer_path=best_dir / FEATURE_IMPUTER_FILENAME,
    )

    assert preds.shape == (len(y),)
    assert np.isfinite(preds).all()
