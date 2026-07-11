"""Yards-gained inference wrapper tests."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from src.data.loaders import yards_numeric_columns
from src.inference.predict import predict_yards, predict_yards_from_dir
from src.models import REGRESSOR_BUILDERS
from src.preprocessing.target_transform import build_target_transform


@pytest.fixture
def yards_inference_artifacts(tmp_path, yards_subsample):
    X_raw, y = yards_subsample
    numeric_cols = yards_numeric_columns(X_raw)
    oof_proba = pd.Series(np.linspace(0.2, 0.8, len(y)), name="pred_pass_proba")
    X = X_raw[numeric_cols].copy()
    X["pred_pass_proba"] = oof_proba.values

    transform = build_target_transform("log")
    y_model = transform.fit_transform(y)

    impute_cols = yards_numeric_columns(X)
    imputer = SimpleImputer(strategy="median")
    X_fit = X.copy()
    X_fit[impute_cols] = imputer.fit_transform(X_fit[impute_cols])

    model = REGRESSOR_BUILDERS["baseline"]()
    model.fit(X_fit, y_model)

    artifact_dir = tmp_path / "yards_gained"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "model.joblib"
    transform_path = artifact_dir / "target_transform.joblib"
    imputer_path = artifact_dir / "imputer.joblib"

    joblib.dump(model, model_path)
    joblib.dump(transform, transform_path)
    joblib.dump(imputer, imputer_path)

    return {
        "X_raw": X,
        "y": y,
        "model_path": model_path,
        "transform_path": transform_path,
        "imputer_path": imputer_path,
        "artifact_dir": artifact_dir,
    }


def test_predict_yards_returns_yards_in_original_space(yards_inference_artifacts) -> None:
    artifacts = yards_inference_artifacts
    preds = predict_yards(
        artifacts["X_raw"],
        artifacts["model_path"],
        artifacts["transform_path"],
        imputer_path=artifacts["imputer_path"],
    )

    assert preds.shape == (len(artifacts["y"]),)
    assert np.isfinite(preds).all()


def test_predict_yards_from_dir(yards_inference_artifacts) -> None:
    artifacts = yards_inference_artifacts
    preds = predict_yards_from_dir(artifacts["X_raw"], artifacts["artifact_dir"])

    assert preds.shape == (len(artifacts["y"]),)
    assert np.isfinite(preds).all()


def test_predict_yards_matches_manual_flow(yards_inference_artifacts) -> None:
    artifacts = yards_inference_artifacts

    model = joblib.load(artifacts["model_path"])
    transform = joblib.load(artifacts["transform_path"])
    imputer = joblib.load(artifacts["imputer_path"])

    X = artifacts["X_raw"].copy()
    impute_cols = yards_numeric_columns(X)
    X[impute_cols] = imputer.transform(X[impute_cols])
    expected = transform.inverse_transform(model.predict(X))

    preds = predict_yards(
        artifacts["X_raw"],
        artifacts["model_path"],
        artifacts["transform_path"],
        imputer_path=artifacts["imputer_path"],
    )
    np.testing.assert_allclose(preds, expected, rtol=1e-6, atol=1e-6)
