"""
Yards-gained inference: load artifacts, preprocess features, return yards.

Usage::

    from src.inference.predict import predict_yards

    yards = predict_yards(
        X,
        model_path="artifacts/modeling/best_model/yards_gained/model.joblib",
        transformer_path="artifacts/modeling/best_model/yards_gained/target_transform.joblib",
        imputer_path="artifacts/modeling/best_model/yards_gained/imputer.joblib",
    )
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loaders import yards_numeric_columns
from src.preprocessing.impute import apply_median_imputer
from src.utils.io import (
    FEATURE_IMPUTER_FILENAME,
    MODEL_FILENAME,
    TARGET_TRANSFORM_FILENAME,
    load_feature_imputer,
    load_model,
    load_target_transform,
)

__all__ = [
    "predict_yards",
    "predict_yards_from_dir",
]


def predict_yards(
    X: pd.DataFrame,
    model_path: Path | str,
    transformer_path: Path | str,
    *,
    imputer_path: Path | str | None = None,
) -> np.ndarray:
    """
    Predict yards gained from raw features.

    Loads the trained regressor and fitted target transformer, optionally
    median-imputes numeric columns, predicts in transformed space, and
    inverse-transforms back to yards.
    """
    model = load_model(model_path)
    transform = load_target_transform(transformer_path)

    X_model = X
    if imputer_path is not None:
        imputer = load_feature_imputer(imputer_path)
        X_model = apply_median_imputer(X, imputer, yards_numeric_columns(X))

    y_pred_model = model.predict(X_model)
    return np.asarray(transform.inverse_transform(y_pred_model), dtype=float)


def predict_yards_from_dir(
    X: pd.DataFrame,
    artifacts_dir: Path | str,
    *,
    model_filename: str = MODEL_FILENAME,
    transformer_filename: str = TARGET_TRANSFORM_FILENAME,
    imputer_filename: str = FEATURE_IMPUTER_FILENAME,
) -> np.ndarray:
    """Predict yards using standard artifact filenames under *artifacts_dir*."""
    root = Path(artifacts_dir)
    imputer_path = root / imputer_filename
    return predict_yards(
        X,
        root / model_filename,
        root / transformer_filename,
        imputer_path=imputer_path if imputer_path.exists() else None,
    )
