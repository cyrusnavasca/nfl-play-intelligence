"""
Full-data refit for the best yards-gained regressor (inference stub).

Usage (from project root):
    python3 -m src.pipelines.yards_gained.predict
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer

from src.data.loaders import yards_numeric_columns
from src.data.schema import SEED, YARDS_GAINED_ARTIFACTS_DIR
from src.evaluation.model_selection import select_best_model
from src.models import REGRESSOR_BUILDERS
from src.pipelines.yards_gained.train import (
    MODEL_COMPARISON_FILENAME,
    build_augmented_yards_frame,
)
from src.utils.io import ensure_artifacts_dir, save_model

__all__ = ["refit_best_regressor"]

BEST_MODEL_DIR = "best_model"
METADATA_FILENAME = "metadata.json"


def refit_best_regressor() -> Path:
    """Refit the best regressor on the full augmented frame; save to ``best_model/``."""
    comparison_path = YARDS_GAINED_ARTIFACTS_DIR / MODEL_COMPARISON_FILENAME
    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Run yards-gained training first: {comparison_path} not found"
        )

    comparison = pd.read_csv(comparison_path)
    best_model = select_best_model(
        comparison,
        "rmse",
        higher_is_better=False,
    )

    X, y = build_augmented_yards_frame()
    impute_cols = yards_numeric_columns(X)
    X_fit = X.copy()
    if impute_cols:
        X_fit[impute_cols] = SimpleImputer(strategy="median").fit_transform(
            X_fit[impute_cols]
        )

    model = REGRESSOR_BUILDERS[best_model]()
    model.fit(X_fit, y)

    out_dir = ensure_artifacts_dir("yards_gained") / BEST_MODEL_DIR
    model_path = save_model(model, "yards_gained", subdir=BEST_MODEL_DIR)

    metadata = {
        "task": "yards_gained",
        "model_key": best_model,
        "seed": SEED,
        "n_rows": len(y),
        "n_features": X.shape[1],
        "model_path": str(model_path.relative_to(YARDS_GAINED_ARTIFACTS_DIR)),
    }
    metadata_path = out_dir / METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    return model_path


if __name__ == "__main__":
    path = refit_best_regressor()
    print(f"Best yards-gained regressor saved → {path}")
