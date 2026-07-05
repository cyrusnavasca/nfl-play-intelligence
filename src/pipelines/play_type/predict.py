"""
Full-data refit for the best play-type classifier (inference stub).

Usage (from project root):
    python3 -m src.pipelines.play_type.predict
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.loaders import load_play_type_dataset
from src.data.schema import PLAY_TYPE_ARTIFACTS_DIR, SEED
from src.evaluation.model_selection import select_best_model
from src.models import CLASSIFIER_BUILDERS
from src.pipelines.play_type.train import MODEL_COMPARISON_FILENAME
from src.utils.io import ensure_artifacts_dir, save_model

__all__ = ["refit_best_classifier"]

BEST_MODEL_DIR = "best_model"
METADATA_FILENAME = "metadata.json"


def refit_best_classifier() -> Path:
    """Refit the best classifier on full data; save to ``best_model/``."""
    comparison_path = PLAY_TYPE_ARTIFACTS_DIR / MODEL_COMPARISON_FILENAME
    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Run play-type training first: {comparison_path} not found"
        )

    comparison = pd.read_csv(comparison_path)
    best_model = select_best_model(
        comparison,
        "roc_auc",
        higher_is_better=True,
    )

    X, y = load_play_type_dataset()
    model = CLASSIFIER_BUILDERS[best_model]()
    model.fit(X, y)

    out_dir = ensure_artifacts_dir("play_type") / BEST_MODEL_DIR
    model_path = save_model(model, "play_type", subdir=BEST_MODEL_DIR)

    metadata = {
        "task": "play_type",
        "model_key": best_model,
        "seed": SEED,
        "n_rows": len(y),
        "n_features": X.shape[1],
        "model_path": str(model_path.relative_to(PLAY_TYPE_ARTIFACTS_DIR)),
    }
    metadata_path = out_dir / METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    return model_path


if __name__ == "__main__":
    path = refit_best_classifier()
    print(f"Best play-type classifier saved → {path}")
