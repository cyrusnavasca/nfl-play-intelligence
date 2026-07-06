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
from src.data.schema import SEED
from src.evaluation.model_selection import select_best_model
from src.models import CLASSIFIER_BUILDERS
from src.pipelines.play_type.train import MODEL_COMPARISON_FILENAME
from src.utils.experiments import (
    get_active_experiment,
    promote_experiment_to_best_model,
    resolve_task_artifacts_dir,
    update_experiment_config,
)
from src.utils.io import save_feature_importance, save_model

__all__ = ["refit_best_classifier"]

METADATA_FILENAME = "metadata.json"


def refit_best_classifier(
    *,
    experiment_id: str | None = None,
    promote: bool = True,
) -> Path:
    """Refit the best classifier on full data; save under experiment and ``best_model/``."""
    exp_id = experiment_id or get_active_experiment("play_type")
    if exp_id is None:
        raise FileNotFoundError(
            "No active play_type experiment. Run training first or pass experiment_id=..."
        )

    artifacts_dir = resolve_task_artifacts_dir("play_type", experiment_id=exp_id)
    comparison_path = artifacts_dir / MODEL_COMPARISON_FILENAME
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

    model_path = save_model(model, "play_type", experiment_id=exp_id)
    best_model_path = save_model(model, "play_type", to_best_model=True)
    save_feature_importance(
        model,
        X.columns.tolist(),
        "play_type",
        experiment_id=exp_id,
    )
    save_feature_importance(
        model,
        X.columns.tolist(),
        "play_type",
        to_best_model=True,
    )

    metadata = {
        "task": "play_type",
        "experiment_id": exp_id,
        "model_key": best_model,
        "seed": SEED,
        "n_rows": len(y),
        "n_features": X.shape[1],
        "model_path": str(model_path.name),
        "best_model_path": str(best_model_path.name),
    }
    for out_dir in (artifacts_dir, best_model_path.parent):
        metadata_path = out_dir / METADATA_FILENAME
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )

    update_experiment_config(
        exp_id,
        {
            "tasks": {
                "play_type": {
                    "best_model": best_model,
                    "refit": metadata,
                }
            }
        },
    )

    if promote:
        promote_experiment_to_best_model(
            "play_type",
            exp_id,
            source_files=[
                MODEL_COMPARISON_FILENAME,
                METADATA_FILENAME,
                "feature_importance.csv",
            ],
        )

    return best_model_path


if __name__ == "__main__":
    path = refit_best_classifier()
    print(f"Best play-type classifier saved → {path}")
