"""
Full-data refit for the best play-type classifier.

Usage (from project root):
    python3 -m src.pipelines.predict
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.loaders import load_play_type_dataset
from src.data.schema import SEED
from src.evaluation.feature_importance import (
    FEATURE_IMPORTANCE_DIRNAME,
    feature_importance_relpath,
    save_feature_importance,
)
from src.evaluation.model_selection import select_best_model
from src.models import CLASSIFIER_BUILDERS, hyperparameters_from_experiment_config
from src.pipelines.train import MODEL_COMPARISON_FILENAME
from src.utils.experiments import (
    get_active_experiment,
    promote_experiment_to_best_model,
    read_experiment_config,
    resolve_artifacts_dir,
    update_experiment_config,
)
from src.utils.io import save_model

__all__ = ["refit_best_classifier"]

METADATA_FILENAME = "metadata.json"


def refit_best_classifier(
    *,
    experiment_id: str | None = None,
    promote: bool = True,
) -> Path:
    """Refit the best classifier on full data; save under experiment and ``best_model/``."""
    exp_id = experiment_id or get_active_experiment()
    if exp_id is None:
        raise FileNotFoundError(
            "No active experiment. Run training first or pass experiment_id=..."
        )

    artifacts_dir = resolve_artifacts_dir(experiment_id=exp_id)
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
    exp_config = read_experiment_config(exp_id)
    hyperparameters = hyperparameters_from_experiment_config(
        exp_config,
        best_model,
    )
    model = CLASSIFIER_BUILDERS[best_model](hyperparameters=hyperparameters)
    model.fit(X, y)

    model_path = save_model(model, experiment_id=exp_id)
    best_model_path = save_model(model, to_best_model=True)
    save_feature_importance(
        model,
        X.columns.tolist(),
        best_model,
        experiment_id=exp_id,
    )
    save_feature_importance(
        model,
        X.columns.tolist(),
        best_model,
        to_best_model=True,
    )

    metadata = {
        "experiment_id": exp_id,
        "model_key": best_model,
        "seed": SEED,
        "n_rows": len(y),
        "n_features": X.shape[1],
        "model_path": str(model_path.name),
        "best_model_path": str(best_model_path.name),
        "feature_importance": feature_importance_relpath(best_model),
    }
    for out_dir in (artifacts_dir, best_model_path.parent):
        metadata_path = out_dir / METADATA_FILENAME
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )

    update_experiment_config(
        exp_id,
        {
            "best_model": best_model,
            "refit": metadata,
        },
    )

    if promote:
        promote_experiment_to_best_model(
            exp_id,
            source_files=[
                MODEL_COMPARISON_FILENAME,
                METADATA_FILENAME,
                FEATURE_IMPORTANCE_DIRNAME,
            ],
        )

    return best_model_path


if __name__ == "__main__":
    path = refit_best_classifier()
    print(f"Best play-type classifier saved → {path}")
