"""
Full-data refit for the best yards-gained regressor.

Persists ``model.joblib``, ``target_transform.joblib``, and ``imputer.joblib``
for downstream inference via :mod:`src.inference.predict`.

Usage (from project root):
    python3 -m src.pipelines.yards_gained.predict
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer

from src.data.loaders import yards_numeric_columns
from src.data.schema import SEED
from src.evaluation.feature_importance import (
    FEATURE_IMPORTANCE_DIRNAME,
    feature_importance_relpath,
    save_feature_importance,
)
from src.evaluation.model_selection import select_best_model
from src.models import REGRESSOR_BUILDERS, hyperparameters_from_experiment_config
from src.pipelines.yards_gained.train import (
    MODEL_COMPARISON_FILENAME,
    TARGET_TRANSFORM_FILENAME,
    build_augmented_yards_frame,
)
from src.utils.experiments import (
    get_active_experiment,
    promote_experiment_to_best_model,
    read_experiment_config,
    resolve_task_artifacts_dir,
    update_experiment_config,
)
from src.utils.io import (
    FEATURE_IMPUTER_FILENAME,
    load_target_transform,
    save_feature_imputer,
    save_model,
    save_target_transform,
)

__all__ = ["refit_best_regressor"]

METADATA_FILENAME = "metadata.json"


def _load_target_transform(*, experiment_id: str):
    artifacts_dir = resolve_task_artifacts_dir("yards_gained", experiment_id=experiment_id)
    transform_path = artifacts_dir / TARGET_TRANSFORM_FILENAME
    if not transform_path.exists():
        raise FileNotFoundError(
            f"Missing target transform artifact: {transform_path}. "
            "Run yards-gained training first."
        )
    return load_target_transform(transform_path)


def refit_best_regressor(
    *,
    experiment_id: str | None = None,
    play_type_experiment_id: str | None = None,
    promote: bool = True,
) -> Path:
    """Refit the best regressor on the full augmented frame; save under experiment and ``best_model/``."""
    exp_id = experiment_id or get_active_experiment("yards_gained")
    if exp_id is None:
        raise FileNotFoundError(
            "No active yards_gained experiment. Run training first or pass experiment_id=..."
        )

    exp_config = read_experiment_config(exp_id)
    play_type_exp = (
        play_type_experiment_id
        or exp_config.get("play_type_experiment")
        or get_active_experiment("play_type")
    )

    artifacts_dir = resolve_task_artifacts_dir("yards_gained", experiment_id=exp_id)
    comparison_path = artifacts_dir / MODEL_COMPARISON_FILENAME
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

    X, y = build_augmented_yards_frame(play_type_experiment_id=play_type_exp)
    impute_cols = yards_numeric_columns(X)
    X_fit = X.copy()
    imputer = SimpleImputer(strategy="median")
    if impute_cols:
        X_fit[impute_cols] = imputer.fit_transform(X_fit[impute_cols])
    else:
        imputer.fit(X_fit.iloc[:, :0])

    target_transform = _load_target_transform(experiment_id=exp_id)
    y_model = target_transform.transform(y)

    hyperparameters = hyperparameters_from_experiment_config(
        exp_config,
        "yards_gained",
        best_model,
    )
    model = REGRESSOR_BUILDERS[best_model](hyperparameters=hyperparameters)
    model.fit(X_fit, y_model)

    model_path = save_model(model, "yards_gained", experiment_id=exp_id)
    best_model_path = save_model(model, "yards_gained", to_best_model=True)
    transform_path = save_target_transform(
        target_transform,
        "yards_gained",
        experiment_id=exp_id,
        filename=TARGET_TRANSFORM_FILENAME,
    )
    best_transform_path = save_target_transform(
        target_transform,
        "yards_gained",
        to_best_model=True,
        filename=TARGET_TRANSFORM_FILENAME,
    )
    imputer_path = save_feature_imputer(
        imputer,
        "yards_gained",
        experiment_id=exp_id,
        filename=FEATURE_IMPUTER_FILENAME,
    )
    best_imputer_path = save_feature_imputer(
        imputer,
        "yards_gained",
        to_best_model=True,
        filename=FEATURE_IMPUTER_FILENAME,
    )
    save_feature_importance(
        model,
        X.columns.tolist(),
        "yards_gained",
        best_model,
        experiment_id=exp_id,
    )
    save_feature_importance(
        model,
        X.columns.tolist(),
        "yards_gained",
        best_model,
        to_best_model=True,
    )

    metadata = {
        "task": "yards_gained",
        "experiment_id": exp_id,
        "play_type_experiment": play_type_exp,
        "model_key": best_model,
        "target_transform": target_transform.name,
        "seed": SEED,
        "n_rows": len(y),
        "n_features": X.shape[1],
        "model_path": str(model_path.name),
        "best_model_path": str(best_model_path.name),
        "target_transform_path": str(transform_path.name),
        "best_target_transform_path": str(best_transform_path.name),
        "imputer_path": str(imputer_path.name),
        "best_imputer_path": str(best_imputer_path.name),
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
            "tasks": {
                "yards_gained": {
                    "best_model": best_model,
                    "refit": metadata,
                }
            }
        },
    )

    if promote:
        promote_experiment_to_best_model(
            "yards_gained",
            exp_id,
            source_files=[
                MODEL_COMPARISON_FILENAME,
                METADATA_FILENAME,
                TARGET_TRANSFORM_FILENAME,
                FEATURE_IMPUTER_FILENAME,
                FEATURE_IMPORTANCE_DIRNAME,
            ],
        )

    return best_model_path


if __name__ == "__main__":
    path = refit_best_regressor()
    print(f"Best yards-gained regressor saved → {path}")
