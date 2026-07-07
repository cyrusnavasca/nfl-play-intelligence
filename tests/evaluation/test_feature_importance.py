"""Feature importance artifact helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.feature_importance import (
    FEATURE_IMPORTANCE_DIRNAME,
    enrich_models_snapshot,
    feature_importance_relpath,
    save_feature_importance,
    to_feature_importance_frame,
)
from src.models.config import snapshot_task_hyperparameters


def test_feature_importance_relpath() -> None:
    assert feature_importance_relpath("xgboost") == "feature_importance/xgboost.csv"


def test_to_feature_importance_frame_sorted() -> None:
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = np.random.default_rng(42).random((50, 3))
    y = (X[:, 0] > 0.5).astype(int)
    model.fit(X, y)

    frame = to_feature_importance_frame(model, ["a", "b", "c"])
    assert frame is not None
    assert list(frame.columns) == ["feature", "importance"]
    assert frame["importance"].is_monotonic_decreasing


def test_save_feature_importance_writes_per_model_file(
    experiment_dirs,
) -> None:
    from src.utils.experiments import allocate_experiment_id

    exp_id = allocate_experiment_id()
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = np.random.default_rng(0).random((40, 2))
    y = (X[:, 0] > 0.5).astype(int)
    model.fit(X, y)

    out_path = save_feature_importance(
        model,
        ["x1", "x2"],
        "play_type",
        "random_forest",
        experiment_id=exp_id,
    )
    assert out_path is not None
    assert out_path == (
        experiment_dirs
        / "experiments"
        / exp_id
        / "play_type"
        / FEATURE_IMPORTANCE_DIRNAME
        / "random_forest.csv"
    )
    assert out_path.exists()


def test_enrich_models_snapshot_adds_paths(experiment_dirs) -> None:
    from src.utils.experiments import allocate_experiment_id, task_experiment_dir

    exp_id = allocate_experiment_id()
    fi_dir = task_experiment_dir(exp_id, "play_type") / FEATURE_IMPORTANCE_DIRNAME
    fi_dir.mkdir(parents=True)
    (fi_dir / "xgboost.csv").write_text("feature,importance\na,1.0\n")

    models = snapshot_task_hyperparameters("play_type")
    enriched = enrich_models_snapshot("play_type", exp_id, models)

    assert "feature_importance" not in enriched["baseline"]
    assert enriched["xgboost"]["feature_importance"] == "feature_importance/xgboost.csv"


def test_build_task_result_config_includes_feature_importance_paths(
    experiment_dirs,
) -> None:
    import pandas as pd

    from src.utils.experiments import (
        allocate_experiment_id,
        build_task_result_config,
        task_experiment_dir,
    )

    exp_id = allocate_experiment_id()
    fi_dir = task_experiment_dir(exp_id, "yards_gained") / FEATURE_IMPORTANCE_DIRNAME
    fi_dir.mkdir(parents=True)
    (fi_dir / "random_forest.csv").write_text("feature,importance\na,1.0\n")

    comparison = pd.DataFrame(
        {
            "model": ["baseline", "random_forest"],
            "rmse_mean": [10.0, 8.0],
            "rmse_std": [0.0, 0.0],
        }
    )
    task_cfg = build_task_result_config(
        "yards_gained",
        comparison,
        metric="rmse",
        higher_is_better=False,
        experiment_id=exp_id,
    )
    assert task_cfg["best_model"] == "random_forest"
    assert (
        task_cfg["models"]["random_forest"]["feature_importance"]
        == "feature_importance/random_forest.csv"
    )
