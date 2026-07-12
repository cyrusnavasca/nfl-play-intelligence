"""Feature importance artifact helpers."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.feature_importance import (
    FEATURE_IMPORTANCE_DIRNAME,
    enrich_models_snapshot,
    feature_importance_relpath,
    save_feature_importance,
    to_feature_importance_frame,
)
from src.models.config import snapshot_hyperparameters


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
        "random_forest",
        experiment_id=exp_id,
    )
    assert out_path is not None
    assert out_path == (
        experiment_dirs
        / "experiments"
        / exp_id
        / FEATURE_IMPORTANCE_DIRNAME
        / "random_forest.csv"
    )
    assert out_path.exists()


def test_enrich_models_snapshot_adds_paths(experiment_dirs) -> None:
    from src.utils.experiments import allocate_experiment_id, experiment_root

    exp_id = allocate_experiment_id()
    fi_dir = experiment_root(exp_id) / FEATURE_IMPORTANCE_DIRNAME
    fi_dir.mkdir(parents=True)
    (fi_dir / "xgboost.csv").write_text("feature,importance\na,1.0\n")

    models = snapshot_hyperparameters()
    enriched = enrich_models_snapshot(exp_id, models)

    assert "feature_importance" not in enriched["baseline"]
    assert enriched["xgboost"]["feature_importance"] == "feature_importance/xgboost.csv"


def test_build_result_config_includes_feature_importance_paths(
    experiment_dirs,
) -> None:
    import pandas as pd

    from src.utils.experiments import (
        allocate_experiment_id,
        build_result_config,
        experiment_root,
    )

    exp_id = allocate_experiment_id()
    fi_dir = experiment_root(exp_id) / FEATURE_IMPORTANCE_DIRNAME
    fi_dir.mkdir(parents=True)
    (fi_dir / "random_forest.csv").write_text("feature,importance\na,1.0\n")

    comparison = pd.DataFrame(
        {
            "model": ["baseline", "random_forest"],
            "roc_auc_mean": [0.5, 0.8],
            "roc_auc_std": [0.0, 0.0],
        }
    )
    result_cfg = build_result_config(
        comparison,
        metric="roc_auc",
        higher_is_better=True,
        experiment_id=exp_id,
    )
    assert result_cfg["best_model"] == "random_forest"
    assert (
        result_cfg["models"]["random_forest"]["feature_importance"]
        == "feature_importance/random_forest.csv"
    )
