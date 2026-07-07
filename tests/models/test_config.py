"""Static model hyperparameter config loading."""
from __future__ import annotations

import pytest

from src.models.config import load_model_hyperparameters, snapshot_task_hyperparameters


def test_load_xgboost_play_type_config() -> None:
    params = load_model_hyperparameters("play_type", "xgboost")
    assert params["n_estimators"] == 300
    assert params["max_depth"] == 6
    assert params["random_state"] == 42


def test_snapshot_includes_all_registry_keys() -> None:
    snapshot = snapshot_task_hyperparameters("yards_gained")
    assert set(snapshot) == {"baseline", "random_forest", "xgboost"}
    assert snapshot["baseline"]["hyperparameters"]["strategy"] == "mean"


def test_build_task_result_config_shape() -> None:
    import pandas as pd

    from src.utils.experiments import build_task_result_config

    comparison = pd.DataFrame(
        {
            "model": ["baseline", "xgboost"],
            "roc_auc_mean": [0.5, 0.81],
            "roc_auc_std": [0.0, 0.01],
        }
    )
    task_cfg = build_task_result_config(
        "play_type",
        comparison,
        metric="roc_auc",
        higher_is_better=True,
        experiment_id="exp_001",
    )
    assert task_cfg["best_model"] == "xgboost"
    assert task_cfg["roc_auc"] == pytest.approx(0.81)
    assert "models" in task_cfg
    assert "hyperparameters" in task_cfg["models"]["xgboost"]
