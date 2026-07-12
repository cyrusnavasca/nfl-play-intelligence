"""Static model hyperparameter config loading."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.models.config import load_model_hyperparameters, snapshot_hyperparameters
from src.utils.experiment_profile import load_experiment_profile, use_experiment_profile


def test_load_xgboost_config() -> None:
    profile = load_experiment_profile(Path("configs/models/xgboost.yaml"))
    with use_experiment_profile(profile):
        params = load_model_hyperparameters("xgboost")
    assert params["n_estimators"] == 300
    assert params["max_depth"] == 6
    assert params["random_state"] == 42


def test_snapshot_uses_provided_models_config() -> None:
    models_config = {
        "baseline": {"strategy": "prior"},
        "logistic_regression": {"C": 1.0},
        "random_forest": {"n_estimators": 300},
        "xgboost": {"max_depth": 6},
    }
    snapshot = snapshot_hyperparameters(models_config)
    assert set(snapshot) == set(models_config)
    assert snapshot["baseline"]["hyperparameters"]["strategy"] == "prior"


def test_build_result_config_shape() -> None:
    import pandas as pd

    from src.utils.experiments import build_result_config

    comparison = pd.DataFrame(
        {
            "model": ["baseline", "xgboost"],
            "roc_auc_mean": [0.5, 0.81],
            "roc_auc_std": [0.0, 0.01],
        }
    )
    result_cfg = build_result_config(
        comparison,
        metric="roc_auc",
        higher_is_better=True,
    )
    assert result_cfg["best_model"] == "xgboost"
    assert result_cfg["roc_auc"] == pytest.approx(0.81)
    assert "models" in result_cfg
    assert "hyperparameters" in result_cfg["models"]["xgboost"]
