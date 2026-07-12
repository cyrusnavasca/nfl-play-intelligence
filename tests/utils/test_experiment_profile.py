"""Experiment profile loading and validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.experiment_profile import (
    DEFAULT_PROFILE_PATH,
    load_experiment_profile,
    use_experiment_profile,
    validate_experiment_profile,
)


def test_load_default_profile() -> None:
    profile = load_experiment_profile(DEFAULT_PROFILE_PATH)
    assert profile.name == "baseline"
    assert profile.model_keys() == ("baseline",)
    assert profile.model_hyperparameters("baseline")["strategy"] == "prior"


@pytest.mark.parametrize(
    "config, key",
    [
        ("configs/models/baseline.yaml", "baseline"),
        ("configs/models/logistic_regression.yaml", "logistic_regression"),
        ("configs/models/random_forest.yaml", "random_forest"),
        ("configs/models/xgboost.yaml", "xgboost"),
    ],
)
def test_model_profiles_load_single_model(config: str, key: str) -> None:
    profile = load_experiment_profile(Path(config))
    assert profile.model_keys() == (key,)


def test_xgboost_profile_hyperparameters() -> None:
    profile = load_experiment_profile(Path("configs/models/xgboost.yaml"))
    assert profile.model_hyperparameters("xgboost")["max_depth"] == 6


def test_validate_rejects_unknown_model_key() -> None:
    with pytest.raises(ValueError, match="unknown model key"):
        validate_experiment_profile(
            {
                "models": {
                    "not_a_model": {"foo": 1},
                }
            }
        )


def test_validate_rejects_missing_models() -> None:
    with pytest.raises(ValueError, match="non-empty 'models' mapping"):
        validate_experiment_profile({"name": "no_models"})


def test_use_experiment_profile_context() -> None:
    profile = load_experiment_profile(Path("configs/models/xgboost.yaml"))
    from src.models.config import load_model_hyperparameters

    with use_experiment_profile(profile):
        assert load_model_hyperparameters("xgboost")["max_depth"] == 6
