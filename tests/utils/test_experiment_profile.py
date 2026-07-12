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
    assert profile.name == "default"
    assert profile.model_keys() == (
        "baseline",
        "random_forest",
        "xgboost",
    )
    assert profile.model_hyperparameters("xgboost")["n_estimators"] == 300


def test_xgboost_tuned_profile_subsets_models() -> None:
    profile = load_experiment_profile(Path("configs/xgboost_tuned.yaml"))
    assert profile.model_keys() == ("baseline", "xgboost")
    assert profile.model_hyperparameters("xgboost")["max_depth"] == 8


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
    profile = load_experiment_profile(DEFAULT_PROFILE_PATH)
    from src.models.config import load_model_hyperparameters
    from src.utils.experiment_profile import get_active_profile_or_none

    assert get_active_profile_or_none() is not None
    assert load_model_hyperparameters("baseline")["strategy"] == "prior"

    with use_experiment_profile(profile):
        assert load_model_hyperparameters("xgboost")["max_depth"] == 6
