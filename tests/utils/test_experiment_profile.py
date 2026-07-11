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


def test_load_xgboost_baseline_profile() -> None:
    profile = load_experiment_profile(DEFAULT_PROFILE_PATH)
    assert profile.name == "xgboost_baseline"
    assert profile.has_task("play_type")
    assert profile.has_task("yards_gained")
    assert profile.task_model_keys("play_type") == (
        "baseline",
        "random_forest",
        "xgboost",
    )
    assert profile.model_hyperparameters("play_type", "xgboost")["n_estimators"] == 300


def test_xgboost_tuned_profile_subsets_models() -> None:
    profile = load_experiment_profile(Path("configs/xgboost_tuned.yaml"))
    assert profile.task_model_keys("play_type") == ("baseline", "xgboost")
    assert profile.model_hyperparameters("play_type", "xgboost")["max_depth"] == 8


def test_validate_rejects_unknown_model_key() -> None:
    with pytest.raises(ValueError, match="unknown model key"):
        validate_experiment_profile(
            {
                "tasks": {
                    "play_type": {
                        "models": {
                            "not_a_model": {"foo": 1},
                        }
                    }
                }
            }
        )


def test_validate_rejects_unknown_target_transform() -> None:
    with pytest.raises(ValueError, match="unknown target_transform"):
        validate_experiment_profile(
            {
                "tasks": {
                    "yards_gained": {
                        "target_transform": "sqrt",
                        "models": {
                            "baseline": {"strategy": "mean"},
                        },
                    }
                }
            }
        )


def test_validate_rejects_target_transform_on_play_type() -> None:
    with pytest.raises(ValueError, match="does not support"):
        validate_experiment_profile(
            {
                "tasks": {
                    "play_type": {
                        "target_transform": "log",
                        "models": {
                            "baseline": {"strategy": "prior"},
                        },
                    }
                }
            }
        )


def test_load_profile_with_target_transform(tmp_path: Path) -> None:
    profile_path = tmp_path / "log_profile.yaml"
    profile_path.write_text(
        """
name: log_trial
tasks:
  yards_gained:
    target_transform: log
    models:
      baseline:
        strategy: mean
"""
    )
    profile = load_experiment_profile(profile_path)
    assert profile.task_target_transform("yards_gained") == "log"


def test_use_experiment_profile_context() -> None:
    profile = load_experiment_profile(DEFAULT_PROFILE_PATH)
    from src.models.config import load_model_hyperparameters
    from src.utils.experiment_profile import get_active_profile_or_none

    assert get_active_profile_or_none() is not None
    assert load_model_hyperparameters("play_type", "baseline")["strategy"] == "prior"

    with use_experiment_profile(profile):
        assert load_model_hyperparameters("play_type", "xgboost")["max_depth"] == 6
