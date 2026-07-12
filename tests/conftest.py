"""Shared fixtures for modeling validation tests."""
from __future__ import annotations

import pytest

from src.data.loaders import load_play_type_dataset
from src.utils.experiment_profile import (
    DEFAULT_PROFILE_PATH,
    load_experiment_profile,
    use_experiment_profile,
)

SUBSAMPLE_ROWS = 1_000


@pytest.fixture(scope="session")
def play_type_subsample() -> tuple:
    """First 1k rows of the play-type modeling frame."""
    X, y = load_play_type_dataset()
    return (
        X.iloc[:SUBSAMPLE_ROWS].reset_index(drop=True),
        y.iloc[:SUBSAMPLE_ROWS].reset_index(drop=True),
    )


@pytest.fixture(autouse=True)
def default_experiment_profile():
    """Activate the baseline profile so builders resolve hyperparameters in tests."""
    if not DEFAULT_PROFILE_PATH.exists():
        yield None
        return
    profile = load_experiment_profile(DEFAULT_PROFILE_PATH)
    with use_experiment_profile(profile):
        yield profile


@pytest.fixture
def experiment_dirs(tmp_path, monkeypatch):
    modeling_dir = tmp_path / "artifacts" / "modeling"
    monkeypatch.setattr(
        "src.utils.experiments.MODELING_ARTIFACTS_DIR",
        modeling_dir,
    )
    monkeypatch.setattr(
        "src.utils.experiments.EXPERIMENTS_DIR",
        modeling_dir / "experiments",
    )
    monkeypatch.setattr(
        "src.utils.experiments.BEST_MODEL_DIR",
        modeling_dir / "best_model",
    )
    monkeypatch.setattr(
        "src.evaluation.feature_importance.MODELING_ARTIFACTS_DIR",
        modeling_dir,
    )
    return modeling_dir
