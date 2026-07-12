"""Shared fixtures for modeling validation tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.loaders import load_play_type_dataset
from src.data.schema import MODEL_REGISTRY_KEYS, N_FOLDS, SEED
from src.utils.experiment_profile import (
    ExperimentProfile,
    load_experiment_profile,
    use_experiment_profile,
)

SUBSAMPLE_ROWS = 1_000

# One profile per registry key lives under configs/models/<key>.yaml.
_MODEL_CONFIG_PATHS = {
    key: Path(f"configs/models/{key}.yaml") for key in MODEL_REGISTRY_KEYS
}


def _all_models_profile() -> ExperimentProfile:
    """Merge every configs/models/<key>.yaml into one all-registry test profile."""
    models: dict = {}
    for key, path in _MODEL_CONFIG_PATHS.items():
        models.update(load_experiment_profile(path).models)
    return ExperimentProfile(
        name="test_all_models",
        description=None,
        seed=SEED,
        n_folds=N_FOLDS,
        persist_best=False,
        models=models,
    )


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
    """Activate an all-registry profile so builders resolve hyperparameters in tests."""
    profile = _all_models_profile()
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
