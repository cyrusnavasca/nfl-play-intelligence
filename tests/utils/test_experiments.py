"""Experiment artifact path helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.experiments import (
    allocate_experiment_id,
    get_active_experiment,
    list_experiments,
    promote_experiment_to_best_model,
    migrate_legacy_artifacts,
    resolve_task_artifacts_dir,
    set_active_experiment,
    task_experiment_dir,
    update_experiment_config,
    write_experiment_config,
)


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
        "src.utils.experiments._LEGACY_TASK_DIRS",
        {
            "play_type": modeling_dir / "play_type",
            "yards_gained": modeling_dir / "yards_gained",
        },
    )
    return modeling_dir


def test_allocate_experiment_id_auto_increments(experiment_dirs) -> None:
    first = allocate_experiment_id()
    second = allocate_experiment_id()
    assert first == "exp_001"
    assert second == "exp_002"
    assert list_experiments() == ["exp_001", "exp_002"]


def test_allocate_experiment_id_named(experiment_dirs) -> None:
    exp_id = allocate_experiment_id("log_transform_v1")
    assert exp_id == "log_transform_v1"
    assert (experiment_dirs / "experiments" / exp_id).exists()


def test_active_experiment_resolution(experiment_dirs) -> None:
    exp_id = allocate_experiment_id()
    task_experiment_dir(exp_id, "yards_gained")
    (task_experiment_dir(exp_id, "yards_gained") / "holdout_results.csv").write_text(
        "model,rmse\nbaseline,1.0\n"
    )
    set_active_experiment("yards_gained", exp_id)

    resolved = resolve_task_artifacts_dir("yards_gained")
    assert resolved == task_experiment_dir(exp_id, "yards_gained")
    assert get_active_experiment("yards_gained") == exp_id


def test_migrate_legacy_artifacts_to_experiment(experiment_dirs) -> None:
    legacy_play = experiment_dirs / "play_type"
    legacy_yards = experiment_dirs / "yards_gained"
    legacy_play.mkdir(parents=True)
    legacy_yards.mkdir(parents=True)
    (legacy_play / "cv_results.csv").write_text("model,fold\nxgboost,1\n")
    (legacy_play / "model_comparison.csv").write_text("model,roc_auc_mean\nxgboost,0.8\n")
    (legacy_yards / "holdout_results.csv").write_text("model,rmse\nxgboost,8.5\n")

    result = migrate_legacy_artifacts("exp_001")

    assert result["experiment_id"] == "exp_001"
    assert (experiment_dirs / "experiments" / "exp_001" / "play_type" / "cv_results.csv").exists()
    assert not legacy_play.exists() or not any(legacy_play.iterdir())
    assert get_active_experiment("play_type") == "exp_001"
    assert get_active_experiment("yards_gained") == "exp_001"
    assert resolve_task_artifacts_dir("play_type") == task_experiment_dir("exp_001", "play_type")


def test_resolve_falls_back_to_legacy_dir(experiment_dirs) -> None:
    legacy = experiment_dirs / "play_type"
    legacy.mkdir(parents=True)
    (legacy / "cv_results.csv").write_text("fold\n1\n")

    resolved = resolve_task_artifacts_dir("play_type")
    assert resolved == legacy


def test_update_experiment_config_merges_nested(experiment_dirs) -> None:
    exp_id = allocate_experiment_id()
    write_experiment_config(exp_id, {"tasks": {"play_type": {"seed": 42}}})
    update_experiment_config(
        exp_id,
        {"tasks": {"yards_gained": {"best_model": "xgboost"}}},
    )

    active_path = experiment_dirs / "active.json"
    assert not active_path.exists()

    config_path = experiment_dirs / "experiments" / exp_id / "config.yaml"
    payload = config_path.read_text()
    assert "play_type" in payload
    assert "yards_gained" in payload


def test_promote_copies_files_without_dropping_model(experiment_dirs) -> None:
    exp_id = allocate_experiment_id()
    src = task_experiment_dir(exp_id, "play_type")
    (src / "model_comparison.csv").write_text("model,roc_auc_mean\nxgboost,0.8\n")
    (src / "metadata.json").write_text("{}\n")

    dst = promote_experiment_to_best_model(
        "play_type",
        exp_id,
        source_files=["model_comparison.csv", "metadata.json"],
    )
    (dst / "model.joblib").write_bytes(b"saved-later")

    promote_experiment_to_best_model(
        "play_type",
        exp_id,
        source_files=["model_comparison.csv"],
    )
    assert (dst / "model.joblib").read_bytes() == b"saved-later"
    assert (dst / "config.yaml").exists()
