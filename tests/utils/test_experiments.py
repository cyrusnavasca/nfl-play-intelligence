"""Experiment artifact path helpers."""
from __future__ import annotations

from src.utils.experiments import (
    allocate_experiment_id,
    experiment_root,
    get_active_experiment,
    list_experiments,
    promote_experiment_to_best_model,
    resolve_artifacts_dir,
    set_active_experiment,
    update_experiment_config,
    write_experiment_config,
)


def test_allocate_experiment_id_auto_increments(experiment_dirs) -> None:
    first = allocate_experiment_id()
    second = allocate_experiment_id()
    assert first == "exp_001"
    assert second == "exp_002"
    assert list_experiments() == ["exp_001", "exp_002"]


def test_allocate_experiment_id_named(experiment_dirs) -> None:
    exp_id = allocate_experiment_id("trial_v1")
    assert exp_id == "trial_v1"
    assert (experiment_dirs / "experiments" / exp_id).exists()


def test_active_experiment_resolution(experiment_dirs) -> None:
    exp_id = allocate_experiment_id()
    (experiment_root(exp_id) / "model_comparison.csv").write_text(
        "model,roc_auc_mean\nbaseline,0.5\n"
    )
    set_active_experiment(exp_id)

    resolved = resolve_artifacts_dir()
    assert resolved == experiment_root(exp_id)
    assert get_active_experiment() == exp_id


def test_resolve_falls_back_to_latest_experiment(experiment_dirs) -> None:
    allocate_experiment_id()
    second = allocate_experiment_id()
    # No active.json set → resolves to the latest experiment.
    assert resolve_artifacts_dir() == experiment_root(second)


def test_update_experiment_config_merges_nested(experiment_dirs) -> None:
    exp_id = allocate_experiment_id()
    write_experiment_config(exp_id, {"models": {"xgboost": {"seed": 42}}})
    update_experiment_config(
        exp_id,
        {"models": {"xgboost": {"best": True}}},
    )

    active_path = experiment_dirs / "active.json"
    assert not active_path.exists()

    config_path = experiment_dirs / "experiments" / exp_id / "config.yaml"
    payload = config_path.read_text()
    assert "seed: 42" in payload
    assert "best: true" in payload


def test_promote_copies_files_without_dropping_model(experiment_dirs) -> None:
    exp_id = allocate_experiment_id()
    src = experiment_root(exp_id)
    (src / "model_comparison.csv").write_text("model,roc_auc_mean\nxgboost,0.8\n")
    (src / "metadata.json").write_text("{}\n")

    dst = promote_experiment_to_best_model(
        exp_id,
        source_files=["model_comparison.csv", "metadata.json"],
    )
    (dst / "model.joblib").write_bytes(b"saved-later")

    promote_experiment_to_best_model(
        exp_id,
        source_files=["model_comparison.csv"],
    )
    assert (dst / "model.joblib").read_bytes() == b"saved-later"
    assert (dst / "config.yaml").exists()
