"""
Experiment profile loading and active-run context.

Profiles live at ``configs/<profile>.yaml`` and describe which tasks and models
to train plus their hyperparameters. The runner copies the resolved profile into
``artifacts/modeling/experiments/<id>/config.yaml`` at run start.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml

from src.data.schema import MODEL_REGISTRY_KEYS, N_FOLDS, SEED, ModelingTask
from src.preprocessing.target_transform import TARGET_TRANSFORM_KEYS

__all__ = [
    "DEFAULT_PROFILE_PATH",
    "ExperimentProfile",
    "TaskProfile",
    "clear_active_profile",
    "get_active_profile",
    "get_active_profile_or_none",
    "load_experiment_profile",
    "use_experiment_profile",
    "validate_experiment_profile",
]

DEFAULT_PROFILE_PATH = Path("configs/xgboost_baseline.yaml")
MODELING_TASKS: tuple[ModelingTask, ...] = ("play_type", "yards_gained")

_active_profile: ExperimentProfile | None = None


@dataclass(frozen=True)
class TaskProfile:
    """Models and hyperparameters for one modeling task."""

    models: dict[str, dict[str, Any]]
    target_transform: str | None = None

    def model_keys(self) -> tuple[str, ...]:
        return tuple(self.models)


@dataclass(frozen=True)
class ExperimentProfile:
    """Resolved experiment profile used to drive a modeling run."""

    name: str
    description: str | None
    seed: int
    n_folds: int
    persist_best: bool
    play_type_experiment: str | None
    tasks: dict[ModelingTask, TaskProfile]
    source_path: Path | None = None

    def has_task(self, task: ModelingTask) -> bool:
        return task in self.tasks

    def task_model_keys(self, task: ModelingTask) -> tuple[str, ...]:
        return self.tasks[task].model_keys()

    def model_hyperparameters(self, task: ModelingTask, model_key: str) -> dict[str, Any]:
        try:
            return dict(self.tasks[task].models[model_key])
        except KeyError as exc:
            raise KeyError(
                f"model {model_key!r} not configured for task {task!r} "
                f"in profile {self.name!r}"
            ) from exc

    def task_models_config(self, task: ModelingTask) -> dict[str, dict[str, Any]]:
        """Return ``{model_key: hyperparameters}`` for experiment snapshots."""
        return {
            model_key: dict(hyperparameters)
            for model_key, hyperparameters in self.tasks[task].models.items()
        }

    def task_target_transform(self, task: ModelingTask) -> str:
        """Return the configured target transform for *task* (yards gained only)."""
        task_profile = self.tasks.get(task)
        if task_profile is None or task_profile.target_transform is None:
            return "none"
        return task_profile.target_transform

    def to_snapshot_base(self, experiment_id: str) -> dict[str, Any]:
        """Initial experiment config written before training starts."""
        payload: dict[str, Any] = {
            "experiment_id": experiment_id,
            "name": self.name,
            "seed": self.seed,
            "n_folds": self.n_folds,
            "persist_best": self.persist_best,
            "pipeline": "src.pipelines.run",
            "tasks": {},
        }
        if self.description:
            payload["description"] = self.description
        if self.source_path is not None:
            payload["profile"] = str(self.source_path)
        if self.play_type_experiment is not None:
            payload["play_type_experiment"] = self.play_type_experiment

        for task, task_profile in self.tasks.items():
            task_payload: dict[str, Any] = {
                "models": {
                    model_key: {"hyperparameters": dict(hyperparameters)}
                    for model_key, hyperparameters in task_profile.models.items()
                }
            }
            if task_profile.target_transform is not None:
                task_payload["target_transform"] = task_profile.target_transform
            payload["tasks"][task] = task_payload
        return payload


def validate_experiment_profile(raw: dict[str, Any], *, source: str = "profile") -> None:
    """Raise ``ValueError`` when *raw* does not match the profile schema."""
    if not isinstance(raw, dict):
        raise ValueError(f"{source} must be a mapping")

    tasks = raw.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError(f"{source} must define at least one task under 'tasks'")

    for task_name, task_block in tasks.items():
        if task_name not in MODELING_TASKS:
            raise ValueError(
                f"{source} has unknown task {task_name!r}; "
                f"expected one of {MODELING_TASKS}"
            )
        if not isinstance(task_block, dict):
            raise ValueError(f"{source} task {task_name!r} must be a mapping")

        models = task_block.get("models")
        if not isinstance(models, dict) or not models:
            raise ValueError(
                f"{source} task {task_name!r} must define a non-empty 'models' mapping"
            )

        target_transform = task_block.get("target_transform")
        if target_transform is not None:
            if task_name != "yards_gained":
                raise ValueError(
                    f"{source} task {task_name!r} does not support "
                    f"'target_transform' (yards_gained only)"
                )
            if str(target_transform) not in TARGET_TRANSFORM_KEYS:
                raise ValueError(
                    f"{source} task {task_name!r} has unknown target_transform "
                    f"{target_transform!r}; expected one of {TARGET_TRANSFORM_KEYS}"
                )

        for model_key, hyperparameters in models.items():
            if model_key not in MODEL_REGISTRY_KEYS:
                raise ValueError(
                    f"{source} task {task_name!r} has unknown model key "
                    f"{model_key!r}; expected one of {MODEL_REGISTRY_KEYS}"
                )
            if not isinstance(hyperparameters, dict):
                raise ValueError(
                    f"{source} task {task_name!r} model {model_key!r} "
                    "hyperparameters must be a mapping"
                )


def _parse_task_profiles(raw_tasks: dict[str, Any]) -> dict[ModelingTask, TaskProfile]:
    tasks: dict[ModelingTask, TaskProfile] = {}
    for task_name, task_block in raw_tasks.items():
        task = task_name  # type: ignore[assignment]
        models = {
            str(model_key): dict(hyperparameters)
            for model_key, hyperparameters in task_block["models"].items()
        }
        raw_transform = task_block.get("target_transform")
        target_transform = str(raw_transform) if raw_transform is not None else None
        tasks[task] = TaskProfile(
            models=models,
            target_transform=target_transform,
        )
    return tasks


def load_experiment_profile(path: str | Path) -> ExperimentProfile:
    """Load and validate an experiment profile from *path*."""
    profile_path = Path(path)
    if not profile_path.exists():
        raise FileNotFoundError(f"experiment profile not found: {profile_path}")

    raw = yaml.safe_load(profile_path.read_text()) or {}
    validate_experiment_profile(raw, source=str(profile_path))

    name = str(raw.get("name") or profile_path.stem)
    return ExperimentProfile(
        name=name,
        description=str(raw["description"]) if raw.get("description") else None,
        seed=int(raw.get("seed", SEED)),
        n_folds=int(raw.get("n_folds", N_FOLDS)),
        persist_best=bool(raw.get("persist_best", False)),
        play_type_experiment=(
            str(raw["play_type_experiment"])
            if raw.get("play_type_experiment")
            else None
        ),
        tasks=_parse_task_profiles(raw["tasks"]),
        source_path=profile_path,
    )


def get_active_profile() -> ExperimentProfile:
    """Return the active experiment profile for the current run."""
    if _active_profile is None:
        raise RuntimeError(
            "No active experiment profile. Pass --config to src.pipelines.run "
            f"or wrap execution in use_experiment_profile()."
        )
    return _active_profile


def get_active_profile_or_none() -> ExperimentProfile | None:
    return _active_profile


def clear_active_profile() -> None:
    global _active_profile
    _active_profile = None


@contextmanager
def use_experiment_profile(profile: ExperimentProfile) -> Iterator[ExperimentProfile]:
    """Temporarily set *profile* as the active hyperparameter source."""
    global _active_profile
    previous = _active_profile
    _active_profile = profile
    try:
        yield profile
    finally:
        _active_profile = previous
