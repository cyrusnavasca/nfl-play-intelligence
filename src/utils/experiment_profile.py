"""
Experiment profile loading and active-run context.

Profiles live at ``configs/models/<profile>.yaml`` and describe which models to train
plus their hyperparameters. The runner copies the resolved profile into
``artifacts/modeling/experiments/<id>/config.yaml`` at run start.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

from src.data.schema import MODEL_REGISTRY_KEYS, N_FOLDS, SEED

__all__ = [
    "DEFAULT_PROFILE_PATH",
    "ExperimentProfile",
    "clear_active_profile",
    "get_active_profile",
    "get_active_profile_or_none",
    "load_experiment_profile",
    "use_experiment_profile",
    "validate_experiment_profile",
]

DEFAULT_PROFILE_PATH = Path("configs/models/baseline.yaml")

_active_profile: ExperimentProfile | None = None


@dataclass(frozen=True)
class ExperimentProfile:
    """Resolved experiment profile used to drive a modeling run."""

    name: str
    description: str | None
    seed: int
    n_folds: int
    persist_best: bool
    models: dict[str, dict[str, Any]]
    source_path: Path | None = None
    # Optuna hyperparameter search: study settings + per-model search spaces.
    tune: dict[str, Any] = field(default_factory=dict)
    search_spaces: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    def model_keys(self) -> tuple[str, ...]:
        return tuple(self.models)

    def has_search_space(self, model_key: str) -> bool:
        return bool(self.search_spaces.get(model_key))

    def search_space(self, model_key: str) -> dict[str, dict[str, Any]]:
        return dict(self.search_spaces.get(model_key, {}))

    def model_hyperparameters(self, model_key: str) -> dict[str, Any]:
        try:
            return dict(self.models[model_key])
        except KeyError as exc:
            raise KeyError(
                f"model {model_key!r} not configured in profile {self.name!r}"
            ) from exc

    def models_config(self) -> dict[str, dict[str, Any]]:
        """Return ``{model_key: hyperparameters}`` for experiment snapshots."""
        return {
            model_key: dict(hyperparameters)
            for model_key, hyperparameters in self.models.items()
        }

    def to_snapshot_base(self, experiment_id: str) -> dict[str, Any]:
        """Initial experiment config written before training starts."""
        payload: dict[str, Any] = {
            "experiment_id": experiment_id,
            "name": self.name,
            "seed": self.seed,
            "n_folds": self.n_folds,
            "persist_best": self.persist_best,
            "pipeline": "src.pipelines.run",
            "models": {
                model_key: {"hyperparameters": dict(hyperparameters)}
                for model_key, hyperparameters in self.models.items()
            },
        }
        if self.description:
            payload["description"] = self.description
        if self.source_path is not None:
            payload["profile"] = str(self.source_path)
        return payload


def validate_experiment_profile(raw: dict[str, Any], *, source: str = "profile") -> None:
    """Raise ``ValueError`` when *raw* does not match the profile schema."""
    if not isinstance(raw, dict):
        raise ValueError(f"{source} must be a mapping")

    models = raw.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"{source} must define a non-empty 'models' mapping")

    for model_key, hyperparameters in models.items():
        if model_key not in MODEL_REGISTRY_KEYS:
            raise ValueError(
                f"{source} has unknown model key {model_key!r}; "
                f"expected one of {MODEL_REGISTRY_KEYS}"
            )
        if not isinstance(hyperparameters, dict):
            raise ValueError(
                f"{source} model {model_key!r} hyperparameters must be a mapping"
            )
        if "search" in hyperparameters:
            _validate_search_space(
                hyperparameters["search"],
                source=f"{source} model {model_key!r}",
            )

    tune = raw.get("tune")
    if tune is not None and not isinstance(tune, dict):
        raise ValueError(f"{source} 'tune' must be a mapping")


_SEARCH_DISTRIBUTIONS = {"int", "float", "categorical"}


def _validate_search_space(search: Any, *, source: str) -> None:
    """Validate a per-model ``search`` block of Optuna distribution specs."""
    if not isinstance(search, dict) or not search:
        raise ValueError(f"{source} 'search' must be a non-empty mapping")

    for param, spec in search.items():
        if not isinstance(spec, dict) or "type" not in spec:
            raise ValueError(
                f"{source} search param {param!r} must be a mapping with a 'type'"
            )
        dist = spec["type"]
        if dist not in _SEARCH_DISTRIBUTIONS:
            raise ValueError(
                f"{source} search param {param!r} has unknown type {dist!r}; "
                f"expected one of {sorted(_SEARCH_DISTRIBUTIONS)}"
            )
        if dist in {"int", "float"}:
            if "low" not in spec or "high" not in spec:
                raise ValueError(
                    f"{source} search param {param!r} ({dist}) needs 'low' and 'high'"
                )
        elif "choices" not in spec:
            raise ValueError(
                f"{source} search param {param!r} (categorical) needs 'choices'"
            )


def load_experiment_profile(path: str | Path) -> ExperimentProfile:
    """Load and validate an experiment profile from *path*."""
    profile_path = Path(path)
    if not profile_path.exists():
        raise FileNotFoundError(f"experiment profile not found: {profile_path}")

    raw = yaml.safe_load(profile_path.read_text()) or {}
    validate_experiment_profile(raw, source=str(profile_path))

    name = str(raw.get("name") or profile_path.stem)

    # Split each model block into fixed hyperparameters and an optional Optuna
    # search space (kept separate so builders never receive the search spec).
    models: dict[str, dict[str, Any]] = {}
    search_spaces: dict[str, dict[str, dict[str, Any]]] = {}
    for model_key, block in raw["models"].items():
        block = dict(block)
        search = block.pop("search", None)
        models[str(model_key)] = block
        if search:
            search_spaces[str(model_key)] = {
                str(param): dict(spec) for param, spec in search.items()
            }

    return ExperimentProfile(
        name=name,
        description=str(raw["description"]) if raw.get("description") else None,
        seed=int(raw.get("seed", SEED)),
        n_folds=int(raw.get("n_folds", N_FOLDS)),
        persist_best=bool(raw.get("persist_best", False)),
        models=models,
        source_path=profile_path,
        tune=dict(raw.get("tune") or {}),
        search_spaces=search_spaces,
    )


def get_active_profile() -> ExperimentProfile:
    """Return the active experiment profile for the current run."""
    if _active_profile is None:
        raise RuntimeError(
            "No active experiment profile. Pass --config to src.pipelines.run "
            "or wrap execution in use_experiment_profile()."
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
