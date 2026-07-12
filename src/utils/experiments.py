"""
Experiment-scoped modeling artifact paths and metadata.

Layout::

    artifacts/modeling/
    ├── experiments/
    │   ├── exp_001/
    │   │   ├── config.yaml
    │   │   ├── cv_results.csv
    │   │   ├── model_comparison.csv
    │   │   ├── feature_importance/
    │   │   ├── model.joblib
    │   │   └── metadata.json
    │   └── exp_002/
    ├── best_model/
    │   ├── config.yaml
    │   ├── model.joblib
    │   ├── metadata.json
    │   └── feature_importance/
    └── active.json          # {"experiment_id": "exp_002"}
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.schema import MODELING_ARTIFACTS_DIR
from src.evaluation.feature_importance import (
    FEATURE_IMPORTANCE_DIRNAME,
    enrich_models_snapshot,
)
from src.evaluation.model_selection import select_best_model
from src.models.config import snapshot_hyperparameters

__all__ = [
    "ACTIVE_EXPERIMENT_FILENAME",
    "BEST_MODEL_DIRNAME",
    "CONFIG_FILENAME",
    "EXPERIMENTS_DIR",
    "allocate_experiment_id",
    "best_model_dir",
    "build_result_config",
    "experiment_root",
    "get_active_experiment",
    "list_experiments",
    "promote_experiment_to_best_model",
    "read_experiment_config",
    "resolve_artifacts_dir",
    "set_active_experiment",
    "update_experiment_config",
    "write_experiment_config",
]

EXPERIMENTS_DIR = MODELING_ARTIFACTS_DIR / "experiments"
BEST_MODEL_DIRNAME = "best_model"
BEST_MODEL_DIR = MODELING_ARTIFACTS_DIR / BEST_MODEL_DIRNAME
ACTIVE_EXPERIMENT_FILENAME = "active.json"
CONFIG_FILENAME = "config.yaml"

_AUTO_ID_PATTERN = re.compile(r"^exp_(\d+)$")


def list_experiments() -> list[str]:
    """Return experiment ids sorted oldest-first (auto ids by number, then named)."""
    if not EXPERIMENTS_DIR.exists():
        return []

    auto_ids: list[tuple[int, str]] = []
    named: list[str] = []
    for path in EXPERIMENTS_DIR.iterdir():
        if not path.is_dir():
            continue
        match = _AUTO_ID_PATTERN.match(path.name)
        if match:
            auto_ids.append((int(match.group(1)), path.name))
        else:
            named.append(path.name)

    auto_ids.sort(key=lambda item: item[0])
    named.sort()
    return [name for _, name in auto_ids] + named


def allocate_experiment_id(name: str | None = None) -> str:
    """
    Return *name* when provided, otherwise the next ``exp_NNN`` id.

    Creates the experiment root directory. Raises when *name* already exists.
    """
    if name:
        if (EXPERIMENTS_DIR / name).exists():
            raise FileExistsError(f"experiment already exists: {name!r}")
        experiment_root(name)
        return name

    max_num = 0
    for exp_id in list_experiments():
        match = _AUTO_ID_PATTERN.match(exp_id)
        if match:
            max_num = max(max_num, int(match.group(1)))
    exp_id = f"exp_{max_num + 1:03d}"
    experiment_root(exp_id)
    return exp_id


def experiment_root(experiment_id: str) -> Path:
    """Root directory for an experiment (created on demand). Also its artifact dir."""
    path = EXPERIMENTS_DIR / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def best_model_dir() -> Path:
    """Promoted best-model directory (created on demand)."""
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return BEST_MODEL_DIR


def _active_experiment_path() -> Path:
    return MODELING_ARTIFACTS_DIR / ACTIVE_EXPERIMENT_FILENAME


def get_active_experiment() -> str | None:
    """Return the active experiment id, if set."""
    path = _active_experiment_path()
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    exp_id = payload.get("experiment_id")
    return str(exp_id) if exp_id else None


def set_active_experiment(experiment_id: str) -> None:
    """Mark *experiment_id* as the active artifact source."""
    path = _active_experiment_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"experiment_id": experiment_id}, indent=2) + "\n")


def resolve_artifacts_dir(*, experiment_id: str | None = None) -> Path:
    """
    Directory for reading experiment artifacts.

    Resolution order:
      1. explicit *experiment_id*
      2. active experiment (``active.json``)
      3. latest experiment directory
    """
    if experiment_id:
        return experiment_root(experiment_id)

    active = get_active_experiment()
    if active:
        return experiment_root(active)

    experiments = list_experiments()
    if experiments:
        return experiment_root(experiments[-1])

    raise FileNotFoundError(
        "No modeling artifacts found. Run training or pass experiment_id=..."
    )


def build_result_config(
    comparison: pd.DataFrame,
    *,
    metric: str,
    higher_is_better: bool,
    experiment_id: str | None = None,
    models_config: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Build the result block for experiment ``config.yaml``.

    Includes ``best_model``, the primary metric scalar, and per-model snapshots
    (hyperparameters plus ``feature_importance`` paths when present).
    """
    best_model = select_best_model(
        comparison,
        metric,
        higher_is_better=higher_is_better,
    )
    mean_col = f"{metric}_mean"
    metric_value = float(
        comparison.loc[comparison["model"] == best_model, mean_col].iloc[0]
    )
    models = snapshot_hyperparameters(models_config=models_config)
    if experiment_id:
        models = enrich_models_snapshot(experiment_id, models)
    return {
        "best_model": best_model,
        metric: metric_value,
        "models": models,
    }


def _deep_merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def write_experiment_config(experiment_id: str, config: dict[str, Any]) -> Path:
    """Write or replace the root ``config.yaml`` for an experiment."""
    root = experiment_root(experiment_id)
    path = root / CONFIG_FILENAME
    payload = dict(config)
    payload.setdefault("experiment_id", experiment_id)
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def read_experiment_config(experiment_id: str) -> dict[str, Any]:
    """Load experiment ``config.yaml``; empty dict when missing."""
    path = experiment_root(experiment_id) / CONFIG_FILENAME
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def update_experiment_config(experiment_id: str, patch: dict[str, Any]) -> Path:
    """Deep-merge *patch* into an experiment config and rewrite ``config.yaml``."""
    config = read_experiment_config(experiment_id)
    merged = _deep_merge_dict(config, patch)
    return write_experiment_config(experiment_id, merged)


def promote_experiment_to_best_model(
    experiment_id: str,
    *,
    source_files: list[str] | None = None,
) -> Path:
    """
    Copy selected artifact files from an experiment dir into ``best_model/``.

    Writes a ``config.yaml`` pointer to the source experiment without removing an
    existing refit ``model.joblib``.
    """
    src_dir = experiment_root(experiment_id)
    dst_dir = best_model_dir()

    if source_files:
        for filename in source_files:
            src = src_dir / filename
            if src.is_dir():
                shutil.copytree(src, dst_dir / filename, dirs_exist_ok=True)
            elif src.exists():
                shutil.copy2(src, dst_dir / filename)

    fi_src = src_dir / FEATURE_IMPORTANCE_DIRNAME
    if fi_src.is_dir() and any(fi_src.iterdir()):
        shutil.copytree(
            fi_src,
            dst_dir / FEATURE_IMPORTANCE_DIRNAME,
            dirs_exist_ok=True,
        )

    pointer = {
        "source_experiment": experiment_id,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        **read_experiment_config(experiment_id),
    }
    (dst_dir / CONFIG_FILENAME).write_text(yaml.safe_dump(pointer, sort_keys=False))
    return dst_dir
