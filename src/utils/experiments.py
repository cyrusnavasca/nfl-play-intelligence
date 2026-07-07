"""
Experiment-scoped modeling artifact paths and metadata.

Layout::

    artifacts/modeling/
    ├── experiments/
    │   ├── exp_001/
    │   │   ├── config.yaml
    │   │   ├── play_type/
    │   │   │   ├── cv_results.csv
    │   │   │   ├── feature_importance/
    │   │   │   ├── model_comparison.csv
    │   │   │   └── oof_predictions.parquet
    │   │   └── yards_gained/
    │   │       ├── feature_importance/
    │   │       ├── holdout_results.csv
    │   │       ├── model_comparison.csv
    │   │       └── test_predictions.parquet
    │   └── exp_002/
    ├── best_model/
    │   ├── play_type/
    │   └── yards_gained/
    └── active.json

Legacy flat task dirs (``play_type/``, ``yards_gained/``) are supported for
reads only until migrated via :func:`migrate_legacy_artifacts`.
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

from src.data.schema import (
    MODELING_ARTIFACTS_DIR,
    PLAY_TYPE_ARTIFACTS_DIR,
    YARDS_GAINED_ARTIFACTS_DIR,
    ModelingTask,
)
from src.evaluation.feature_importance import (
    FEATURE_IMPORTANCE_DIRNAME,
    enrich_models_snapshot,
)
from src.evaluation.model_selection import select_best_model
from src.models.config import snapshot_task_hyperparameters
from src.utils.experiment_profile import (
    DEFAULT_PROFILE_PATH,
    load_experiment_profile,
    use_experiment_profile,
)

__all__ = [
    "ACTIVE_EXPERIMENTS_FILENAME",
    "BEST_MODEL_DIRNAME",
    "CONFIG_FILENAME",
    "EXPERIMENTS_DIR",
    "allocate_experiment_id",
    "build_task_result_config",
    "experiment_root",
    "get_active_experiment",
    "list_experiments",
    "migrate_legacy_artifacts",
    "promote_experiment_to_best_model",
    "read_experiment_config",
    "resolve_task_artifacts_dir",
    "set_active_experiment",
    "task_experiment_dir",
    "update_experiment_config",
    "write_experiment_config",
]

EXPERIMENTS_DIR = MODELING_ARTIFACTS_DIR / "experiments"
BEST_MODEL_DIRNAME = "best_model"
BEST_MODEL_DIR = MODELING_ARTIFACTS_DIR / BEST_MODEL_DIRNAME
ACTIVE_EXPERIMENTS_FILENAME = "active.json"
CONFIG_FILENAME = "config.yaml"

_AUTO_ID_PATTERN = re.compile(r"^exp_(\d+)$")

_LEGACY_TASK_DIRS: dict[ModelingTask, Path] = {
    "play_type": PLAY_TYPE_ARTIFACTS_DIR,
    "yards_gained": YARDS_GAINED_ARTIFACTS_DIR,
}

_LEGACY_TASK_ARTIFACTS: dict[ModelingTask, tuple[str, ...]] = {
    "play_type": (
        "cv_results.csv",
        "model_comparison.csv",
        "oof_predictions.parquet",
        "model.joblib",
        "metadata.json",
    ),
    "yards_gained": (
        "holdout_results.csv",
        "model_comparison.csv",
        "test_predictions.parquet",
        "model.joblib",
        "metadata.json",
    ),
}


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
    """Root directory for an experiment (created on demand)."""
    path = EXPERIMENTS_DIR / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def task_experiment_dir(experiment_id: str, task: ModelingTask) -> Path:
    """Task subdirectory inside an experiment."""
    path = experiment_root(experiment_id) / task
    path.mkdir(parents=True, exist_ok=True)
    return path


def best_model_task_dir(task: ModelingTask) -> Path:
    """Promoted best-model directory for *task*."""
    path = BEST_MODEL_DIR / task
    path.mkdir(parents=True, exist_ok=True)
    return path


def _active_experiments_path() -> Path:
    return MODELING_ARTIFACTS_DIR / ACTIVE_EXPERIMENTS_FILENAME


def get_active_experiment(task: ModelingTask) -> str | None:
    """Return the active experiment id for *task*, if set."""
    path = _active_experiments_path()
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    exp_id = payload.get(task)
    return str(exp_id) if exp_id else None


def set_active_experiment(task: ModelingTask, experiment_id: str) -> None:
    """Mark *experiment_id* as the active artifact source for *task*."""
    path = _active_experiments_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, str] = {}
    if path.exists():
        payload = json.loads(path.read_text())
    payload[task] = experiment_id
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def resolve_task_artifacts_dir(
    task: ModelingTask,
    *,
    experiment_id: str | None = None,
) -> Path:
    """
    Directory for reading task artifacts.

    Resolution order:
      1. explicit *experiment_id*
      2. active experiment for *task* (``active.json``)
      3. latest experiment directory
      4. legacy flat task dir when it contains files
    """
    if experiment_id:
        return task_experiment_dir(experiment_id, task)

    active = get_active_experiment(task)
    if active:
        return task_experiment_dir(active, task)

    experiments = list_experiments()
    if experiments:
        return task_experiment_dir(experiments[-1], task)

    legacy = _LEGACY_TASK_DIRS[task]
    if legacy.exists() and any(legacy.iterdir()):
        return legacy

    raise FileNotFoundError(
        f"No modeling artifacts found for task {task!r}. "
        "Run training or pass experiment_id=..."
    )


def build_task_result_config(
    task: ModelingTask,
    comparison: pd.DataFrame,
    *,
    metric: str,
    higher_is_better: bool,
    experiment_id: str | None = None,
    models_config: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Build a task block for experiment ``config.yaml``.

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
    models = snapshot_task_hyperparameters(task, models_config=models_config)
    if experiment_id:
        models = enrich_models_snapshot(task, experiment_id, models)
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
    task: ModelingTask,
    experiment_id: str,
    *,
    source_files: list[str] | None = None,
) -> Path:
    """
    Copy selected artifact files from an experiment task dir into ``best_model/<task>/``.

    Writes a ``config.yaml`` pointer to the source experiment without removing an
    existing refit ``model.joblib``.
    """
    src_dir = task_experiment_dir(experiment_id, task)
    dst_dir = best_model_task_dir(task)

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
        "task": task,
        "source_experiment": experiment_id,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        **read_experiment_config(experiment_id),
    }
    (dst_dir / CONFIG_FILENAME).write_text(yaml.safe_dump(pointer, sort_keys=False))
    return dst_dir


def _legacy_has_artifacts(task: ModelingTask) -> bool:
    legacy = _LEGACY_TASK_DIRS[task]
    if not legacy.exists():
        return False
    for name in _LEGACY_TASK_ARTIFACTS[task]:
        if (legacy / name).exists():
            return True
    if (legacy / FEATURE_IMPORTANCE_DIRNAME).is_dir() and any(
        (legacy / FEATURE_IMPORTANCE_DIRNAME).iterdir()
    ):
        return True
    if (legacy / "feature_importance.csv").exists():
        return True
    legacy_best = legacy / BEST_MODEL_DIRNAME
    return legacy_best.exists() and any(legacy_best.iterdir())


def migrate_legacy_artifacts(
    experiment_id: str = "exp_001",
    *,
    force: bool = False,
) -> dict[str, Any]:
    """
    Move flat ``play_type/`` and ``yards_gained/`` artifacts into an experiment.

    Evaluation outputs land in ``experiments/<id>/<task>/``. Nested legacy
    ``<task>/best_model/`` moves to ``best_model/<task>/``. Sets ``active.json``
    and writes ``config.yaml`` from task artifacts and static model configs.
    """
    exp_root = EXPERIMENTS_DIR / experiment_id
    if exp_root.exists() and any(exp_root.iterdir()) and not force:
        raise FileExistsError(
            f"experiment {experiment_id!r} already exists; pass force=True to merge"
        )

    migrated: dict[str, list[str]] = {"play_type": [], "yards_gained": []}

    for task in ("play_type", "yards_gained"):
        legacy = _LEGACY_TASK_DIRS[task]
        if not _legacy_has_artifacts(task):
            continue

        dst = task_experiment_dir(experiment_id, task)
        for filename in _LEGACY_TASK_ARTIFACTS[task]:
            src = legacy / filename
            if src.exists():
                shutil.move(str(src), str(dst / filename))
                migrated[task].append(filename)

        legacy_fi_dir = legacy / FEATURE_IMPORTANCE_DIRNAME
        if legacy_fi_dir.is_dir() and any(legacy_fi_dir.iterdir()):
            dst_fi = dst / FEATURE_IMPORTANCE_DIRNAME
            dst_fi.mkdir(parents=True, exist_ok=True)
            for item in legacy_fi_dir.iterdir():
                shutil.move(str(item), str(dst_fi / item.name))
            migrated[task].append(f"{FEATURE_IMPORTANCE_DIRNAME}/")

        legacy_fi_file = legacy / "feature_importance.csv"
        if legacy_fi_file.exists():
            model_key = "xgboost"
            metadata_path = legacy / "metadata.json"
            if metadata_path.exists():
                model_key = str(
                    json.loads(metadata_path.read_text()).get("model_key", model_key)
                )
            dst_fi = dst / FEATURE_IMPORTANCE_DIRNAME
            dst_fi.mkdir(parents=True, exist_ok=True)
            shutil.move(
                str(legacy_fi_file),
                str(dst_fi / f"{model_key}.csv"),
            )
            migrated[task].append(f"{FEATURE_IMPORTANCE_DIRNAME}/{model_key}.csv")

        legacy_best = legacy / BEST_MODEL_DIRNAME
        if legacy_best.exists():
            promoted = best_model_task_dir(task)
            for item in legacy_best.iterdir():
                target = promoted / item.name
                if target.exists():
                    target.unlink()
                shutil.move(str(item), str(target))
            migrated[task].append(f"{BEST_MODEL_DIRNAME}/")

            model_src = promoted / "model.joblib"
            if model_src.exists() and not (dst / "model.joblib").exists():
                shutil.copy2(model_src, dst / "model.joblib")
                migrated[task].append("model.joblib (from best_model)")

        if legacy.exists() and not any(legacy.iterdir()):
            legacy.rmdir()

        set_active_experiment(task, experiment_id)

    config: dict[str, Any] = {
        "experiment_id": experiment_id,
        "migrated_from_legacy": True,
        "migrated_at": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
        "n_folds": 5,
        "tasks": {},
    }
    profile = load_experiment_profile(DEFAULT_PROFILE_PATH)
    with use_experiment_profile(profile):
        for task, metric, higher_is_better in (
            ("play_type", "roc_auc", True),
            ("yards_gained", "rmse", False),
        ):
            comparison_path = (
                task_experiment_dir(experiment_id, task) / "model_comparison.csv"
            )
            if not comparison_path.exists():
                continue
            comparison = pd.read_csv(comparison_path)
            config["tasks"][task] = build_task_result_config(
                task,
                comparison,
                metric=metric,
                higher_is_better=higher_is_better,
                experiment_id=experiment_id,
                models_config=profile.task_models_config(task),
            )
    write_experiment_config(experiment_id, config)

    for task in ("play_type", "yards_gained"):
        if migrated[task]:
            dst = best_model_task_dir(task)
            pointer = {
                "task": task,
                "source_experiment": experiment_id,
                "promoted_at": datetime.now(timezone.utc).isoformat(),
                **read_experiment_config(experiment_id),
            }
            (dst / CONFIG_FILENAME).write_text(
                yaml.safe_dump(pointer, sort_keys=False)
            )

    active_payload: dict[str, str] = {}
    if _active_experiments_path().exists():
        active_payload = json.loads(_active_experiments_path().read_text())

    return {
        "experiment_id": experiment_id,
        "migrated": migrated,
        "active": active_payload,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate legacy flat modeling artifacts into exp_001",
    )
    parser.add_argument(
        "--experiment",
        default="exp_001",
        help="Target experiment id (default: exp_001)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Migrate even when the experiment directory already has files",
    )
    args = parser.parse_args()
    result = migrate_legacy_artifacts(args.experiment, force=args.force)
    print(json.dumps(result, indent=2))
