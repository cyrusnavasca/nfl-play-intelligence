"""
End-to-end modeling orchestrator.

Runs play-type CV + OOF export, yards-gained holdout evaluation, and optional
best-model persistence in a single command.

Usage (from project root):
    python3 -m src.pipelines.run --config configs/xgboost_baseline.yaml
    python3 -m src.pipelines.run --config configs/xgboost_tuned.yaml --persist-best
    python3 -m src.pipelines.run --config configs/xgboost_baseline.yaml --experiment exp_003
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.schema import (
    BEST_MODEL_DIR,
    EXPERIMENTS_DIR,
    MODELING_ARTIFACTS_DIR,
    PLAY_TYPE_MODELING_PARQUET_PATH,
    YARDS_GAINED_MODELING_PARQUET_PATH,
    validate_modeling_parquet,
)
from src.evaluation.model_selection import select_best_model
from src.pipelines.play_type.oof import export_oof_predictions
from src.pipelines.play_type.predict import refit_best_classifier
from src.pipelines.play_type.train import (
    MODEL_COMPARISON_FILENAME as PLAY_TYPE_COMPARISON_FILENAME,
    train_play_type,
)
from src.pipelines.yards_gained.predict import refit_best_regressor
from src.pipelines.yards_gained.train import (
    MODEL_COMPARISON_FILENAME as YARDS_GAINED_COMPARISON_FILENAME,
    train_yards_gained,
)
from src.utils.experiment_profile import (
    DEFAULT_PROFILE_PATH,
    ExperimentProfile,
    load_experiment_profile,
    use_experiment_profile,
)
from src.utils.experiments import (
    allocate_experiment_id,
    resolve_task_artifacts_dir,
    set_active_experiment,
    update_experiment_config,
    write_experiment_config,
)

__all__ = ["run_pipeline"]


def run_pipeline(
    *,
    config_path: Path | str | None = None,
    profile: ExperimentProfile | None = None,
    skip_play_type: bool = False,
    skip_yards_gained: bool = False,
    persist_best: bool = False,
    experiment_id: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run the two-step modeling pipeline end to end.

    Order: validate parquets → play-type CV → OOF export → yards holdout →
    experiment ``config.yaml``. When *persist_best* is True, refit and save both
    best estimators via each task's ``predict`` module.

    All task artifacts for a run share one *experiment_id* under
    ``artifacts/modeling/experiments/<id>/``.
    """
    if profile is None:
        profile = load_experiment_profile(config_path or DEFAULT_PROFILE_PATH)

    started_at = datetime.now(timezone.utc).isoformat()
    results: dict[str, pd.DataFrame] = {}

    exp_id = experiment_id or allocate_experiment_id()
    run_play_type = not skip_play_type and profile.has_task("play_type")
    run_yards_gained = not skip_yards_gained and profile.has_task("yards_gained")
    should_persist = persist_best or profile.persist_best
    play_type_exp = exp_id if run_play_type else profile.play_type_experiment

    with use_experiment_profile(profile):
        snapshot = profile.to_snapshot_base(exp_id)
        snapshot["started_at"] = started_at
        write_experiment_config(exp_id, snapshot)

        print(f"Experiment:    {exp_id}")
        print(f"Profile:       {profile.name}")
        if profile.source_path is not None:
            print(f"Config:        {profile.source_path}")
        print("[1/5] Validating modeling parquets...")
        validate_modeling_parquet(PLAY_TYPE_MODELING_PARQUET_PATH, "play_type")
        validate_modeling_parquet(YARDS_GAINED_MODELING_PARQUET_PATH, "yards_gained")
        print("  OK")

        if run_play_type:
            print("[2/5] Play-type cross-validation...")
            results["play_type"], _ = train_play_type(experiment_id=exp_id)
            print("[3/5] Exporting OOF pass probabilities...")
            export_oof_predictions(experiment_id=exp_id)
        else:
            print("[2/5] Skipping play-type training")
            print("[3/5] Skipping OOF export")
            comparison_path = (
                resolve_task_artifacts_dir("play_type", experiment_id=exp_id)
                / PLAY_TYPE_COMPARISON_FILENAME
            )
            if comparison_path.exists():
                results["play_type"] = pd.read_csv(comparison_path)

        if run_yards_gained:
            print("[4/5] Yards-gained holdout evaluation...")
            results["yards_gained"], _ = train_yards_gained(
                experiment_id=exp_id,
                play_type_experiment_id=play_type_exp,
            )
        else:
            print("[4/5] Skipping yards-gained training")
            comparison_path = (
                resolve_task_artifacts_dir("yards_gained", experiment_id=exp_id)
                / YARDS_GAINED_COMPARISON_FILENAME
            )
            if comparison_path.exists():
                results["yards_gained"] = pd.read_csv(comparison_path)

        config_patch: dict[str, Any] = {
            "skipped": {
                "play_type": not run_play_type,
                "yards_gained": not run_yards_gained,
            },
            "persisted_best": should_persist,
        }
        if run_yards_gained and run_play_type:
            config_patch["play_type_experiment"] = exp_id
        elif play_type_exp is not None:
            config_patch["play_type_experiment"] = play_type_exp

        if should_persist:
            print("[5/5] Persisting best models...")
            if "play_type" in results:
                refit_best_classifier(experiment_id=exp_id)
            if "yards_gained" in results:
                refit_best_regressor(
                    experiment_id=exp_id,
                    play_type_experiment_id=play_type_exp,
                )
        else:
            print("[5/5] Skipping best-model persistence (use --persist-best to save)")

        update_experiment_config(exp_id, config_patch)

        if "play_type" in results:
            set_active_experiment("play_type", exp_id)
        if "yards_gained" in results:
            set_active_experiment("yards_gained", exp_id)

    print(f"\nExperiment config → {EXPERIMENTS_DIR / exp_id / 'config.yaml'}")
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end NFL play modeling pipeline",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PROFILE_PATH,
        help=f"Experiment profile YAML (default: {DEFAULT_PROFILE_PATH})",
    )
    parser.add_argument(
        "--experiment",
        help="Experiment id for this run (default: next exp_NNN)",
    )
    parser.add_argument(
        "--persist-best",
        action="store_true",
        help="Refit and save best play-type classifier and yards-gained regressor",
    )
    parser.add_argument(
        "--skip-play-type",
        action="store_true",
        help="Skip play-type CV and OOF export (requires existing artifacts)",
    )
    parser.add_argument(
        "--skip-yards-gained",
        action="store_true",
        help="Skip yards-gained holdout evaluation (requires existing artifacts)",
    )
    return parser.parse_args()


def main() -> dict[str, pd.DataFrame]:
    args = _parse_args()
    results = run_pipeline(
        config_path=args.config,
        skip_play_type=args.skip_play_type,
        skip_yards_gained=args.skip_yards_gained,
        persist_best=args.persist_best,
        experiment_id=args.experiment,
    )

    print("\n=== Modeling pipeline complete ===")
    if "play_type" in results:
        comparison = results["play_type"]
        best_model = select_best_model(
            comparison,
            "roc_auc",
            higher_is_better=True,
        )
        roc_auc = float(
            comparison.loc[comparison["model"] == best_model, "roc_auc_mean"].iloc[0]
        )
        print(f"Play type:     best={best_model}  roc_auc={roc_auc:.4f}")
    if "yards_gained" in results:
        comparison = results["yards_gained"]
        best_model = select_best_model(
            comparison,
            "rmse",
            higher_is_better=False,
        )
        rmse = float(
            comparison.loc[comparison["model"] == best_model, "rmse_mean"].iloc[0]
        )
        print(f"Yards gained:  best={best_model}  rmse={rmse:.4f}")
    print(f"Experiments:   {EXPERIMENTS_DIR}")
    print(f"Best models:   {BEST_MODEL_DIR}")
    print(f"Artifacts:     {MODELING_ARTIFACTS_DIR}")

    return results


if __name__ == "__main__":
    main()
