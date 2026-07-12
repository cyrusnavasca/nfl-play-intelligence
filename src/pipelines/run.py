"""
End-to-end modeling orchestrator.

Runs play-type CV and optional best-model persistence in a single command.

Usage (from project root):
    python3 -m src.pipelines.run --config configs/default.yaml
    python3 -m src.pipelines.run --config configs/xgboost_tuned.yaml --persist-best
    python3 -m src.pipelines.run --config configs/default.yaml --experiment exp_003
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.schema import (
    BEST_MODEL_DIR,
    EXPERIMENTS_DIR,
    MODELING_ARTIFACTS_DIR,
    PLAY_TYPE_MODELING_PARQUET_PATH,
    validate_modeling_parquet,
)
from src.evaluation.model_selection import select_best_model
from src.pipelines.predict import refit_best_classifier
from src.pipelines.train import (
    MODEL_COMPARISON_FILENAME,
    train_play_type,
)
from src.utils.experiment_profile import (
    DEFAULT_PROFILE_PATH,
    ExperimentProfile,
    load_experiment_profile,
    use_experiment_profile,
)
from src.utils.experiments import (
    allocate_experiment_id,
    resolve_artifacts_dir,
    set_active_experiment,
    update_experiment_config,
    write_experiment_config,
)

__all__ = ["run_pipeline"]


def run_pipeline(
    *,
    config_path: Path | str | None = None,
    profile: ExperimentProfile | None = None,
    skip_training: bool = False,
    persist_best: bool = False,
    experiment_id: str | None = None,
) -> pd.DataFrame | None:
    """
    Run the play-type modeling pipeline end to end.

    Order: validate parquet → play-type CV → experiment ``config.yaml``. When
    *persist_best* is True, refit and save the best estimator via
    ``pipelines.predict``.

    Artifacts for a run live under ``artifacts/modeling/experiments/<id>/``.
    Returns the model-comparison frame (or None when training is skipped and no
    prior artifacts exist).
    """
    if profile is None:
        profile = load_experiment_profile(config_path or DEFAULT_PROFILE_PATH)

    started_at = datetime.now(timezone.utc).isoformat()
    comparison: pd.DataFrame | None = None

    exp_id = experiment_id or allocate_experiment_id()
    should_persist = persist_best or profile.persist_best

    with use_experiment_profile(profile):
        snapshot = profile.to_snapshot_base(exp_id)
        snapshot["started_at"] = started_at
        write_experiment_config(exp_id, snapshot)

        print(f"Experiment:    {exp_id}")
        print(f"Profile:       {profile.name}")
        if profile.source_path is not None:
            print(f"Config:        {profile.source_path}")
        print("[1/3] Validating modeling parquet...")
        validate_modeling_parquet(PLAY_TYPE_MODELING_PARQUET_PATH)
        print("  OK")

        if not skip_training:
            print("[2/3] Play-type cross-validation...")
            comparison, _ = train_play_type(experiment_id=exp_id)
        else:
            print("[2/3] Skipping training")
            comparison_path = (
                resolve_artifacts_dir(experiment_id=exp_id) / MODEL_COMPARISON_FILENAME
            )
            if comparison_path.exists():
                comparison = pd.read_csv(comparison_path)

        if should_persist:
            print("[3/3] Persisting best model...")
            if comparison is not None:
                refit_best_classifier(experiment_id=exp_id)
        else:
            print("[3/3] Skipping best-model persistence (use --persist-best to save)")

        update_experiment_config(
            exp_id,
            {"skipped_training": skip_training, "persisted_best": should_persist},
        )

        if comparison is not None:
            set_active_experiment(exp_id)

    print(f"\nExperiment config → {EXPERIMENTS_DIR / exp_id / 'config.yaml'}")
    return comparison


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end NFL play-type modeling pipeline",
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
        help="Refit and save the best play-type classifier",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip CV (requires existing artifacts)",
    )
    return parser.parse_args()


def main() -> pd.DataFrame | None:
    args = _parse_args()
    comparison = run_pipeline(
        config_path=args.config,
        skip_training=args.skip_training,
        persist_best=args.persist_best,
        experiment_id=args.experiment,
    )

    print("\n=== Modeling pipeline complete ===")
    if comparison is not None:
        best_model = select_best_model(
            comparison,
            "roc_auc",
            higher_is_better=True,
        )
        roc_auc = float(
            comparison.loc[comparison["model"] == best_model, "roc_auc_mean"].iloc[0]
        )
        print(f"Play type:     best={best_model}  roc_auc={roc_auc:.4f}")
    print(f"Experiments:   {EXPERIMENTS_DIR}")
    print(f"Best models:   {BEST_MODEL_DIR}")
    print(f"Artifacts:     {MODELING_ARTIFACTS_DIR}")

    return comparison


if __name__ == "__main__":
    main()
