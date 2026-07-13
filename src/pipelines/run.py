"""
End-to-end modeling orchestrator.

Runs play-type CV and optional best-model persistence in a single command.

Usage (from project root):
    python3 -m src.pipelines.run --config configs/models/default.yaml
    python3 -m src.pipelines.run --config configs/models/xgboost_tuned.yaml --persist-best
    python3 -m src.pipelines.run --config configs/models/default.yaml --experiment exp_003
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


def _write_experiment_notes(exp_id: str, note: str) -> None:
    """Write a human-readable notes.md describing this run's intent."""
    notes_path = resolve_artifacts_dir(experiment_id=exp_id) / "notes.md"
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text(f"# {exp_id} — notes\n\n{note}\n", encoding="utf-8")


def _run_tuning(profile: ExperimentProfile, exp_id: str) -> dict:
    """
    Run an Optuna study for every model with a search space and update the
    profile's fixed hyperparameters in place with the best params found.

    Writes per-model trial CSVs to the experiment dir and returns a summary.
    """
    from src.data.loaders import load_play_type_dataset
    from src.pipelines.tune import DEFAULT_CV_FOLDS, DEFAULT_N_TRIALS, tune_model

    tune_cfg = profile.tune
    n_trials = int(tune_cfg.get("n_trials", DEFAULT_N_TRIALS))
    cv_folds = int(tune_cfg.get("cv_folds", DEFAULT_CV_FOLDS))
    subsample_rows = tune_cfg.get("subsample_rows")

    X, y = load_play_type_dataset()
    out_dir = resolve_artifacts_dir(experiment_id=exp_id)
    summary: dict[str, dict] = {}

    for model_key in profile.model_keys():
        if not profile.has_search_space(model_key):
            continue
        print(f"[tune] {model_key}: {n_trials} trials (TPE)...")
        result = tune_model(
            model_key,
            X,
            y,
            fixed_params=profile.model_hyperparameters(model_key),
            search_space=profile.search_space(model_key),
            n_trials=n_trials,
            cv_folds=cv_folds,
            seed=profile.seed,
            subsample_rows=subsample_rows,
        )
        # Update the profile in place so downstream CV uses the best params.
        profile.models[model_key].clear()
        profile.models[model_key].update(result.best_params)

        result.trials.to_csv(out_dir / f"tuning_{model_key}.csv", index=False)
        summary[model_key] = {
            "n_trials": result.n_trials,
            "best_value": result.best_value,
            "best_params": result.best_params,
        }
        print(f"[tune] {model_key}: best roc_auc={result.best_value:.4f}")

    return summary


def run_pipeline(
    *,
    config_path: Path | str | None = None,
    profile: ExperimentProfile | None = None,
    skip_training: bool = False,
    persist_best: bool = False,
    experiment_id: str | None = None,
    note: str | None = None,
    tune: bool = False,
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
        if note:
            snapshot["notes"] = note
        write_experiment_config(exp_id, snapshot)
        if note:
            _write_experiment_notes(exp_id, note)

        print(f"Experiment:    {exp_id}")
        if note:
            print(f"Notes:         {note}")
        print(f"Profile:       {profile.name}")
        if profile.source_path is not None:
            print(f"Config:        {profile.source_path}")
        print("[1/3] Validating modeling parquet...")
        validate_modeling_parquet(PLAY_TYPE_MODELING_PARQUET_PATH)
        print("  OK")

        if tune:
            tuning_summary = _run_tuning(profile, exp_id)
            if tuning_summary:
                update_experiment_config(exp_id, {"tuning": tuning_summary})

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
    parser.add_argument(
        "--note",
        help="Free-text description of this run; saved to config.yaml (notes) "
        "and experiments/<id>/notes.md",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna search for models with a 'search' block before CV",
    )
    return parser.parse_args()


def main() -> pd.DataFrame | None:
    args = _parse_args()
    comparison = run_pipeline(
        config_path=args.config,
        skip_training=args.skip_training,
        persist_best=args.persist_best,
        experiment_id=args.experiment,
        note=args.note,
        tune=args.tune,
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
