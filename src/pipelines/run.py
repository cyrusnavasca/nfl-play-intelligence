"""
End-to-end modeling orchestrator.

Runs play-type CV + OOF export, yards-gained holdout evaluation, and optional
best-model persistence in a single command.

Usage (from project root):
    python3 -m src.pipelines.run
    python3 -m src.pipelines.run --persist-best
    python3 -m src.pipelines.run --skip-play-type
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.data.schema import (
    MODELING_ARTIFACTS_DIR,
    N_FOLDS,
    PLAY_TYPE_ARTIFACTS_DIR,
    PLAY_TYPE_MODELING_PARQUET_PATH,
    SEED,
    YARDS_GAINED_ARTIFACTS_DIR,
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
from src.utils.io import write_run_summary

__all__ = ["run_pipeline"]


def _metric_snapshot(comparison: pd.DataFrame, model_key: str) -> dict[str, float]:
    matches = comparison.loc[comparison["model"] == model_key]
    if matches.empty:
        raise KeyError(f"model {model_key!r} not in comparison table")
    row = matches.iloc[0]
    return {col: float(row[col]) for col in comparison.columns if col != "model"}


def _task_summary(
    comparison: pd.DataFrame,
    *,
    metric: str,
    higher_is_better: bool,
) -> dict[str, Any]:
    best_model = select_best_model(
        comparison,
        metric,
        higher_is_better=higher_is_better,
    )
    return {
        "best_model": best_model,
        "metrics": _metric_snapshot(comparison, best_model),
    }


def run_pipeline(
    *,
    skip_play_type: bool = False,
    skip_yards_gained: bool = False,
    persist_best: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Run the two-step modeling pipeline end to end.

    Order: validate parquets → play-type CV → OOF export → yards holdout →
    ``run_summary.json``. When *persist_best* is True, refit and save both
    best estimators via each task's ``predict`` module.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    results: dict[str, pd.DataFrame] = {}

    print("[1/5] Validating modeling parquets...")
    validate_modeling_parquet(PLAY_TYPE_MODELING_PARQUET_PATH, "play_type")
    validate_modeling_parquet(YARDS_GAINED_MODELING_PARQUET_PATH, "yards_gained")
    print("  OK")

    if not skip_play_type:
        print("[2/5] Play-type cross-validation...")
        results["play_type"] = train_play_type()
        print("[3/5] Exporting OOF pass probabilities...")
        export_oof_predictions()
    else:
        print("[2/5] Skipping play-type training")
        print("[3/5] Skipping OOF export")
        comparison_path = PLAY_TYPE_ARTIFACTS_DIR / PLAY_TYPE_COMPARISON_FILENAME
        if comparison_path.exists():
            results["play_type"] = pd.read_csv(comparison_path)

    if not skip_yards_gained:
        print("[4/5] Yards-gained holdout evaluation...")
        results["yards_gained"] = train_yards_gained()
    else:
        print("[4/5] Skipping yards-gained training")
        comparison_path = YARDS_GAINED_ARTIFACTS_DIR / YARDS_GAINED_COMPARISON_FILENAME
        if comparison_path.exists():
            results["yards_gained"] = pd.read_csv(comparison_path)

    summary: dict[str, Any] = {
        "started_at": started_at,
        "seed": SEED,
        "n_folds": N_FOLDS,
        "skipped": {
            "play_type": skip_play_type,
            "yards_gained": skip_yards_gained,
        },
    }

    if "play_type" in results:
        summary["play_type"] = _task_summary(
            results["play_type"],
            metric="roc_auc",
            higher_is_better=True,
        )
    if "yards_gained" in results:
        summary["yards_gained"] = _task_summary(
            results["yards_gained"],
            metric="rmse",
            higher_is_better=False,
        )

    if persist_best:
        print("[5/5] Persisting best models...")
        if "play_type" in results:
            refit_best_classifier()
        if "yards_gained" in results:
            refit_best_regressor()
        summary["persisted_best"] = True
    else:
        print("[5/5] Skipping best-model persistence (use --persist-best to save)")
        summary["persisted_best"] = False

    summary_path = write_run_summary(summary)
    print(f"\nRun summary → {summary_path}")
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end NFL play modeling pipeline",
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
        skip_play_type=args.skip_play_type,
        skip_yards_gained=args.skip_yards_gained,
        persist_best=args.persist_best,
    )

    print("\n=== Modeling pipeline complete ===")
    if "play_type" in results:
        play_summary = _task_summary(
            results["play_type"],
            metric="roc_auc",
            higher_is_better=True,
        )
        metrics = play_summary["metrics"]
        print(
            f"Play type:     best={play_summary['best_model']}  "
            f"roc_auc={metrics['roc_auc_mean']:.4f}"
        )
    if "yards_gained" in results:
        yards_summary = _task_summary(
            results["yards_gained"],
            metric="rmse",
            higher_is_better=False,
        )
        metrics = yards_summary["metrics"]
        print(
            f"Yards gained:  best={yards_summary['best_model']}  "
            f"rmse={metrics['rmse_mean']:.4f}"
        )
    print(f"Artifacts:     {MODELING_ARTIFACTS_DIR}")

    return results


if __name__ == "__main__":
    main()
