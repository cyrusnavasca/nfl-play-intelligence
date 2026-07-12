"""
End-to-end feature selection orchestrator.

Flow (no automated filter stage):
    manual feature config → embedded (LightGBM gain) → embedded gate → parquet

Usage (from project root):
    python3 -m src.selection.run_selection
    python3 -m src.selection.run_selection --features configs/features/my_trial.yaml
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.selection.build_dataset import (
    build_task1_dataset,
    validate_task1_dataset,
)
from src.selection.embedded import (
    run_embedded,
    write_embedded_artifacts,
)
from src.selection.feature_config import FeatureConfig, load_feature_config
from src.selection.shared.common import load_features_full
from src.selection.shared.feature_schema import (
    DEFAULT_FEATURES_CONFIG,
    FEATURES_FULL_PATH,
    PLAY_TYPE_MODELING_PARQUET_PATH,
    PROCESSED_DIR,
    TARGET_CLF,
    validate_feature_schema,
)
from src.selection.shared.manifest import (
    run_threshold_selection,
    write_threshold_selection_artifacts,
)


@dataclass(frozen=True)
class SelectionPipelineResults:
    """Combined outputs from a full selection pipeline run."""

    task1_dataset: pd.DataFrame
    selection_manifest: dict[str, object]


def run_selection_pipeline(
    df: pd.DataFrame,
    feature_config: FeatureConfig,
) -> SelectionPipelineResults:
    """Run all selection phases and return the modeling dataset + manifest."""
    validate_feature_schema(df.columns.tolist())

    print("[Phase 1] Embedded selection on manually selected features...")
    task1_embedded = run_embedded(df, feature_config.numeric, feature_config.categorical)
    write_embedded_artifacts(task1_embedded)

    print("[Phase 2] Embedded gate + manifest...")
    threshold_results = run_threshold_selection(df, feature_config, task1_embedded)
    write_threshold_selection_artifacts(threshold_results)

    print("[Phase 3] Building modeling dataset...")
    manifest = threshold_results.selection_manifest
    task1 = build_task1_dataset(df, manifest)

    expected_rows = len(df)
    validate_task1_dataset(task1, manifest, df, expected_rows=expected_rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    task1.to_parquet(PLAY_TYPE_MODELING_PARQUET_PATH, index=False)

    return SelectionPipelineResults(
        task1_dataset=task1,
        selection_manifest=manifest,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end play-type feature selection",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES_CONFIG,
        help=f"Manual feature config YAML (default: {DEFAULT_FEATURES_CONFIG})",
    )
    return parser.parse_args()


def main() -> SelectionPipelineResults:
    args = _parse_args()

    df = load_features_full()
    print(f"[INFO] Loaded features_full: {df.shape} from {FEATURES_FULL_PATH}")

    feature_config = load_feature_config(args.features)
    print(
        f"[INFO] Feature config '{feature_config.name}': "
        f"{len(feature_config.numeric)} numeric + "
        f"{len(feature_config.categorical)} categorical "
        f"(embedded threshold={feature_config.embedded_importance_threshold})"
    )

    results = run_selection_pipeline(df, feature_config)
    manifest = results.selection_manifest
    t1 = results.task1_dataset
    t1_final = manifest["task1"]["final"]  # type: ignore[index]

    print("\n=== Selection pipeline complete ===")
    print(f"Rows: {len(df):,}")
    print(
        f"Final: {len(t1_final['numeric'])} numeric + "
        f"{len(t1_final['categorical'])} categorical → {t1.shape[1]} cols"
    )
    print(f"  play_type_modeling → {PLAY_TYPE_MODELING_PARQUET_PATH}  shape={t1.shape}")
    print(f"  target: {TARGET_CLF}")

    return results


if __name__ == "__main__":
    main()
