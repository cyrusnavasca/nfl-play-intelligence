"""
Phase 6 — end-to-end feature selection orchestrator.

Wires the play-type selection phases in order, writes all artifacts and the
modeling parquet in a single command.

Usage (from project root):
    python3 -m src.selection.run_selection
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.selection.build_dataset import (
    build_task1_dataset,
    validate_task1_dataset,
)
from src.selection.embedded import (
    run_task1_embedded,
    write_task1_embedded_artifacts,
)
from src.selection.univariate import (
    run_task1_univariate,
    write_task1_univariate_artifacts,
)
from src.selection.shared.common import load_features_full
from src.selection.shared.feature_schema import (
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


def run_selection_pipeline(df: pd.DataFrame) -> SelectionPipelineResults:
    """Run all selection phases and return the modeling dataset + manifest."""
    validate_feature_schema(df.columns.tolist())

    print("[Phase 2] Task 1 univariate selection...")
    task1_univariate = run_task1_univariate(df)
    write_task1_univariate_artifacts(task1_univariate)

    print("[Phase 3] Task 1 embedded selection...")
    task1_embedded = run_task1_embedded(df, task1_univariate)
    write_task1_embedded_artifacts(task1_embedded)

    print("[Phase 4] Threshold selection + manifest...")
    threshold_results = run_threshold_selection(
        df,
        task1_univariate,
        task1_embedded,
    )
    write_threshold_selection_artifacts(threshold_results)

    print("[Phase 5] Building modeling dataset...")
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


def main() -> SelectionPipelineResults:
    df = load_features_full()
    print(f"[INFO] Loaded features_full: {df.shape} from {FEATURES_FULL_PATH}")

    results = run_selection_pipeline(df)
    manifest = results.selection_manifest
    t1 = results.task1_dataset
    t1_final = manifest["task1"]["final"]  # type: ignore[index]

    print("\n=== Selection pipeline complete ===")
    print(f"Rows: {len(df):,}")
    print(
        f"Task 1 final: {len(t1_final['numeric'])} numeric + "
        f"{len(t1_final['categorical'])} categorical → {t1.shape[1]} cols"
    )
    print(f"  play_type_modeling → {PLAY_TYPE_MODELING_PARQUET_PATH}  shape={t1.shape}")
    print(f"  target: {TARGET_CLF}")

    return results


if __name__ == "__main__":
    main()
