"""
Phase 6 — end-to-end feature selection orchestrator.

Wires Task 1 and Task 2 selection phases in order, writes all artifacts and
modeling parquets in a single command.

Usage (from project root):
    python3 -m src.selection.run_selection
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.selection.plots import write_selection_plots
from src.selection.play_type.build_dataset import (
    build_task1_dataset,
    validate_task1_dataset,
)
from src.selection.play_type.embedded import (
    run_task1_embedded,
    write_task1_embedded_artifacts,
)
from src.selection.play_type.univariate import (
    run_task1_univariate,
    write_task1_univariate_artifacts,
)
from src.selection.shared.common import load_features_full
from src.selection.shared.feature_schema import (
    FEATURES_FULL_PATH,
    PLAY_TYPE_MODELING_PARQUET_PATH,
    PROCESSED_DIR,
    TARGET_CLF,
    TARGET_REG,
    YARDS_GAINED_MODELING_PARQUET_PATH,
    validate_feature_schema,
)
from src.selection.shared.manifest import (
    combine_embedded_results,
    combine_univariate_results,
    run_threshold_selection,
    write_threshold_selection_artifacts,
)
from src.selection.yards_gained.build_dataset import (
    build_task2_dataset,
    oof_pass_proba_from_manifest,
    validate_task2_dataset,
)
from src.selection.yards_gained.embedded import (
    run_task2_embedded,
    write_task2_embedded_artifacts,
)
from src.selection.yards_gained.univariate import (
    run_task2_univariate,
    write_task2_univariate_artifacts,
)


@dataclass(frozen=True)
class SelectionPipelineResults:
    """Combined outputs from a full selection pipeline run."""

    task1_dataset: pd.DataFrame
    task2_dataset: pd.DataFrame
    selection_manifest: dict[str, object]


def run_selection_pipeline(df: pd.DataFrame) -> SelectionPipelineResults:
    """Run all selection phases and return modeling datasets + manifest."""
    validate_feature_schema(df.columns.tolist())

    print("[Phase 2] Task 1 univariate selection...")
    task1_univariate = run_task1_univariate(df)
    write_task1_univariate_artifacts(task1_univariate)

    print("[Phase 3] Task 1 embedded selection...")
    task1_embedded = run_task1_embedded(df, task1_univariate)
    write_task1_embedded_artifacts(task1_embedded)

    print("[Phase 2] Task 2 univariate selection...")
    task2_univariate = run_task2_univariate(df)
    write_task2_univariate_artifacts(task2_univariate)

    print("[Phase 3] Task 2 embedded selection...")
    task2_embedded = run_task2_embedded(
        df, task2_univariate, task1_embedded.oof_pass_proba
    )
    write_task2_embedded_artifacts(task2_embedded)

    print("[Phase 4] Threshold selection + manifest...")
    threshold_results = run_threshold_selection(
        df,
        task1_univariate,
        task1_embedded,
        task2_univariate,
        task2_embedded,
    )
    write_threshold_selection_artifacts(threshold_results)

    univariate_combined = combine_univariate_results(task1_univariate, task2_univariate)
    embedded_combined = combine_embedded_results(task1_embedded, task2_embedded)

    print("[Phase 6] Writing screening plots...")
    plot_paths = write_selection_plots(univariate_combined, embedded_combined)
    for path in plot_paths:
        print(f"  plot → {path}")

    print("[Phase 5] Building modeling datasets...")
    manifest = threshold_results.selection_manifest
    task1 = build_task1_dataset(df, manifest)
    oof_for_modeling = oof_pass_proba_from_manifest(df, manifest)
    task2 = build_task2_dataset(df, manifest, oof_for_modeling)

    expected_rows = len(df)
    validate_task1_dataset(task1, manifest, df, expected_rows=expected_rows)
    validate_task2_dataset(task2, manifest, expected_rows=expected_rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    task1.to_parquet(PLAY_TYPE_MODELING_PARQUET_PATH, index=False)
    task2.to_parquet(YARDS_GAINED_MODELING_PARQUET_PATH, index=False)

    return SelectionPipelineResults(
        task1_dataset=task1,
        task2_dataset=task2,
        selection_manifest=manifest,
    )


def main() -> SelectionPipelineResults:
    df = load_features_full()
    print(f"[INFO] Loaded features_full: {df.shape} from {FEATURES_FULL_PATH}")

    results = run_selection_pipeline(df)
    manifest = results.selection_manifest
    t1 = results.task1_dataset
    t2 = results.task2_dataset
    t1_final = manifest["task1"]["final"]  # type: ignore[index]
    t2_final = manifest["task2"]["final"]  # type: ignore[index]

    print("\n=== Selection pipeline complete ===")
    print(f"Rows: {len(df):,}")
    print(
        f"Task 1 final: {len(t1_final['numeric'])} numeric + "
        f"{len(t1_final['categorical'])} categorical → {t1.shape[1]} cols"
    )
    print(
        f"Task 2 final: {len(t2_final['numeric'])} numeric + "
        f"{len(t2_final['generated'])} generated → {t2.shape[1]} cols"
    )
    print(f"  play_type_modeling → {PLAY_TYPE_MODELING_PARQUET_PATH}  shape={t1.shape}")
    print(f"  yards_gained_modeling → {YARDS_GAINED_MODELING_PARQUET_PATH}  shape={t2.shape}")
    print(f"  target Task 1: {TARGET_CLF}  |  target Task 2: {TARGET_REG}")

    return results


if __name__ == "__main__":
    main()
