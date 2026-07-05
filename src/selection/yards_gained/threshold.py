"""
Task 2 — threshold-based final keep lists.

Usage (from project root):
    python3 -m src.selection.yards_gained.threshold
"""
from __future__ import annotations

from src.selection.shared.common import (
    _assert_no_drop_always,
    load_features_full,
    passes_embedded_gate,
    passes_univariate_gate,
)
from src.selection.yards_gained.embedded import YardsGainedEmbeddedResults
from src.selection.yards_gained.univariate import YardsGainedUnivariateResults


def _task2_univariate_pass(univariate_results: YardsGainedUnivariateResults) -> list[str]:
    return passes_univariate_gate(univariate_results.task2_numeric, mi_col="mi_reg")


def build_task2_final(
    univariate_results: YardsGainedUnivariateResults,
    embedded_results: YardsGainedEmbeddedResults,
) -> list[str]:
    """Stage 2 numeric keep plus always-included ``pred_pass_proba``."""
    univariate_numeric = _task2_univariate_pass(univariate_results)
    final_numeric = passes_embedded_gate(
        embedded_results.task2, univariate_numeric
    )
    _assert_no_drop_always(final_numeric, "task2 final numeric")
    return sorted(final_numeric)


def main() -> list[str]:
    from src.selection.play_type.embedded import run_task1_embedded
    from src.selection.play_type.univariate import run_task1_univariate
    from src.selection.yards_gained.embedded import run_task2_embedded
    from src.selection.yards_gained.univariate import run_task2_univariate

    df = load_features_full()
    print(f"[INFO] Loaded features_full: {df.shape}")

    task1_univariate = run_task1_univariate(df)
    task1_embedded = run_task1_embedded(df, task1_univariate)
    task2_univariate = run_task2_univariate(df)
    task2_embedded = run_task2_embedded(
        df, task2_univariate, task1_embedded.oof_pass_proba
    )
    final_numeric = build_task2_final(task2_univariate, task2_embedded)

    print(f"[INFO] Task 2 final: {len(final_numeric)} numeric + 1 generated")
    print("\nTask 2 final numeric:")
    for feat in final_numeric:
        print(f"  {feat}")

    return final_numeric


if __name__ == "__main__":
    main()
