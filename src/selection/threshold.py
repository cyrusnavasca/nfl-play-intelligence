"""
Task 1 — threshold-based final keep lists.

Usage (from project root):
    python3 -m src.selection.threshold
"""
from __future__ import annotations

from src.selection.embedded import PlayTypeEmbeddedResults
from src.selection.univariate import PlayTypeUnivariateResults
from src.selection.shared.common import (
    _assert_no_drop_always,
    chi2_significant_categoricals,
    load_features_full,
    passes_embedded_gate,
    passes_univariate_gate,
)


def _task1_univariate_pass(
    univariate_results: PlayTypeUnivariateResults,
) -> tuple[list[str], list[str]]:
    numeric = passes_univariate_gate(univariate_results.task1_numeric, mi_col="mi_clf")
    categorical = chi2_significant_categoricals(univariate_results.chi2)
    return numeric, categorical


def build_task1_final(
    univariate_results: PlayTypeUnivariateResults,
    embedded_results: PlayTypeEmbeddedResults,
) -> tuple[list[str], list[str]]:
    """Stage 2 numeric keep + chi-squared categoricals (Task 1)."""
    univariate_numeric, univariate_categorical = _task1_univariate_pass(univariate_results)
    final_numeric = passes_embedded_gate(
        embedded_results.task1, univariate_numeric
    )
    _assert_no_drop_always(final_numeric, "task1 final numeric")
    _assert_no_drop_always(univariate_categorical, "task1 final categorical")
    return sorted(final_numeric), sorted(univariate_categorical)


def main() -> tuple[list[str], list[str]]:
    from src.selection.embedded import run_task1_embedded
    from src.selection.univariate import run_task1_univariate

    df = load_features_full()
    print(f"[INFO] Loaded features_full: {df.shape}")

    univariate_results = run_task1_univariate(df)
    embedded_results = run_task1_embedded(df, univariate_results)
    final_numeric, final_categorical = build_task1_final(
        univariate_results, embedded_results
    )

    print(f"[INFO] Task 1 final: {len(final_numeric)} numeric + {len(final_categorical)} categorical")
    print("\nTask 1 final numeric:")
    for feat in final_numeric:
        print(f"  {feat}")
    print("\nTask 1 final categorical:")
    for feat in final_categorical:
        print(f"  {feat}")

    return final_numeric, final_categorical


if __name__ == "__main__":
    main()
