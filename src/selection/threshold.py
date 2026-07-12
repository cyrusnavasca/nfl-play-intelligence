"""
Embedded-gate final keep lists.

Given the manually selected features and their cross-validated embedded gain
importance, keep those meeting the config's ``embedded_importance_threshold``.
This is the only automated pruning stage (filter screening was removed).

Usage (from project root):
    python3 -m src.selection.threshold
"""
from __future__ import annotations

from src.selection.embedded import PlayTypeEmbeddedResults
from src.selection.shared.common import (
    _assert_no_drop_always,
    categorical_passes_embedded_gate,
    load_features_full,
    passes_embedded_gate,
)
from src.selection.shared.feature_schema import EMBEDDED_THRESHOLD


def build_task1_final(
    embedded_results: PlayTypeEmbeddedResults,
    numeric_cols: list[str],
    cat_cols: list[str],
    *,
    threshold: float = EMBEDDED_THRESHOLD,
) -> tuple[list[str], list[str]]:
    """Numeric + categorical features passing the embedded gain gate."""
    final_numeric = passes_embedded_gate(
        embedded_results.task1, numeric_cols, threshold=threshold
    )
    final_categorical = categorical_passes_embedded_gate(
        embedded_results.task1, cat_cols, threshold=threshold
    )
    _assert_no_drop_always(final_numeric, "final numeric")
    _assert_no_drop_always(final_categorical, "final categorical")
    return sorted(final_numeric), sorted(final_categorical)


def main() -> tuple[list[str], list[str]]:
    from src.selection.embedded import run_embedded
    from src.selection.feature_config import load_feature_config

    df = load_features_full()
    print(f"[INFO] Loaded features_full: {df.shape}")

    cfg = load_feature_config()
    embedded_results = run_embedded(df, cfg.numeric, cfg.categorical)
    final_numeric, final_categorical = build_task1_final(
        embedded_results,
        cfg.numeric,
        cfg.categorical,
        threshold=cfg.embedded_importance_threshold,
    )

    print(
        f"[INFO] Final: {len(final_numeric)} numeric + "
        f"{len(final_categorical)} categorical "
        f"(threshold={cfg.embedded_importance_threshold})"
    )
    print("\nFinal numeric:")
    for feat in final_numeric:
        print(f"  {feat}")
    print("\nFinal categorical:")
    for feat in final_categorical:
        print(f"  {feat}")

    return final_numeric, final_categorical


if __name__ == "__main__":
    main()
