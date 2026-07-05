"""
Task 2 — univariate numeric screening for yards_gained.

Usage (from project root):
    python3 -m src.selection.yards_gained.univariate
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.selection.shared.common import (
    ensure_artifacts_dir,
    load_features_full,
    median_impute_numeric,
    task2_numeric_screening,
)
from src.selection.shared.feature_schema import (
    TASK2_NUMERIC_SCREENING_CSV,
    validate_feature_schema,
)


@dataclass(frozen=True)
class YardsGainedUnivariateResults:
    """Container for Task 2 univariate selection table."""

    task2_numeric: pd.DataFrame


def run_task2_univariate(df: pd.DataFrame) -> YardsGainedUnivariateResults:
    """Run Task 2 univariate numeric screening."""
    validate_feature_schema(df.columns.tolist())

    X_imp = median_impute_numeric(df)
    task2_df = task2_numeric_screening(df, X_imp)

    return YardsGainedUnivariateResults(task2_numeric=task2_df)


def write_task2_univariate_artifacts(results: YardsGainedUnivariateResults) -> None:
    """Write Task 2 numeric screening CSV."""
    ensure_artifacts_dir()
    results.task2_numeric.reset_index().to_csv(TASK2_NUMERIC_SCREENING_CSV, index=False)


def main() -> YardsGainedUnivariateResults:
    df = load_features_full()
    print(f"[INFO] Loaded features_full: {df.shape}")

    results = run_task2_univariate(df)
    write_task2_univariate_artifacts(results)

    print(
        f"[DONE] task2 numeric → {TASK2_NUMERIC_SCREENING_CSV} "
        f"({len(results.task2_numeric)} features)"
    )

    print("\nTop 10 Task 2 (combined_score):")
    print(
        results.task2_numeric[["combined_score", "spearman_rho", "mi_reg"]]
        .head(10)
        .to_string(float_format="{:.4f}".format)
    )

    return results


if __name__ == "__main__":
    main()
