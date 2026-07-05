"""
Task 1 — univariate feature selection (chi-squared + numeric screening).

Usage (from project root):
    python3 -m src.selection.play_type.univariate
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.stats import chi2_contingency

from src.selection.shared.common import (
    ensure_artifacts_dir,
    load_features_full,
    median_impute_numeric,
    task1_numeric_screening,
)
from src.selection.shared.feature_schema import (
    CHI2_FEATURES,
    CHI2_P_THRESHOLD,
    CHI2_SCREENING_CSV,
    DROP_ALWAYS,
    TARGET_CLF,
    TASK1_NUMERIC_SCREENING_CSV,
    validate_feature_schema,
)


@dataclass(frozen=True)
class PlayTypeUnivariateResults:
    """Container for Task 1 univariate selection tables."""

    chi2: pd.DataFrame
    task1_numeric: pd.DataFrame


def _assert_no_excluded(columns: list[str], context: str) -> None:
    leaked = sorted(set(columns) & set(DROP_ALWAYS))
    if leaked:
        raise ValueError(f"{context}: excluded columns in screening: {leaked}")


def chi2_categorical_screening(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chi-squared test of independence for each CHI2 feature vs. play_type.

    Returns a DataFrame sorted by chi2 descending with a significant flag.
    """
    _assert_no_excluded(CHI2_FEATURES, "chi2 screening")

    results: list[dict[str, object]] = []
    for col in CHI2_FEATURES:
        sub = df[[col, TARGET_CLF]].dropna()
        ct = pd.crosstab(sub[col], sub[TARGET_CLF])
        chi2_stat, p_val, dof, _ = chi2_contingency(ct, correction=False)
        results.append(
            {
                "feature": col,
                "n_categories": sub[col].nunique(),
                "n_rows": len(sub),
                "chi2": chi2_stat,
                "dof": dof,
                "p_value": p_val,
            }
        )

    chi2_df = (
        pd.DataFrame(results)
        .sort_values("chi2", ascending=False)
        .reset_index(drop=True)
    )
    chi2_df["significant"] = chi2_df["p_value"].apply(
        lambda p: "✓" if p < CHI2_P_THRESHOLD else "✗"
    )
    return chi2_df


def run_task1_univariate(df: pd.DataFrame) -> PlayTypeUnivariateResults:
    """Run Task 1 univariate selection (chi-squared + numeric screening)."""
    validate_feature_schema(df.columns.tolist())

    chi2_df = chi2_categorical_screening(df)
    X_imp = median_impute_numeric(df)
    task1_df = task1_numeric_screening(df, X_imp)

    return PlayTypeUnivariateResults(chi2=chi2_df, task1_numeric=task1_df)


def write_task1_univariate_artifacts(results: PlayTypeUnivariateResults) -> None:
    """Write Task 1 univariate selection CSVs to artifacts/feature_importance/."""
    ensure_artifacts_dir()
    results.chi2.to_csv(CHI2_SCREENING_CSV, index=False)
    results.task1_numeric.reset_index().to_csv(TASK1_NUMERIC_SCREENING_CSV, index=False)


def main() -> PlayTypeUnivariateResults:
    df = load_features_full()
    print(f"[INFO] Loaded features_full: {df.shape}")

    results = run_task1_univariate(df)
    write_task1_univariate_artifacts(results)

    n_sig = (results.chi2["significant"] == "✓").sum()
    print(f"[DONE] chi2 screening → {CHI2_SCREENING_CSV} ({n_sig}/{len(results.chi2)} significant)")
    print(
        f"[DONE] task1 numeric → {TASK1_NUMERIC_SCREENING_CSV} "
        f"({len(results.task1_numeric)} features)"
    )

    print("\nTop 10 Task 1 (combined_score):")
    print(
        results.task1_numeric[["combined_score", "spearman_rho", "mi_clf"]]
        .head(10)
        .to_string(float_format="{:.4f}".format)
    )

    return results


if __name__ == "__main__":
    main()
