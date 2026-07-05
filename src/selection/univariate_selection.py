"""
Phase 2 — univariate feature selection (chi-squared, Spearman, MI, combined score).

Port of ``notebooks/02_feature_selection.ipynb`` Cells 5–12. Column lists and
thresholds come from ``feature_schema`` only.

Usage (from project root):
    python3 -m src.selection.univariate_selection
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer

from src.selection.feature_schema import (
    ALL_NUMERIC,
    ARTIFACTS_DIR,
    CHI2_FEATURES,
    CHI2_P_THRESHOLD,
    CHI2_SCREENING_CSV,
    DROP_ALWAYS,
    FEATURES_FULL_PATH,
    MI_N_NEIGHBORS,
    MI_THRESHOLD,
    MIN_SPEARMAN_ROWS,
    SEED,
    SP_THRESHOLD,
    TARGET_CLF,
    TARGET_REG,
    TASK1_NUMERIC_SCREENING_CSV,
    TASK2_NUMERIC_SCREENING_CSV,
    validate_feature_schema,
)


@dataclass(frozen=True)
class UnivariateSelectionResults:
    """Container for Phase 2 univariate selection tables."""

    chi2: pd.DataFrame
    task1_numeric: pd.DataFrame
    task2_numeric: pd.DataFrame


def _assert_no_excluded(columns: list[str], context: str) -> None:
    leaked = sorted(set(columns) & set(DROP_ALWAYS))
    if leaked:
        raise ValueError(f"{context}: excluded columns in screening: {leaked}")


def binary_play_type(series: pd.Series) -> pd.Series:
    """Encode play_type as pass=1, run=0 (matches notebook Cell 8)."""
    return (series == "pass").astype(int)


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


def spearman_numeric_screening(
    df: pd.DataFrame,
    *,
    target_col: str,
    y: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Pairwise Spearman correlation for each ALL_NUMERIC column vs. target.

    Drops NaN pairwise per feature; returns NaN stats if fewer than
    MIN_SPEARMAN_ROWS valid rows.
    """
    _assert_no_excluded(ALL_NUMERIC, "spearman screening")

    rows: list[dict[str, object]] = []
    for col in ALL_NUMERIC:
        if y is not None:
            sub = df[[col]].join(y).dropna()
            target_values = sub[target_col]
        else:
            sub = df[[col, target_col]].dropna()
            target_values = sub[target_col]

        if len(sub) < MIN_SPEARMAN_ROWS:
            rows.append(
                {
                    "feature": col,
                    "spearman_rho": np.nan,
                    "abs_rho": np.nan,
                    "p_value": np.nan,
                }
            )
            continue

        rho, p = spearmanr(sub[col], target_values)
        rows.append(
            {
                "feature": col,
                "spearman_rho": rho,
                "abs_rho": abs(rho),
                "p_value": p,
            }
        )

    return pd.DataFrame(rows).set_index("feature")


def median_impute_numeric(df: pd.DataFrame) -> np.ndarray:
    """Median-impute ALL_NUMERIC columns (handles week-1 rolling NaNs)."""
    imp = SimpleImputer(strategy="median")
    return imp.fit_transform(df[ALL_NUMERIC])


def combine_numeric_screening(
    spearman_df: pd.DataFrame,
    mi: pd.Series,
    *,
    mi_col: str,
) -> pd.DataFrame:
    """
    Min-max normalize abs_rho and MI to [0, 1], then equal-weight average.

    Normalization matches the notebook: divide each column by its max.
    """
    mi_norm_col = f"{mi_col}_norm"
    screen = (
        spearman_df[["spearman_rho", "abs_rho", "p_value"]]
        .join(mi.rename(mi_col))
        .assign(
            **{
                mi_norm_col: lambda d: d[mi_col] / d[mi_col].max(),
                "abs_rho_norm": lambda d: d["abs_rho"] / d["abs_rho"].max(),
            }
        )
        .assign(
            combined_score=lambda d: 0.5 * d["abs_rho_norm"] + 0.5 * d[mi_norm_col]
        )
        .sort_values("combined_score", ascending=False)
    )
    return screen


def task1_numeric_screening(df: pd.DataFrame, X_imp: np.ndarray) -> pd.DataFrame:
    """Spearman + classification MI combined table for play_type."""
    y_clf = binary_play_type(df[TARGET_CLF])
    y_clf.name = TARGET_CLF

    sp_df = spearman_numeric_screening(df, target_col=TARGET_CLF, y=y_clf)
    mi_clf = mutual_info_classif(
        X_imp,
        y_clf,
        random_state=SEED,
        n_neighbors=MI_N_NEIGHBORS,
    )
    mi_s = pd.Series(mi_clf, index=ALL_NUMERIC)
    return combine_numeric_screening(sp_df, mi_s, mi_col="mi_clf")


def task2_numeric_screening(df: pd.DataFrame, X_imp: np.ndarray) -> pd.DataFrame:
    """Spearman + regression MI combined table for yards_gained."""
    y_reg = df[TARGET_REG]
    sp_df = spearman_numeric_screening(df, target_col=TARGET_REG)
    y_reg_imp = y_reg.fillna(y_reg.median())
    mi_reg = mutual_info_regression(
        X_imp,
        y_reg_imp,
        random_state=SEED,
        n_neighbors=MI_N_NEIGHBORS,
    )
    mi_s = pd.Series(mi_reg, index=ALL_NUMERIC)
    return combine_numeric_screening(sp_df, mi_s, mi_col="mi_reg")


def run_univariate_selection(df: pd.DataFrame) -> UnivariateSelectionResults:
    """Run all Phase 2 univariate selection steps."""
    validate_feature_schema(df.columns.tolist())

    chi2_df = chi2_categorical_screening(df)
    X_imp = median_impute_numeric(df)
    task1_df = task1_numeric_screening(df, X_imp)
    task2_df = task2_numeric_screening(df, X_imp)

    return UnivariateSelectionResults(
        chi2=chi2_df,
        task1_numeric=task1_df,
        task2_numeric=task2_df,
    )


def chi2_significant_features(chi2_df: pd.DataFrame) -> list[str]:
    """Categorical features significant at CHI2_P_THRESHOLD (Task 1 gate input)."""
    passed = chi2_df.loc[chi2_df["p_value"] < CHI2_P_THRESHOLD, "feature"]
    return passed.tolist()


def passes_univariate_gate(screen_df: pd.DataFrame, *, mi_col: str) -> list[str]:
    """
    Numeric features passing Spearman OR MI threshold (Stage 1 gate input).

    Keeps a feature when ``abs_rho >= SP_THRESHOLD`` or ``mi >= MI_THRESHOLD``.
    """
    mask = (screen_df["abs_rho"] >= SP_THRESHOLD) | (screen_df[mi_col] >= MI_THRESHOLD)
    return screen_df.loc[mask].index.tolist()


def write_univariate_artifacts(results: UnivariateSelectionResults) -> None:
    """Write Phase 2 univariate selection CSVs to artifacts/feature_importance/."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    results.chi2.to_csv(CHI2_SCREENING_CSV, index=False)
    results.task1_numeric.reset_index().to_csv(TASK1_NUMERIC_SCREENING_CSV, index=False)
    results.task2_numeric.reset_index().to_csv(TASK2_NUMERIC_SCREENING_CSV, index=False)


def load_features_full(path=FEATURES_FULL_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def main() -> UnivariateSelectionResults:
    df = load_features_full()
    print(f"[INFO] Loaded features_full: {df.shape}")

    results = run_univariate_selection(df)
    write_univariate_artifacts(results)

    n_sig = (results.chi2["significant"] == "✓").sum()
    print(f"[DONE] chi2 screening → {CHI2_SCREENING_CSV} ({n_sig}/{len(results.chi2)} significant)")
    print(f"[DONE] task1 numeric → {TASK1_NUMERIC_SCREENING_CSV} ({len(results.task1_numeric)} features)")
    print(f"[DONE] task2 numeric → {TASK2_NUMERIC_SCREENING_CSV} ({len(results.task2_numeric)} features)")

    print("\nTop 10 Task 1 (combined_score):")
    print(
        results.task1_numeric[["combined_score", "spearman_rho", "mi_clf"]]
        .head(10)
        .to_string(float_format="{:.4f}".format)
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
