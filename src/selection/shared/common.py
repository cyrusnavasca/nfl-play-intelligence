"""
Shared helpers for both play-type and yards-gained selection tasks.

Column lists and thresholds live in ``feature_schema`` only.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

from src.selection.shared.feature_schema import (
    ALL_NUMERIC,
    ARTIFACTS_DIR,
    CAT_FEATURES,
    CHI2_P_THRESHOLD,
    DROP_ALWAYS,
    EMBEDDED_THRESHOLD,
    FEATURES_FULL_PATH,
    MI_N_NEIGHBORS,
    MI_THRESHOLD,
    MIN_SPEARMAN_ROWS,
    SEED,
    SP_THRESHOLD,
    TARGET_CLF,
)

# ---------------------------------------------------------------------------
# Reproducibility — embedded CV
# ---------------------------------------------------------------------------

N_FOLDS = 5

LGBM_PARAMS: dict[str, object] = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "verbosity": -1,
    "importance_type": "gain",
}

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_features_full(path=FEATURES_FULL_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def ensure_artifacts_dir() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def binary_play_type(series: pd.Series) -> pd.Series:
    """Encode play_type as pass=1, run=0 (matches notebook Cell 8)."""
    return (series == "pass").astype(int)


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def _assert_no_excluded(columns: list[str], context: str) -> None:
    leaked = sorted(set(columns) & set(DROP_ALWAYS))
    if leaked:
        raise ValueError(f"{context}: excluded columns present: {leaked}")


def _assert_no_drop_always(columns: list[str], context: str) -> None:
    leaked = sorted(set(columns) & set(DROP_ALWAYS))
    if leaked:
        raise ValueError(f"{context}: DROP_ALWAYS columns in keep list: {leaked}")


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------


def median_impute_numeric(df: pd.DataFrame) -> np.ndarray:
    """Median-impute ALL_NUMERIC columns (handles week-1 rolling NaNs)."""
    imp = SimpleImputer(strategy="median")
    return imp.fit_transform(df[ALL_NUMERIC])


def median_impute_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Median-impute *columns* on the full dataset; leaves other columns unchanged."""
    if not columns:
        return df

    out = df.copy()
    imp = SimpleImputer(strategy="median")
    out[columns] = imp.fit_transform(out[columns])
    return out


def _median_impute_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median-impute inside CV folds — fit on train, transform test."""
    imp = SimpleImputer(strategy="median")
    train_vals = imp.fit_transform(X_train)
    test_vals = imp.transform(X_test)
    return (
        pd.DataFrame(train_vals, columns=X_train.columns, index=X_train.index),
        pd.DataFrame(test_vals, columns=X_test.columns, index=X_test.index),
    )


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def chi2_significant_features(chi2_df: pd.DataFrame) -> list[str]:
    """Categorical features significant at CHI2_P_THRESHOLD (Task 1 gate input)."""
    passed = chi2_df.loc[chi2_df["p_value"] < CHI2_P_THRESHOLD, "feature"]
    return passed.tolist()


def chi2_significant_categoricals(chi2_df: pd.DataFrame) -> list[str]:
    """χ²-significant raw categoricals only (exclude binary/discrete numerics)."""
    passed = set(chi2_significant_features(chi2_df))
    return [col for col in CAT_FEATURES if col in passed]


def passes_univariate_gate(screen_df: pd.DataFrame, *, mi_col: str) -> list[str]:
    """
    Numeric features passing Spearman OR MI threshold (Stage 1 gate input).

    Keeps a feature when ``abs_rho >= SP_THRESHOLD`` or ``mi >= MI_THRESHOLD``.
    """
    mask = (screen_df["abs_rho"] >= SP_THRESHOLD) | (screen_df[mi_col] >= MI_THRESHOLD)
    return screen_df.loc[mask].index.tolist()


def passes_embedded_gate(
    embedded_df: pd.DataFrame,
    numeric_cols: list[str],
    *,
    threshold: float = EMBEDDED_THRESHOLD,
) -> list[str]:
    """
    Univariate-passing numerics that also meet the embedded gain threshold.

    Only rows whose ``feature`` is in *numeric_cols* are considered; one-hot
    categoricals are excluded from this gate.
    """
    eligible = set(numeric_cols)
    passed = embedded_df.loc[
        embedded_df["feature"].isin(eligible)
        & (embedded_df["embedded_importance_norm"] >= threshold),
        "feature",
    ]
    return passed.tolist()


# ---------------------------------------------------------------------------
# Univariate stats
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Embedded helpers
# ---------------------------------------------------------------------------


def _accumulate_fold_gain(
    model: lgb.LGBMClassifier,
    feature_names: list[str],
    accum: dict[str, float],
) -> None:
    gains = model.booster_.feature_importance(importance_type="gain")
    for name, gain in zip(feature_names, gains):
        accum[name] += float(gain)


def _importance_dataframe(accum: dict[str, float]) -> pd.DataFrame:
    total = sum(accum.values())
    rows = [
        {
            "feature": name,
            "embedded_importance": gain,
            "embedded_importance_norm": gain / total if total > 0 else 0.0,
        }
        for name, gain in accum.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("embedded_importance_norm", ascending=False)
        .reset_index(drop=True)
    )
