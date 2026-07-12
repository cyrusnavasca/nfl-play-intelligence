"""
Shared helpers for play-type feature selection.

Column lists and thresholds live in ``feature_schema`` only. Filter-based
screening (Spearman / mutual information / chi-square) is intentionally absent —
features are chosen by hand in the feature config; the notebook covers those
statistics for reference. Only the embedded (LightGBM gain) stage remains.
"""
from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from sklearn.impute import SimpleImputer

from src.selection.shared.feature_schema import (
    ARTIFACTS_DIR,
    DROP_ALWAYS,
    EMBEDDED_THRESHOLD,
    FEATURES_FULL_PATH,
    SEED,
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
    """Encode play_type as pass=1, run=0."""
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
# Embedded gate
# ---------------------------------------------------------------------------


def passes_embedded_gate(
    embedded_df: pd.DataFrame,
    numeric_cols: list[str],
    *,
    threshold: float = EMBEDDED_THRESHOLD,
) -> list[str]:
    """
    Numeric features whose mean normalized gain meets the embedded threshold.

    Only rows whose ``feature`` is in *numeric_cols* are considered; one-hot
    categorical dummies are handled by ``categorical_passes_embedded_gate``.
    """
    eligible = set(numeric_cols)
    passed = embedded_df.loc[
        embedded_df["feature"].isin(eligible)
        & (embedded_df["embedded_importance_norm"] >= threshold),
        "feature",
    ]
    return passed.tolist()


def categorical_passes_embedded_gate(
    embedded_df: pd.DataFrame,
    cat_cols: list[str],
    *,
    threshold: float = EMBEDDED_THRESHOLD,
) -> list[str]:
    """
    Categoricals whose one-hot dummies' summed normalized gain meets the gate.

    Dummy columns are named ``"<cat>_<level>"`` (see ``build_task1_feature_matrix``);
    a categorical is kept when the total importance across its dummies >= threshold.
    """
    kept: list[str] = []
    importance = dict(
        zip(embedded_df["feature"], embedded_df["embedded_importance_norm"])
    )
    for cat in cat_cols:
        prefix = f"{cat}_"
        total = sum(
            imp for name, imp in importance.items() if name.startswith(prefix)
        )
        if total >= threshold:
            kept.append(cat)
    return kept


# ---------------------------------------------------------------------------
# Embedded importance helpers
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
