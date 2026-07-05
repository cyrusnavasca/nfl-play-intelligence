"""
Phase 4 — threshold-based feature selection and selection manifest.

Applies fixed Stage 1 (univariate) and Stage 2 (embedded) gates to produce
machine-readable keep lists. Generates out-of-fold ``pred_pass_proba`` for
Task 2 when not supplied from Phase 3.

Usage (from project root):
    python3 -m src.selection.threshold_selection
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from src.selection.embedded_selection import (
    EmbeddedSelectionResults,
    build_task1_feature_matrix,
    chi2_significant_categoricals,
    cross_fitted_pass_proba,
    run_embedded_selection,
)
from src.selection.feature_schema import (
    ALL_NUMERIC,
    ARTIFACTS_DIR,
    CAT_FEATURES,
    CHI2_P_THRESHOLD,
    CROSS_TASK_SUMMARY_CSV,
    DROP_ALWAYS,
    EMBEDDED_THRESHOLD,
    FEATURE_SELECTION_MANIFEST_PATH,
    FEATURES_FULL_PATH,
    MI_THRESHOLD,
    SP_THRESHOLD,
    TARGET_REG,
    TASK2_GENERATED_FEATURES,
    validate_feature_schema,
)
from src.selection.univariate_selection import (
    UnivariateSelectionResults,
    binary_play_type,
    load_features_full,
    passes_univariate_gate,
    run_univariate_selection,
)


@dataclass(frozen=True)
class ThresholdSelectionResults:
    """Container for Phase 4 threshold selection artifacts."""

    selection_manifest: dict[str, object]
    cross_task_summary: pd.DataFrame
    oof_pass_proba: pd.Series


def _assert_no_drop_always(columns: list[str], context: str) -> None:
    leaked = sorted(set(columns) & set(DROP_ALWAYS))
    if leaked:
        raise ValueError(f"{context}: DROP_ALWAYS columns in keep list: {leaked}")


def passes_embedded_gate(
    embedded_df: pd.DataFrame,
    numeric_cols: list[str],
    *,
    threshold: float = EMBEDDED_THRESHOLD,
) -> list[str]:
    """
    Univariate-passing numerics that also meet the embedded gain threshold.

    Only rows whose ``feature`` is in *numeric_cols* are considered; one-hot
    categoricals and ``pred_pass_proba`` are excluded from this gate.
    """
    eligible = set(numeric_cols)
    passed = embedded_df.loc[
        embedded_df["feature"].isin(eligible)
        & (embedded_df["embedded_importance_norm"] >= threshold),
        "feature",
    ]
    return passed.tolist()


def _task1_univariate_pass(
    univariate_results: UnivariateSelectionResults,
) -> tuple[list[str], list[str]]:
    numeric = passes_univariate_gate(univariate_results.task1_numeric, mi_col="mi_clf")
    categorical = chi2_significant_categoricals(univariate_results.chi2)
    return numeric, categorical


def _task2_univariate_pass(univariate_results: UnivariateSelectionResults) -> list[str]:
    return passes_univariate_gate(univariate_results.task2_numeric, mi_col="mi_reg")


def build_task1_final(
    univariate_results: UnivariateSelectionResults,
    embedded_results: EmbeddedSelectionResults,
) -> tuple[list[str], list[str]]:
    """Stage 2 numeric keep + chi-squared categoricals (Task 1)."""
    univariate_numeric, univariate_categorical = _task1_univariate_pass(univariate_results)
    final_numeric = passes_embedded_gate(
        embedded_results.task1, univariate_numeric
    )
    _assert_no_drop_always(final_numeric, "task1 final numeric")
    _assert_no_drop_always(univariate_categorical, "task1 final categorical")
    return sorted(final_numeric), sorted(univariate_categorical)


def build_task2_final(
    univariate_results: UnivariateSelectionResults,
    embedded_results: EmbeddedSelectionResults,
) -> list[str]:
    """Stage 2 numeric keep plus always-included ``pred_pass_proba``."""
    univariate_numeric = _task2_univariate_pass(univariate_results)
    final_numeric = passes_embedded_gate(
        embedded_results.task2, univariate_numeric
    )
    _assert_no_drop_always(final_numeric, "task2 final numeric")
    return sorted(final_numeric)


def _oof_pass_proba(
    df: pd.DataFrame,
    univariate_results: UnivariateSelectionResults,
    embedded_results: EmbeddedSelectionResults | None,
) -> pd.Series:
    if embedded_results is not None:
        return embedded_results.oof_pass_proba

    univariate_numeric, univariate_categorical = _task1_univariate_pass(univariate_results)
    X_task1 = build_task1_feature_matrix(df, univariate_numeric, univariate_categorical)
    y_clf = binary_play_type(df[TARGET_CLF])
    return cross_fitted_pass_proba(X_task1, y_clf)


def build_selection_manifest(
    univariate_results: UnivariateSelectionResults,
    embedded_results: EmbeddedSelectionResults,
    *,
    task1_final_numeric: list[str],
    task1_final_categorical: list[str],
    task2_final_numeric: list[str],
) -> dict[str, object]:
    task1_univariate_numeric, task1_univariate_categorical = _task1_univariate_pass(
        univariate_results
    )
    task2_univariate_numeric = _task2_univariate_pass(univariate_results)

    return {
        "thresholds": {
            "spearman_abs_rho": SP_THRESHOLD,
            "mutual_information": MI_THRESHOLD,
            "embedded_importance_norm": EMBEDDED_THRESHOLD,
            "chi2_p_value": CHI2_P_THRESHOLD,
        },
        "drop_always": sorted(DROP_ALWAYS),
        "task1": {
            "filter_pass": {
                "numeric": sorted(task1_univariate_numeric),
                "categorical": sorted(task1_univariate_categorical),
            },
            "final": {
                "numeric": task1_final_numeric,
                "categorical": task1_final_categorical,
            },
        },
        "task2": {
            "filter_pass": {
                "numeric": sorted(task2_univariate_numeric),
            },
            "final": {
                "numeric": task2_final_numeric,
                "generated": sorted(TASK2_GENERATED_FEATURES),
            },
        },
    }


def _rank_series(series: pd.Series) -> pd.Series:
    """Rank descending; NaN values receive no rank."""
    return series.rank(ascending=False, method="min")


def build_cross_task_summary(
    univariate_results: UnivariateSelectionResults,
    embedded_results: EmbeddedSelectionResults,
    *,
    task1_final_numeric: list[str],
    task1_final_categorical: list[str],
    task2_final_numeric: list[str],
) -> pd.DataFrame:
    """Side-by-side univariate and embedded ranks for both tasks."""
    task1_univariate_numeric, task1_univariate_categorical = _task1_univariate_pass(
        univariate_results
    )
    task2_univariate_numeric = _task2_univariate_pass(univariate_results)

    task1_final_numeric_set = set(task1_final_numeric)
    task1_final_categorical_set = set(task1_final_categorical)
    task2_final_numeric_set = set(task2_final_numeric)
    proba_col = TASK2_GENERATED_FEATURES[0]

    features = sorted(
        set(ALL_NUMERIC)
        | set(CAT_FEATURES)
        | set(TASK2_GENERATED_FEATURES)
    )

    t1_num = univariate_results.task1_numeric
    t2_num = univariate_results.task2_numeric
    t1_emb = embedded_results.task1.set_index("feature")
    t2_emb = embedded_results.task2.set_index("feature")

    rows: list[dict[str, object]] = []
    for feat in features:
        row: dict[str, object] = {"feature": feat}

        if feat in t1_num.index:
            row["task1_spearman_rho"] = t1_num.loc[feat, "spearman_rho"]
            row["task1_abs_rho"] = t1_num.loc[feat, "abs_rho"]
            row["task1_mi_clf"] = t1_num.loc[feat, "mi_clf"]
            row["task1_combined_score"] = t1_num.loc[feat, "combined_score"]
            row["task1_filter_pass"] = feat in task1_univariate_numeric
            row["task1_embedded_pass"] = feat in task1_final_numeric_set
            row["task1_final_numeric"] = feat in task1_final_numeric_set
        elif feat in CAT_FEATURES:
            chi2_row = univariate_results.chi2.loc[
                univariate_results.chi2["feature"] == feat
            ]
            if not chi2_row.empty:
                row["task1_chi2"] = chi2_row.iloc[0]["chi2"]
                row["task1_chi2_p_value"] = chi2_row.iloc[0]["p_value"]
            row["task1_filter_pass"] = feat in task1_univariate_categorical
            row["task1_final_categorical"] = feat in task1_final_categorical_set
            row["task1_final_keep"] = feat in task1_final_categorical_set
        else:
            row["task1_filter_pass"] = False
            row["task1_final_keep"] = False

        if feat in t1_emb.index:
            row["task1_embedded_importance_norm"] = t1_emb.loc[
                feat, "embedded_importance_norm"
            ]

        if feat == proba_col:
            row["task2_filter_pass"] = True
            row["task2_embedded_pass"] = True
            row["task2_final_keep"] = True
        elif feat in t2_num.index:
            row["task2_spearman_rho"] = t2_num.loc[feat, "spearman_rho"]
            row["task2_abs_rho"] = t2_num.loc[feat, "abs_rho"]
            row["task2_mi_reg"] = t2_num.loc[feat, "mi_reg"]
            row["task2_combined_score"] = t2_num.loc[feat, "combined_score"]
            row["task2_filter_pass"] = feat in task2_univariate_numeric
            row["task2_embedded_pass"] = feat in task2_final_numeric_set
            row["task2_final_keep"] = feat in task2_final_numeric_set

        if feat in t2_emb.index:
            row["task2_embedded_importance_norm"] = t2_emb.loc[
                feat, "embedded_importance_norm"
            ]

        if feat in task1_final_numeric_set:
            row["task1_final_keep"] = True
        elif feat not in CAT_FEATURES and "task1_final_keep" not in row:
            row["task1_final_keep"] = False

        rows.append(row)

    summary = pd.DataFrame(rows)

    for col in ("task1_combined_score", "task1_embedded_importance_norm"):
        if col in summary.columns:
            summary[f"{col}_rank"] = _rank_series(summary[col])

    for col in ("task2_combined_score", "task2_embedded_importance_norm"):
        if col in summary.columns:
            summary[f"{col}_rank"] = _rank_series(summary[col])

    return summary.sort_values("feature").reset_index(drop=True)


def validate_selection_manifest(selection_manifest: dict[str, object]) -> None:
    """Enforce Phase 4 done-when checks on the selection manifest."""
    drop_always = set(selection_manifest["drop_always"])  # type: ignore[index]

    task1 = selection_manifest["task1"]  # type: ignore[index]
    task2 = selection_manifest["task2"]  # type: ignore[index]

    task1_final_numeric = task1["final"]["numeric"]
    task1_final_categorical = task1["final"]["categorical"]
    task2_final_numeric = task2["final"]["numeric"]
    task2_generated = task2["final"]["generated"]

    all_final = (
        list(task1_final_numeric)
        + list(task1_final_categorical)
        + list(task2_final_numeric)
        + list(task2_generated)
    )
    leaked = sorted(set(all_final) & drop_always)
    if leaked:
        raise ValueError(f"final lists contain DROP_ALWAYS columns: {leaked}")

    if TARGET_REG in all_final:
        raise ValueError("task1 final must not include yards_gained")

    if proba_col := TASK2_GENERATED_FEATURES[0]:
        if proba_col not in task2_generated:
            raise ValueError("task2 final must include pred_pass_proba in generated")


def run_threshold_selection(
    df: pd.DataFrame,
    univariate_results: UnivariateSelectionResults,
    embedded_results: EmbeddedSelectionResults | None = None,
) -> ThresholdSelectionResults:
    """Apply threshold gates and assemble Phase 4 outputs."""
    validate_feature_schema(df.columns.tolist())

    if embedded_results is None:
        embedded_results = run_embedded_selection(df, univariate_results)

    task1_final_numeric, task1_final_categorical = build_task1_final(
        univariate_results, embedded_results
    )
    task2_final_numeric = build_task2_final(univariate_results, embedded_results)
    oof_pass_proba = _oof_pass_proba(df, univariate_results, embedded_results)

    selection_manifest = build_selection_manifest(
        univariate_results,
        embedded_results,
        task1_final_numeric=task1_final_numeric,
        task1_final_categorical=task1_final_categorical,
        task2_final_numeric=task2_final_numeric,
    )
    validate_selection_manifest(selection_manifest)

    cross_task_summary = build_cross_task_summary(
        univariate_results,
        embedded_results,
        task1_final_numeric=task1_final_numeric,
        task1_final_categorical=task1_final_categorical,
        task2_final_numeric=task2_final_numeric,
    )

    return ThresholdSelectionResults(
        selection_manifest=selection_manifest,
        cross_task_summary=cross_task_summary,
        oof_pass_proba=oof_pass_proba,
    )


def write_threshold_selection_artifacts(results: ThresholdSelectionResults) -> None:
    """Write feature_selection_manifest.json and cross_task_summary.csv."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(FEATURE_SELECTION_MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(results.selection_manifest, fh, indent=2)
        fh.write("\n")

    results.cross_task_summary.to_csv(CROSS_TASK_SUMMARY_CSV, index=False)


def main() -> ThresholdSelectionResults:
    df = load_features_full()
    print(f"[INFO] Loaded features_full: {df.shape}")

    print("[INFO] Running univariate selection (Phase 2)...")
    univariate_results = run_univariate_selection(df)

    print("[INFO] Running embedded selection (Phase 3)...")
    embedded_results = run_embedded_selection(df, univariate_results)

    print("[INFO] Applying threshold selection (Phase 4)...")
    results = run_threshold_selection(df, univariate_results, embedded_results)
    write_threshold_selection_artifacts(results)

    manifest = results.selection_manifest
    t1 = manifest["task1"]["final"]  # type: ignore[index]
    t2 = manifest["task2"]["final"]  # type: ignore[index]

    print(f"[DONE] selection manifest → {FEATURE_SELECTION_MANIFEST_PATH}")
    print(f"[DONE] cross-task summary → {CROSS_TASK_SUMMARY_CSV}")
    print(
        f"[INFO] Task 1 final: {len(t1['numeric'])} numeric + "
        f"{len(t1['categorical'])} categorical"
    )
    print(
        f"[INFO] Task 2 final: {len(t2['numeric'])} numeric + "
        f"{len(t2['generated'])} generated"
    )
    print(f"[INFO] OOF pred_pass_proba mean={results.oof_pass_proba.mean():.4f}")

    print("\nTask 1 final numeric:")
    for feat in t1["numeric"]:
        print(f"  {feat}")
    print("\nTask 1 final categorical:")
    for feat in t1["categorical"]:
        print(f"  {feat}")
    print("\nTask 2 final numeric:")
    for feat in t2["numeric"]:
        print(f"  {feat}")

    return results


if __name__ == "__main__":
    main()
