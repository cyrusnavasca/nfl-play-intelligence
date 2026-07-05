"""
Aggregate CV fold records and select the best model per task.

Reference: ``docs/modeling_plan.md`` Phase 2.
"""
from __future__ import annotations

import pandas as pd

__all__ = ["select_best_model", "summarize_cv_results"]


def summarize_cv_results(records: list[dict]) -> pd.DataFrame:
    """
    Aggregate fold-level CV records into mean ± std per model.

    *records* must include a ``model`` key and one row per fold per model.
    Metric columns are every key except ``model`` and ``fold``.
    """
    if not records:
        raise ValueError("summarize_cv_results requires at least one record")

    df = pd.DataFrame(records)
    if "model" not in df.columns:
        raise ValueError("records must include a 'model' column")

    metric_cols = [
        col for col in df.columns if col not in {"model", "fold"}
    ]
    if not metric_cols:
        raise ValueError("records must include at least one metric column")

    grouped = df.groupby("model", sort=True)
    summary_rows: list[dict[str, object]] = []

    for model_key, group in grouped:
        row: dict[str, object] = {"model": model_key}
        for metric in metric_cols:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=0)
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def select_best_model(
    comparison_df: pd.DataFrame,
    metric: str,
    *,
    higher_is_better: bool,
) -> str:
    """
    Return the model key with the best mean *metric* from a comparison table.

    Expects ``{metric}_mean`` column (output of ``summarize_cv_results``).
    Play type: ``metric="roc_auc"``, ``higher_is_better=True``.
    Yards gained: ``metric="rmse"``, ``higher_is_better=False``.
    """
    mean_col = f"{metric}_mean"
    if mean_col not in comparison_df.columns:
        raise ValueError(f"comparison_df missing column {mean_col!r}")

    if "model" not in comparison_df.columns:
        raise ValueError("comparison_df must include a 'model' column")

    ranked = comparison_df.sort_values(
        mean_col,
        ascending=not higher_is_better,
        kind="mergesort",
    )
    return str(ranked.iloc[0]["model"])
