"""
Play-type selection manifest artifact.

Composes the Task 1 final keep lists from the threshold module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from src.selection.embedded import PlayTypeEmbeddedResults
from src.selection.threshold import build_task1_final
from src.selection.univariate import PlayTypeUnivariateResults
from src.selection.shared.common import (
    chi2_significant_categoricals,
    ensure_artifacts_dir,
    passes_univariate_gate,
)
from src.selection.shared.feature_schema import (
    CHI2_P_THRESHOLD,
    DROP_ALWAYS,
    EMBEDDED_THRESHOLD,
    FEATURE_SELECTION_MANIFEST_PATH,
    MI_THRESHOLD,
    SP_THRESHOLD,
    validate_feature_schema,
)


@dataclass(frozen=True)
class UnivariateSelectionResults:
    """Univariate results for play-type selection (manifest view)."""

    chi2: pd.DataFrame
    task1_numeric: pd.DataFrame


@dataclass(frozen=True)
class EmbeddedSelectionResults:
    """Embedded results for play-type selection."""

    task1: pd.DataFrame


@dataclass(frozen=True)
class ThresholdSelectionResults:
    """Container for Phase 4 threshold selection artifacts."""

    selection_manifest: dict[str, object]


def combine_univariate_results(
    task1: PlayTypeUnivariateResults,
) -> UnivariateSelectionResults:
    return UnivariateSelectionResults(
        chi2=task1.chi2,
        task1_numeric=task1.task1_numeric,
    )


def combine_embedded_results(
    task1: PlayTypeEmbeddedResults,
) -> EmbeddedSelectionResults:
    return EmbeddedSelectionResults(task1=task1.task1)


def load_selection_manifest(
    path=FEATURE_SELECTION_MANIFEST_PATH,
) -> dict[str, object]:
    """Load the Phase 4 selection manifest from disk."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _task1_univariate_pass(
    univariate_results: UnivariateSelectionResults,
) -> tuple[list[str], list[str]]:
    numeric = passes_univariate_gate(univariate_results.task1_numeric, mi_col="mi_clf")
    categorical = chi2_significant_categoricals(univariate_results.chi2)
    return numeric, categorical


def build_selection_manifest(
    univariate_results: UnivariateSelectionResults,
    *,
    task1_final_numeric: list[str],
    task1_final_categorical: list[str],
) -> dict[str, object]:
    task1_univariate_numeric, task1_univariate_categorical = _task1_univariate_pass(
        univariate_results
    )

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
    }


def validate_selection_manifest(selection_manifest: dict[str, object]) -> None:
    """Enforce Phase 4 done-when checks on the selection manifest."""
    drop_always = set(selection_manifest["drop_always"])  # type: ignore[index]

    task1 = selection_manifest["task1"]  # type: ignore[index]
    task1_final_numeric = task1["final"]["numeric"]
    task1_final_categorical = task1["final"]["categorical"]

    all_final = list(task1_final_numeric) + list(task1_final_categorical)
    leaked = sorted(set(all_final) & drop_always)
    if leaked:
        raise ValueError(f"final lists contain DROP_ALWAYS columns: {leaked}")


def run_threshold_selection(
    df: pd.DataFrame,
    task1_univariate: PlayTypeUnivariateResults,
    task1_embedded: PlayTypeEmbeddedResults,
) -> ThresholdSelectionResults:
    """Apply threshold gates and assemble Phase 4 outputs."""
    validate_feature_schema(df.columns.tolist())

    univariate_results = combine_univariate_results(task1_univariate)

    task1_final_numeric, task1_final_categorical = build_task1_final(
        task1_univariate, task1_embedded
    )

    selection_manifest = build_selection_manifest(
        univariate_results,
        task1_final_numeric=task1_final_numeric,
        task1_final_categorical=task1_final_categorical,
    )
    validate_selection_manifest(selection_manifest)

    return ThresholdSelectionResults(selection_manifest=selection_manifest)


def write_threshold_selection_artifacts(results: ThresholdSelectionResults) -> None:
    """Write feature_selection_manifest.json."""
    ensure_artifacts_dir()

    with open(FEATURE_SELECTION_MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(results.selection_manifest, fh, indent=2)
        fh.write("\n")
