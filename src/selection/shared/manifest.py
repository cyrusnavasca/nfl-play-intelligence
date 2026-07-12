"""
Play-type selection manifest artifact.

Records the manual feature selection (from the feature config) and the embedded
gate final keep lists. There is no filter-based screening stage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from src.selection.embedded import PlayTypeEmbeddedResults
from src.selection.feature_config import FeatureConfig
from src.selection.threshold import build_task1_final
from src.selection.shared.common import ensure_artifacts_dir
from src.selection.shared.feature_schema import (
    DROP_ALWAYS,
    FEATURE_SELECTION_MANIFEST_PATH,
    validate_feature_schema,
)


@dataclass(frozen=True)
class ThresholdSelectionResults:
    """Container for embedded-gate selection artifacts."""

    selection_manifest: dict[str, object]


def load_selection_manifest(
    path=FEATURE_SELECTION_MANIFEST_PATH,
) -> dict[str, object]:
    """Load the selection manifest from disk."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build_selection_manifest(
    feature_config: FeatureConfig,
    *,
    task1_final_numeric: list[str],
    task1_final_categorical: list[str],
) -> dict[str, object]:
    return {
        "feature_config": feature_config.name,
        "thresholds": {
            "embedded_importance_norm": feature_config.embedded_importance_threshold,
        },
        "drop_always": sorted(DROP_ALWAYS),
        "task1": {
            "manual_selection": {
                "numeric": sorted(feature_config.numeric),
                "categorical": sorted(feature_config.categorical),
            },
            "final": {
                "numeric": task1_final_numeric,
                "categorical": task1_final_categorical,
            },
        },
    }


def validate_selection_manifest(selection_manifest: dict[str, object]) -> None:
    """Enforce done-when checks on the selection manifest."""
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
    feature_config: FeatureConfig,
    task1_embedded: PlayTypeEmbeddedResults,
) -> ThresholdSelectionResults:
    """Apply the embedded gate and assemble selection outputs."""
    validate_feature_schema(df.columns.tolist())

    task1_final_numeric, task1_final_categorical = build_task1_final(
        task1_embedded,
        feature_config.numeric,
        feature_config.categorical,
        threshold=feature_config.embedded_importance_threshold,
    )

    selection_manifest = build_selection_manifest(
        feature_config,
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
