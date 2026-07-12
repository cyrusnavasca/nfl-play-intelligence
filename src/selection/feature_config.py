"""
Manual feature selection config.

Loads the hand-curated keep lists from ``configs/features/<name>.yaml`` — the
only source of which candidate features enter the selection pipeline. There is
no automated filter/threshold stage; selected features feed the embedded
(LightGBM gain) stage for further pruning.

Usage (from project root):
    python3 -m src.selection.feature_config
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.selection.shared.feature_schema import (
    ALL_NUMERIC,
    CAT_FEATURES,
    DEFAULT_FEATURES_CONFIG,
    DROP_ALWAYS,
    EMBEDDED_THRESHOLD,
)


@dataclass(frozen=True)
class FeatureConfig:
    """Hand-curated feature lists + embedded gate threshold for one run."""

    name: str
    numeric: list[str]
    categorical: list[str]
    embedded_importance_threshold: float


def load_feature_config(path: Path | str = DEFAULT_FEATURES_CONFIG) -> FeatureConfig:
    """Load and validate a manual feature-selection config."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"feature config not found: {config_path}")

    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    numeric = list(raw.get("numeric", []) or [])
    categorical = list(raw.get("categorical", []) or [])
    threshold = float(raw.get("embedded_importance_threshold", EMBEDDED_THRESHOLD))
    name = str(raw.get("name", config_path.stem))

    _validate(numeric, categorical, threshold, context=str(config_path))

    return FeatureConfig(
        name=name,
        numeric=numeric,
        categorical=categorical,
        embedded_importance_threshold=threshold,
    )


def _validate(
    numeric: list[str],
    categorical: list[str],
    threshold: float,
    *,
    context: str,
) -> None:
    if not numeric and not categorical:
        raise ValueError(f"{context}: feature config selects zero features")

    dup_numeric = _duplicates(numeric)
    dup_cat = _duplicates(categorical)
    if dup_numeric or dup_cat:
        raise ValueError(
            f"{context}: duplicate features — numeric={dup_numeric}, "
            f"categorical={dup_cat}"
        )

    unknown_numeric = sorted(set(numeric) - set(ALL_NUMERIC))
    if unknown_numeric:
        raise ValueError(
            f"{context}: numeric features not in ALL_NUMERIC: {unknown_numeric}"
        )

    unknown_cat = sorted(set(categorical) - set(CAT_FEATURES))
    if unknown_cat:
        raise ValueError(
            f"{context}: categorical features not in CAT_FEATURES: {unknown_cat}"
        )

    leaked = sorted(set(numeric + categorical) & set(DROP_ALWAYS))
    if leaked:
        raise ValueError(f"{context}: DROP_ALWAYS (leaky) columns selected: {leaked}")

    if threshold < 0:
        raise ValueError(f"{context}: embedded_importance_threshold must be >= 0")


def _duplicates(items: list[str]) -> list[str]:
    seen: set[str] = set()
    dups: set[str] = set()
    for item in items:
        if item in seen:
            dups.add(item)
        seen.add(item)
    return sorted(dups)


def main() -> FeatureConfig:
    cfg = load_feature_config()
    print(f"[INFO] Feature config: {cfg.name}")
    print(f"  numeric ({len(cfg.numeric)}): {cfg.numeric}")
    print(f"  categorical ({len(cfg.categorical)}): {cfg.categorical}")
    print(f"  embedded_importance_threshold: {cfg.embedded_importance_threshold}")
    return cfg


if __name__ == "__main__":
    main()
