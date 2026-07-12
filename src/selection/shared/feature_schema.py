"""
Feature schema and data contract for the feature-selection pipeline.

Single source of truth for every column in ``features_full.parquet``.
Downstream selection modules must import lists from here — no hardcoded
column names elsewhere.

Reference: ``notebooks/02_feature_selection.ipynb`` Cell 4 (lifted and aligned
with ``docs/feature_selection_plan.md`` Phase 1).
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FEATURES_FULL_PATH = Path("data/interim/features_full.parquet")
PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("artifacts/feature_importance")

PLAY_TYPE_MODELING_PARQUET_PATH = PROCESSED_DIR / "play_type_modeling.parquet"

# Manual feature-selection configs (the only source of which features enter
# the pipeline; see configs/features/default.yaml + src/selection/feature_config.py).
FEATURES_CONFIG_DIR = Path("configs/features")
DEFAULT_FEATURES_CONFIG = FEATURES_CONFIG_DIR / "default.yaml"

TASK1_EMBEDDED_CSV = ARTIFACTS_DIR / "task1_embedded_importance.csv"
FEATURE_SELECTION_MANIFEST_PATH = ARTIFACTS_DIR / "feature_selection_manifest.json"

# ---------------------------------------------------------------------------
# Reproducibility & embedded selection threshold
# ---------------------------------------------------------------------------
#
# Filter-based screening (Spearman / mutual information / chi-square) is no
# longer applied automatically — features are chosen by hand in the feature
# config. The notebook (02_feat_selection.ipynb) covers those statistics for
# reference. Only the embedded (LightGBM gain) gate remains, and its threshold
# is set per feature config (`embedded_importance_threshold`), defaulting here.

SEED = 42

EMBEDDED_THRESHOLD = 0.01

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

TARGET_CLF = "play_type"

TARGET_COLS: list[str] = [TARGET_CLF]

# ---------------------------------------------------------------------------
# Excluded columns (never screened or modeled)
# ---------------------------------------------------------------------------

ID_COLS: list[str] = [
    "game_id",
    "play_id",
    "game_date",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
]

LEAKY_COLS: list[str] = [
    "epa",  # outcome-derived
    "yards_gained",  # play outcome; used upstream for team features, never a model feature
]

# Pipeline metadata; not model inputs (game_date already excluded above)
TEMPORAL_COLS: list[str] = [
    "season",
    "week",
]

REDUNDANT_TIME_COLS: list[str] = [
    "quarter_seconds_remaining",
    "half_seconds_remaining",
]

REDUNDANT_SCORE_COLS: list[str] = [
    "posteam_score",
    "defteam_score",
]

RAW_REPLACED_COLS: list[str] = [
    "offense_personnel",
    "defense_personnel",
]

DROP_ALWAYS: list[str] = sorted(
    set(
        ID_COLS
        + LEAKY_COLS
        + TEMPORAL_COLS
        + REDUNDANT_TIME_COLS
        + REDUNDANT_SCORE_COLS
        + RAW_REPLACED_COLS
    )
)

# ---------------------------------------------------------------------------
# Candidate features
# ---------------------------------------------------------------------------

CAT_FEATURES: list[str] = [
    "season_type",
    "posteam_type",
    "game_half",
    "roof",
    "surface",
    "offense_formation",
]

RAW_NUMERIC: list[str] = [
    "qtr",
    "game_seconds_remaining",
    "down",
    "ydstogo",
    "yardline_100",
    "defenders_in_box",
    "temp",
    "wind",
]

ROLLING_FEATURES: list[str] = [
    "off_pass_rate_L3",
    "off_rush_yds_L3",
    "off_pass_yds_L3",
    "off_epa_L3",
    "off_rush_epa_L3",
    "off_pass_epa_L3",
    "def_rush_yds_allowed_L3",
    "def_pass_yds_allowed_L3",
    "def_epa_L3",
    "def_rush_epa_L3",
    "def_pass_epa_L3",
]

FORMATION_FEATURES: list[str] = [
    "is_qb_in_gun",
    "box_advantage",
]

SITUATIONAL_FEATURES: list[str] = [
    "score_differential",
    "time_adjusted_score_diff",
    "red_zone",
    "backed_up",
    "two_minute_drill",
]

ENCODED_FEATURES: list[str] = [
    "is_turf",
    "is_indoor",
    "is_playoffs",
    "is_home",
]

# Binary flags + discrete integers tested via chi-squared (Task 1 only).
# Encoded binaries (is_turf, etc.) are excluded — raw categoricals cover them.
CHI2_BINARY_FEATURES: list[str] = [
    "is_qb_in_gun",
    "red_zone",
    "backed_up",
    "two_minute_drill",
]

CHI2_DISCRETE_FEATURES: list[str] = [
    "down",
    "qtr",
]

# ---------------------------------------------------------------------------
# Derived unions
# ---------------------------------------------------------------------------

ENGINEERED_FEATURES: list[str] = (
    FORMATION_FEATURES + SITUATIONAL_FEATURES + ENCODED_FEATURES
)

ALL_NUMERIC: list[str] = RAW_NUMERIC + ROLLING_FEATURES + ENGINEERED_FEATURES

CHI2_FEATURES: list[str] = (
    CAT_FEATURES + CHI2_BINARY_FEATURES + CHI2_DISCRETE_FEATURES
)

CANDIDATE_FEATURES: list[str] = sorted(set(ALL_NUMERIC + CAT_FEATURES))

# Every column expected in features_full.parquet (55 columns as of pipeline v1)
EXPECTED_PARQUET_COLUMNS: list[str] = sorted(
    set(TARGET_COLS + DROP_ALWAYS + CANDIDATE_FEATURES)
)


class FeatureSchemaValidationError(ValueError):
    """Raised when parquet columns do not match the feature schema contract."""


def validate_feature_schema(columns: list[str] | set[str]) -> None:
    """
    Verify that *columns* matches the feature schema exactly.

    Each column must belong to exactly one group: target, excluded, or
    candidate. Raises ``FeatureSchemaValidationError`` on mismatch.
    """
    cols = set(columns)
    expected = set(EXPECTED_PARQUET_COLUMNS)

    missing = expected - cols
    extra = cols - expected

    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append(f"missing from parquet: {sorted(missing)}")
        if extra:
            parts.append(f"unassigned in feature schema: {sorted(extra)}")
        raise FeatureSchemaValidationError("; ".join(parts))

    overlap_drop_numeric = set(DROP_ALWAYS) & set(ALL_NUMERIC)
    if overlap_drop_numeric:
        raise FeatureSchemaValidationError(
            f"DROP_ALWAYS overlaps ALL_NUMERIC: {sorted(overlap_drop_numeric)}"
        )

    overlap_drop_chi2 = set(DROP_ALWAYS) & set(CHI2_FEATURES)
    if overlap_drop_chi2:
        raise FeatureSchemaValidationError(
            f"DROP_ALWAYS overlaps CHI2_FEATURES: {sorted(overlap_drop_chi2)}"
        )

    overlap_targets_drop = set(TARGET_COLS) & set(DROP_ALWAYS)
    if overlap_targets_drop:
        raise FeatureSchemaValidationError(
            f"targets overlap DROP_ALWAYS: {sorted(overlap_targets_drop)}"
        )


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_parquet(FEATURES_FULL_PATH)
    validate_feature_schema(df.columns.tolist())
    print(f"Feature schema OK — {len(df.columns)} columns accounted for")
    print(f"  targets:    {len(TARGET_COLS)}")
    print(f"  excluded:   {len(DROP_ALWAYS)}")
    print(f"  candidates: {len(CANDIDATE_FEATURES)}")
    print(f"  ALL_NUMERIC: {len(ALL_NUMERIC)}  |  CHI2_FEATURES: {len(CHI2_FEATURES)}")
