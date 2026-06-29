"""
Usage (from project root):
    python3 -m src.preprocessing.clean_pbp

Input:  data/raw/pbp/season=*.parquet
Output: data/interim/pbp_clean.parquet
"""
import pandas as pd
from pathlib import Path


RAW_DIR = Path("data/raw/pbp")
OUT_PATH = Path("data/interim/pbp_clean.parquet")


# -------------------------
# Core cleaning function
# -------------------------
def load_raw_data() -> pd.DataFrame:
    files = list(RAW_DIR.glob("season=*.parquet"))

    if not files:
        raise FileNotFoundError("No raw parquet files found in data/raw/pbp/")

    df_list = [pd.read_parquet(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)

    print(f"[INFO] Loaded raw data: {df.shape}")
    return df


def filter_offensive_plays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only meaningful offensive plays for modeling yards gained.
    """

    # Keep only standard play types
    valid_play_types = ["run", "pass", "sack", "qb_kneel", "qb_spike"]

    df = df[df["play_type"].isin(valid_play_types)].copy()

    # Remove special cases that distort yardage learning
    df = df[df["play_type"] != "qb_spike"]

    print(f"[INFO] After filtering plays: {df.shape}")
    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep pre-play context + outcome fields.
    Broad selection preserves flexibility for downstream filtering.
    No leakage allowed in model features — filter appropriately at training time.
    """

    cols = [
        # ── Identifiers ──────────────────────────────────────────────
        "game_id",
        "play_id",
        "season",
        "week",
        "game_date",
        "season_type",              # REG / POST

        # ── Teams & matchup ──────────────────────────────────────────
        "home_team",
        "away_team",
        "posteam",
        "defteam",
        "posteam_type",             # home / away

        # ── Game state ───────────────────────────────────────────────
        "game_half",                # Half1 / Half2 / Overtime
        "qtr",
        "quarter_seconds_remaining",
        "half_seconds_remaining",
        "game_seconds_remaining",
        "down",
        "ydstogo",
        "yardline_100",

        # ── Score & win probability ───────────────────────────────────
        "posteam_score",
        "defteam_score",

        # ── Play type & formation ─────────────────────────────────────
        "play_type",
        "offense_formation",
        "defenders_in_box",
        "number_of_pass_rushers",
        "offense_personnel",
        "defense_personnel",

        # ── Environment ──────────────────────────────────────────────
        "roof",                     # open / closed / dome / outdoors
        "surface",                  # grass / turf
        "temp",
        "wind",

        # ── Play efficiency ───────────────────────────────────────────
        "epa",                      # expected points added per play

        # ── Main Target ───────────────────────────────────────────────
        "yards_gained"
    ]

    # Keep only columns that exist (robustness across seasons)
    cols = [c for c in cols if c in df.columns]

    df = df[cols].copy()

    print(f"[INFO] Selected columns: {df.shape}")
    return df


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing data in a simple, deterministic way.
    """

    # Drop rows where core fields are missing
    df = df.dropna(subset=["yards_gained", "down", "ydstogo", "yardline_100"])


    # Neutral default for score context
    for col in ["posteam_score", "defteam_score"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    print(f"[INFO] After missing value handling: {df.shape}")
    return df


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure consistent types for modeling.

    season                             → int64
    week, qtr, down, ydstogo,
    posteam/defteam scores,
    defenders_in_box,
    number_of_pass_rushers             → Int64 (nullable int)
    time/field floats, temp, wind, epa → float32
    yards_gained                       → float32
    """

    if "season" in df.columns:
        df["season"] = df["season"].astype("int64")

    int_cols = [
        "week", "qtr", "down", "ydstogo",
        "posteam_score", "defteam_score",
        "defenders_in_box", "number_of_pass_rushers",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype("float").astype("Int64")

    float_cols = [
        "quarter_seconds_remaining", "half_seconds_remaining",
        "game_seconds_remaining", "yardline_100",
        "temp", "wind", "epa",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    if "yards_gained" in df.columns:
        df["yards_gained"] = df["yards_gained"].astype("float32")

    print("[INFO] Dtypes standardized")
    return df


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["ydstogo"] > 0]

    print(f"[INFO] After final filtering: {df.shape}")
    return df


def save_clean_data(df: pd.DataFrame):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(OUT_PATH, index=False)

    print(f"[DONE] Saved cleaned dataset → {OUT_PATH}")
    print(f"[DONE] Final shape: {df.shape}")


# -------------------------
# Main pipeline
# -------------------------
def main():
    df = load_raw_data()

    df = filter_offensive_plays(df)
    df = select_columns(df)
    df = clean_missing_values(df)
    df = convert_dtypes(df)
    df = remove_invalid_rows(df)

    save_clean_data(df)


if __name__ == "__main__":
    main()