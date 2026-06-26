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
    Keep ONLY pre-play + outcome fields.
    No leakage allowed here.
    """

    cols = [
        # Game state
        "game_id",
        "season",
        "quarter",
        "down",
        "ydstogo",
        "yardline_100",
        "half_seconds_remaining",
        "game_seconds_remaining",
        "score_differential",

        # Play info
        "play_type",
        "shotgun",
        "no_huddle",

        # Teams
        "posteam",
        "defteam",

        # Target
        "yards_gained",
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

    # Drop rows where target is missing
    df = df.dropna(subset=["yards_gained", "down", "ydstogo"])

    # Fill safe binary features
    for col in ["shotgun", "no_huddle"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Fill game state (safe default = neutral)
    if "score_differential" in df.columns:
        df["score_differential"] = df["score_differential"].fillna(0)

    print(f"[INFO] After missing value handling: {df.shape}")
    return df


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure consistent types for modeling.
    """

    int_cols = ["down", "ydstogo", "quarter"]
    float_cols = ["yardline_100", "half_seconds_remaining", "game_seconds_remaining"]

    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype("float").astype("Int64")

    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    if "yards_gained" in df.columns:
        df["yards_gained"] = df["yards_gained"].astype("float32")

    print("[INFO] Dtypes standardized")
    return df


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final sanity filters.
    """

    # Remove impossible values
    df = df[df["ydstogo"] > 0]

    # Remove extreme missing game state
    df = df.dropna(subset=["yardline_100"])

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