"""
Usage (from project root):
    python3 -m src.preprocessing.clean_players

Input:  data/raw/players/season=*.parquet
Output: data/interim/players_clean.parquet
"""
import pandas as pd
from pathlib import Path


RAW_DIR = Path("data/raw/players")
OUT_PATH = Path("data/interim/players_clean.parquet")

# Positions with meaningful offensive stat lines
SKILL_POSITIONS = {"QB", "RB", "WR", "TE", "FB"}


# -------------------------
# Core cleaning functions
# -------------------------
def load_raw_data() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("season=*.parquet"))

    if not files:
        raise FileNotFoundError(f"No raw parquet files found in {RAW_DIR}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    print(f"[INFO] Loaded raw data: {df.shape} ({len(files)} seasons)")
    return df


def filter_skill_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop non-offensive rows (punters, kickers, defensive players, linemen).
    These positions produce no meaningful stat lines for play prediction.
    """
    df = df[df["position"].isin(SKILL_POSITIONS)].copy()

    print(f"[INFO] After filtering to skill positions {SKILL_POSITIONS}: {df.shape}")
    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns needed for modeling and feature engineering.
    Leakage guards (e.g. season_type, fantasy_points) belong at training time.
    """
    player_cols = [
        # Keys
        "player_id",
        "player_name",
        "position_group",
        "recent_team",
        "season",
        "week",

        # QB
        "completions",
        "attempts",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "passing_air_yards",
        "passing_epa",

        # RB
        "carries",
        "rushing_yards",
        "rushing_tds",
        "rushing_epa",

        # WR / TE
        "receptions",
        "targets",
        "receiving_yards",
        "receiving_tds",
        "receiving_air_yards",
        "receiving_epa",

        # Advanced metrics
        "target_share",
        "air_yards_share"
    ]

    cols = [c for c in player_cols if c in df.columns]
    df = df[cols].copy()

    print(f"[INFO] After column selection: {df.shape}")
    return df


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows missing core identifiers.

    Volume/count stats (carries, receptions, etc.) are already 0-filled by
    nfl_data_py for players with no activity that week, so no imputation needed.

    EPA and rate metrics (passing_epa, rushing_epa, receiving_epa,
    target_share, air_yards_share) are left as NaN — they are undefined for
    players without qualifying activity and imputing 0 would incorrectly treat
    "no plays" as "average performance."
    """
    required = ["player_id", "player_name", "position_group", "recent_team", "season", "week"]
    before = len(df)
    df = df.dropna(subset=[c for c in required if c in df.columns])

    dropped = before - len(df)
    if dropped:
        print(f"[WARN] Dropped {dropped} rows missing core identifiers")

    print(f"[INFO] After missing value handling: {df.shape}")
    return df


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize types for downstream feature engineering and storage.

    season, week               → int64
    count/integer stats        → Int64  (nullable, handles edge-season gaps)
    yards / EPA / rate floats  → float32
    categorical strings        → string (pandas StringDtype)
    """
    if "season" in df.columns:
        df["season"] = df["season"].astype("int64")

    int_cols = [
        "week",
        "completions", "attempts", "passing_tds",
        "carries", "rushing_tds",
        "receptions", "targets", "receiving_tds",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype("float").astype("Int64")

    float_cols = [
        "passing_yards", "interceptions",
        "passing_air_yards", "passing_epa",
        "rushing_yards", "rushing_epa",
        "receiving_yards", "receiving_air_yards", "receiving_epa",
        "target_share", "air_yards_share",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    str_cols = [
        "player_id", "player_name", "position_group", "recent_team",
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    print("[INFO] Dtypes standardized")
    return df


def sort_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure deterministic row ordering and remove any duplicate
    (player_id, season, week) rows that can arise from multi-file loads.
    """
    sort_keys = [c for c in ["season", "week", "player_id"] if c in df.columns]
    df = df.sort_values(sort_keys).reset_index(drop=True)

    before = len(df)
    dedup_keys = [c for c in ["player_id", "season", "week"] if c in df.columns]
    df = df.drop_duplicates(subset=dedup_keys, keep="first")

    dropped = before - len(df)
    if dropped:
        print(f"[WARN] Dropped {dropped} duplicate (player_id, season, week) rows")

    print(f"[INFO] After sorting and deduplication: {df.shape}")
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

    df = filter_skill_positions(df)
    df = select_columns(df)
    df = clean_missing_values(df)
    df = convert_dtypes(df)
    df = sort_and_deduplicate(df)

    save_clean_data(df)


if __name__ == "__main__":
    main()
