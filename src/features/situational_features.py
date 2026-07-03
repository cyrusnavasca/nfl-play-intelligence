import pandas as pd


# ---------------------------------------------------------------------------
# Score features
# ---------------------------------------------------------------------------

def add_score_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    score_differential – posteam advantage (positive = leading).
    """
    df = df.copy()
    df["score_differential"] = df["posteam_score"] - df["defteam_score"]
    return df


def add_time_adjusted_score_diff(df: pd.DataFrame) -> pd.DataFrame:
    """
    time_adjusted_score_diff – score_differential weighted by game progress.
    Importance of the lead grows as the game progresses. Game progress is
    computed inline (1 - game_seconds_remaining / 3600) so this stays
    self-contained.
    Requires score_differential (call add_score_features first).
    """
    df = df.copy()
    progress = 1 - (df["game_seconds_remaining"] / 3600)
    df["time_adjusted_score_diff"] = df["score_differential"] * progress
    return df


# ---------------------------------------------------------------------------
# Field position features
# ---------------------------------------------------------------------------

def add_field_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    red_zone  – yardline_100 <= 20 (scoring pressure)
    backed_up – yardline_100 >= 80 (danger zone near own end zone)
    """
    df = df.copy()
    df["red_zone"] = (df["yardline_100"] <= 20).astype("Int8")
    df["backed_up"] = (df["yardline_100"] >= 80).astype("Int8")
    return df


# ---------------------------------------------------------------------------
# Two-minute drill features
# ---------------------------------------------------------------------------

def add_two_minute_drill_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    two_minute_drill – <= 120 s left in regulation OR in the current half.

    Computed inline from the raw time columns so it carries no dependency on
    any intermediate two-minute flags.
    """
    df = df.copy()
    df["two_minute_drill"] = (
        (df["game_seconds_remaining"] <= 120) | (df["half_seconds_remaining"] <= 120)
    ).astype("Int8")
    return df


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def build_situational_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all situational feature functions in dependency order.

    Output columns: score_differential, time_adjusted_score_diff, red_zone,
    backed_up, two_minute_drill.

    Call after raw PBP columns are available; does not require any
    previously engineered columns except raw game state fields.
    """
    df = add_score_features(df)             # score_differential
    df = add_time_adjusted_score_diff(df)   # time_adjusted_score_diff
    df = add_field_position_features(df)     # red_zone, backed_up
    df = add_two_minute_drill_features(df)  # two_minute_drill (inline)
    return df
