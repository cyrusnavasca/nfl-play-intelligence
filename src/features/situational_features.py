import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Score features
# ---------------------------------------------------------------------------

def add_score_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    score_differential  – posteam advantage (positive = leading)
    total_score         – combined points scored
    """
    df = df.copy()
    df["score_differential"] = df["posteam_score"] - df["defteam_score"]
    df["total_score"] = df["posteam_score"] + df["defteam_score"]
    return df


def add_time_adjusted_score_diff(df: pd.DataFrame) -> pd.DataFrame:
    """
    time_adjusted_score_diff – score_differential weighted by game progress.
    Importance of the lead grows as the game progresses.
    Requires score_differential (call add_score_features first).
    """
    df = df.copy()
    progress = 1 - (df["game_seconds_remaining"] / 3600)
    df["time_adjusted_score_diff"] = df["score_differential"] * progress
    return df


def add_score_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    score_diff_abs  – absolute value of the lead
    is_blowout      – lead >= 17 (conservative run/prevent-defence territory)
    is_close        – lead <= 3  (one-score game, pass-heavy)
    game_script     – score_differential × game_progress (urgency signal)

    Requires score_differential (call add_score_features first).
    """
    df = df.copy()
    df["score_diff_abs"] = df["score_differential"].abs()
    df["is_blowout"] = (df["score_diff_abs"] >= 17).astype("Int8")
    df["is_close"] = (df["score_diff_abs"] <= 3).astype("Int8")
    game_progress = 1 - (df["game_seconds_remaining"] / 3600)
    df["game_script"] = df["score_differential"] * game_progress
    return df


# ---------------------------------------------------------------------------
# Game progress features
# ---------------------------------------------------------------------------

def add_game_progress_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    game_progress  – fraction of regulation elapsed (0.0 at kickoff → 1.0 at whistle)
    half_progress  – fraction of current half elapsed
    is_overtime    – 1 if qtr == 5 (OT plays have distinct tendencies)
    """
    df = df.copy()
    df["game_progress"] = 1 - (df["game_seconds_remaining"] / 3600)
    df["half_progress"] = 1 - (df["half_seconds_remaining"] / 1800)
    df["is_overtime"] = (df["qtr"] == 5).astype("Int8")
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
    two_minute_game  – <= 120 s left in regulation
    two_minute_half  – <= 120 s left in the current half
    two_minute_drill – either flag is 1
    """
    df = df.copy()
    df["two_minute_game"] = (df["game_seconds_remaining"] <= 120).astype("Int8")
    df["two_minute_half"] = (df["half_seconds_remaining"] <= 120).astype("Int8")
    df["two_minute_drill"] = (
        (df["two_minute_game"] == 1) | (df["two_minute_half"] == 1)
    ).astype("Int8")
    return df


# ---------------------------------------------------------------------------
# Down & distance features
# ---------------------------------------------------------------------------

def add_down_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    distance_bucket   – ordinal: 1 = short (1–3 yd), 2 = medium (4–6), 3 = long (7+)
    ydstogo_normalized – ydstogo / 10  (typical 10-yard first-down normalised to 1.0)
    yards_available   – 100 - yardline_100 (yards between LOS and end zone)
    ydstogo_ratio     – ydstogo / yards_available (fraction of field needed)
    is_short_yardage  – down >= 2 AND ydstogo <= 2 (classic power-run situation)
    is_third_and_long – down == 3 AND ydstogo >= 7  (pass-heavy; defense knows it)
    is_fourth_down    – down == 4 (distinct decision point)
    """
    df = df.copy()

    df["distance_bucket"] = pd.cut(
        df["ydstogo"],
        bins=[0, 3, 6, np.inf],
        labels=[1, 2, 3],
        right=True,
    ).astype("Int8")

    df["ydstogo_normalized"] = df["ydstogo"] / 10.0
    df["yards_available"] = 100 - df["yardline_100"]
    df["ydstogo_ratio"] = df["ydstogo"] / df["yards_available"].replace(0, np.nan)

    df["is_short_yardage"] = (
        (df["down"] >= 2) & (df["ydstogo"] <= 2)
    ).astype("Int8")
    df["is_third_and_long"] = (
        (df["down"] == 3) & (df["ydstogo"] >= 7)
    ).astype("Int8")
    df["is_fourth_down"] = (df["down"] == 4).astype("Int8")

    return df


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def build_situational_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all situational feature functions in dependency order.

    Call after raw PBP columns are available; does not require any
    previously engineered columns except raw game state fields.
    """
    df = add_score_features(df)
    df = add_time_adjusted_score_diff(df)
    df = add_score_context_features(df)
    df = add_game_progress_features(df)
    df = add_field_position_features(df)
    df = add_two_minute_drill_features(df)
    df = add_down_distance_features(df)
    return df
