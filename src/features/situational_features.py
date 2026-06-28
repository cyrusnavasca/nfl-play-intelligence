import numpy as np
import pandas as pd


def add_score_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates score-based features using posteam_score and defteam_score.
    """

    df = df.copy()
    df["score_differential"] = df["posteam_score"] - df["defteam_score"]

    return df


def add_time_adjusted_score_diff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Time-weighted score differential:
    Importance of score increases as game progresses.
    """

    df = df.copy()

    progress = 1 - (df["game_seconds_remaining"] / 3600)

    df["time_adjusted_score_diff"] = (
        df["score_differential"] * progress
    )

    return df


def add_field_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Red zone + backed up indicators.
    """

    df = df.copy()

    # Red zone (scoring pressure)
    df["red_zone"] = (df["yardline_100"] <= 20).astype(int)

    # Backed up (danger zone near own end)
    df["backed_up"] = (df["yardline_100"] >= 80).astype(int)

    return df


def add_two_minute_drill_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-minute drill features (end of half or game).
    """

    df = df.copy()

    df["two_minute_game"] = (
        df["game_seconds_remaining"] <= 120
    ).astype(int)

    df["two_minute_half"] = (
        df["half_seconds_remaining"] <= 120
    ).astype(int)

    # Combined signal (useful for modeling urgency)
    df["two_minute_drill"] = (
        (df["two_minute_game"] == 1) |
        (df["two_minute_half"] == 1)
    ).astype(int)

    return df


def build_situational_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function to apply all situational features.
    """

    df = df.copy()

    # Score diff from raw columns
    df = add_score_features(df)

    # Time-weighted score importance
    df = add_time_adjusted_score_diff(df)

    # Field position logic
    df = add_field_position_features(df)

    # End-of-game urgency
    df = add_two_minute_drill_features(df)

    return df