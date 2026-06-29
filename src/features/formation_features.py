import re

import numpy as np
import pandas as pd


def _extract_position_count(personnel_str: pd.Series, position: str) -> pd.Series:
    """
    Extract the count of a given position abbreviation from an NFL personnel string.
    E.g. "1 RB, 2 TE, 2 WR" → position="RB" → 1.
    Returns NaN where the string is null or the position is absent.
    """
    pattern = rf"(\d+)\s+{position}\b"

    def _parse(s):
        if pd.isna(s):
            return np.nan
        m = re.search(pattern, str(s))
        return int(m.group(1)) if m else np.nan

    return personnel_str.map(_parse)


def parse_offense_personnel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the `offense_personnel` string into numeric position counts.

    New columns: off_rb_count, off_te_count, off_wr_count
    """
    df = df.copy()
    df["off_rb_count"] = _extract_position_count(df["offense_personnel"], "RB")
    df["off_te_count"] = _extract_position_count(df["offense_personnel"], "TE")
    df["off_wr_count"] = _extract_position_count(df["offense_personnel"], "WR")
    return df


def add_formation_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary formation indicators.

    Requires off_rb_count, off_te_count, off_wr_count to already exist
    (call parse_offense_personnel first).

    New columns:
        is_heavy_formation  – off_rb_count + off_te_count >= 3 (run-leaning)
        is_spread_formation – off_wr_count >= 3               (pass-leaning)
    """
    df = df.copy()

    heavy_mask = (df["off_rb_count"] + df["off_te_count"]) >= 3
    df["is_heavy_formation"] = heavy_mask.astype("Int8")

    spread_mask = df["off_wr_count"] >= 3
    df["is_spread_formation"] = spread_mask.astype("Int8")

    return df


def add_box_advantage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add box_advantage: numerical advantage for the offense in the run game.

    box_advantage = off_rb_count + off_te_count - defenders_in_box

    Positive  → offense outnumbers box defenders (run-favorable)
    Negative  → defense has extra bodies in the box (stacked box)

    Requires off_rb_count, off_te_count (call parse_offense_personnel first)
    and defenders_in_box in the dataframe.
    """
    df = df.copy()
    df["box_advantage"] = (
        df["off_rb_count"] + df["off_te_count"] - df["defenders_in_box"]
    )
    return df


def build_formation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function: parse personnel strings and derive all formation features.

    Applied in order:
        1. parse_offense_personnel   → off_rb_count, off_te_count, off_wr_count
        2. add_formation_flags       → is_heavy_formation, is_spread_formation
        3. add_box_advantage         → box_advantage
    """
    df = parse_offense_personnel(df)
    df = add_formation_flags(df)
    df = add_box_advantage(df)
    return df
