import re

import numpy as np
import pandas as pd


# Offense formations where the QB lines up in the shotgun (EMPTY is always a
# shotgun-aligned set). Used to flag "QB in gun" passing looks.
_GUN_FORMATIONS = frozenset({"SHOTGUN", "EMPTY"})


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

    New columns: off_rb_count, off_te_count
    """
    df = df.copy()
    df["off_rb_count"] = _extract_position_count(df["offense_personnel"], "RB")
    df["off_te_count"] = _extract_position_count(df["offense_personnel"], "TE")
    return df


def add_qb_in_gun(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add is_qb_in_gun: 1 when the QB lines up in the shotgun.

    True for SHOTGUN and EMPTY formations (EMPTY is always shotgun-aligned),
    0 for every other formation (SINGLEBACK, I_FORM, PISTOL, JUMBO, WILDCAT,
    UNDER CENTER). NaN offense_formation → 0.

    A shotgun look is a strong pre-snap pass-tendency signal; PISTOL is
    intentionally excluded (shallow-gun, more run-balanced).
    """
    df = df.copy()
    df["is_qb_in_gun"] = df["offense_formation"].isin(_GUN_FORMATIONS).astype("Int8")
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


_TEMP_COUNT_COLS = ["off_rb_count", "off_te_count"]


def build_formation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function: derive the formation features.

    Output columns: is_qb_in_gun, box_advantage.

    Applied in order:
        1. parse_offense_personnel   → off_rb_count, off_te_count (temp)
        2. add_qb_in_gun             → is_qb_in_gun
        3. add_box_advantage         → box_advantage
        4. drop the temporary count columns so only the 2 features are added

    is_qb_in_gun derives directly from offense_formation; box_advantage derives
    from the parsed personnel counts, which are internal-only and not emitted.
    """
    df = parse_offense_personnel(df)
    df = add_qb_in_gun(df)
    df = add_box_advantage(df)
    df = df.drop(columns=_TEMP_COUNT_COLS)
    return df
