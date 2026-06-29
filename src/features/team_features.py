"""
Team-level rolling features derived directly from pbp_clean.

Unlike player-level rolling features, team stats are populated on every
offensive play row, eliminating the 55–72% structural missingness caused
by player-ID joins. posteam and defteam are non-null on every play, so
all play rows receive a feature value — only week 1 of each season is NaN
(no prior games exist to roll over).

Rolling pattern
---------------
For each metric M aggregated at the (team, season, week) level:

    weekly_stats
      .sort_values(["team", "season", "week"])
      .groupby(["team", "season"])["M"]
      .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())

shift(1)  →  week N only sees data from weeks 1…N-1 (no leakage).
Rolling resets at every season boundary because the group key includes season.

Functions
---------
build_offense_rolling  →  (posteam, season, week) + 6 offense rolling stats
build_defense_rolling  →  (defteam, season, week) + 5 defense rolling stats
build_team_features    →  joins both onto pbp rows; returns pbp + 11 new columns

Yardage features (L3)
---------------------
off_pass_rate_L3          – rolling mean of team pass rate
off_rush_yds_L3           – rolling mean of yards per rush attempt
off_pass_yds_L3           – rolling mean of yards per pass attempt
def_rush_yds_allowed_L3   – rolling mean of rush yards allowed per attempt
def_pass_yds_allowed_L3   – rolling mean of pass yards allowed per attempt

EPA features (L3)
-----------------
EPA captures down-and-distance efficiency that raw yardage cannot; a team
gaining 3 yards on 3rd-and-2 is very different from 3 yards on 1st-and-10.
off_epa_L3                – rolling mean EPA per play (all play types)
off_rush_epa_L3           – rolling mean EPA on run plays
off_pass_epa_L3           – rolling mean EPA on pass plays
def_epa_L3                – rolling mean EPA allowed per play (all play types)
def_rush_epa_L3           – rolling mean rush EPA allowed
def_pass_epa_L3           – rolling mean pass EPA allowed
"""

import pandas as pd


def build_offense_rolling(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-week offense stats for each team and apply a 3-week lagged
    rolling mean within each (team, season) group.

    Aggregation is over pure run/pass plays only; sack and qb_kneel rows
    are excluded from yardage averages to avoid distorting scheme tendencies.
    EPA is computed across all play types (including sack/kneel) so it
    reflects the true cost of negative plays.

    Parameters
    ----------
    pbp : play-by-play DataFrame (must contain posteam, season, week,
          play_type, yards_gained, epa)

    Returns
    -------
    DataFrame keyed by (posteam, season, week) with columns:
        off_pass_rate_L3  – rolling 3-week mean of team pass rate
        off_rush_yds_L3   – rolling 3-week mean of yards per rush attempt
        off_pass_yds_L3   – rolling 3-week mean of yards per pass attempt
        off_epa_L3        – rolling 3-week mean EPA per play (all play types)
        off_rush_epa_L3   – rolling 3-week mean EPA on run plays
        off_pass_epa_L3   – rolling 3-week mean EPA on pass plays
    """
    df = pbp.copy()
    df["_is_pass"] = (df["play_type"] == "pass").astype(float)
    df["_rush_yds"] = df["yards_gained"].where(df["play_type"] == "run")
    df["_pass_yds"] = df["yards_gained"].where(df["play_type"] == "pass")
    df["_rush_epa"] = df["epa"].where(df["play_type"] == "run")
    df["_pass_epa"] = df["epa"].where(df["play_type"] == "pass")

    weekly = (
        df.groupby(["posteam", "season", "week"], observed=True)
        .agg(
            off_pass_rate=("_is_pass", "mean"),
            off_rush_yds=("_rush_yds", "mean"),
            off_pass_yds=("_pass_yds", "mean"),
            off_epa=("epa", "mean"),
            off_rush_epa=("_rush_epa", "mean"),
            off_pass_epa=("_pass_epa", "mean"),
        )
        .reset_index()
        .sort_values(["posteam", "season", "week"])
    )

    for raw_col, feat_col in [
        ("off_pass_rate", "off_pass_rate_L3"),
        ("off_rush_yds",  "off_rush_yds_L3"),
        ("off_pass_yds",  "off_pass_yds_L3"),
        ("off_epa",       "off_epa_L3"),
        ("off_rush_epa",  "off_rush_epa_L3"),
        ("off_pass_epa",  "off_pass_epa_L3"),
    ]:
        weekly[feat_col] = (
            weekly.groupby(["posteam", "season"])[raw_col]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )

    return weekly[
        ["posteam", "season", "week",
         "off_pass_rate_L3", "off_rush_yds_L3", "off_pass_yds_L3",
         "off_epa_L3", "off_rush_epa_L3", "off_pass_epa_L3"]
    ]


def build_defense_rolling(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-week defense stats for each team and apply a 3-week lagged
    rolling mean within each (team, season) group.

    Yards allowed are measured from the offense's yards_gained column,
    filtered to run and pass play types only. EPA allowed uses the same
    epa column (positive EPA = good for offense = bad for defense).

    Parameters
    ----------
    pbp : play-by-play DataFrame (must contain defteam, season, week,
          play_type, yards_gained, epa)

    Returns
    -------
    DataFrame keyed by (defteam, season, week) with columns:
        def_rush_yds_allowed_L3  – rolling 3-week mean of rush yards allowed
        def_pass_yds_allowed_L3  – rolling 3-week mean of pass yards allowed
        def_epa_L3               – rolling 3-week mean EPA allowed per play
        def_rush_epa_L3          – rolling 3-week mean rush EPA allowed
        def_pass_epa_L3          – rolling 3-week mean pass EPA allowed
    """
    df = pbp.copy()
    df["_rush_yds"] = df["yards_gained"].where(df["play_type"] == "run")
    df["_pass_yds"] = df["yards_gained"].where(df["play_type"] == "pass")
    df["_rush_epa"] = df["epa"].where(df["play_type"] == "run")
    df["_pass_epa"] = df["epa"].where(df["play_type"] == "pass")

    weekly = (
        df.groupby(["defteam", "season", "week"], observed=True)
        .agg(
            def_rush_yds_allowed=("_rush_yds", "mean"),
            def_pass_yds_allowed=("_pass_yds", "mean"),
            def_epa=("epa", "mean"),
            def_rush_epa=("_rush_epa", "mean"),
            def_pass_epa=("_pass_epa", "mean"),
        )
        .reset_index()
        .sort_values(["defteam", "season", "week"])
    )

    for raw_col, feat_col in [
        ("def_rush_yds_allowed", "def_rush_yds_allowed_L3"),
        ("def_pass_yds_allowed", "def_pass_yds_allowed_L3"),
        ("def_epa",              "def_epa_L3"),
        ("def_rush_epa",         "def_rush_epa_L3"),
        ("def_pass_epa",         "def_pass_epa_L3"),
    ]:
        weekly[feat_col] = (
            weekly.groupby(["defteam", "season"])[raw_col]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )

    return weekly[
        ["defteam", "season", "week",
         "def_rush_yds_allowed_L3", "def_pass_yds_allowed_L3",
         "def_epa_L3", "def_rush_epa_L3", "def_pass_epa_L3"]
    ]


def build_team_features(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Master function. Builds offense and defense rolling features and
    left-joins them onto every play row in pbp.

    Join keys:
        offense → (posteam, season, week)
        defense → (defteam, season, week)

    All play rows receive team feature values because posteam/defteam are
    non-null on every offensive play. Week 1 of each season will be NaN
    (no prior games to roll over); all subsequent weeks are non-null.

    Parameters
    ----------
    pbp : cleaned play-by-play DataFrame (output of clean_pbp pipeline)

    Returns
    -------
    pbp with 11 new columns appended:
        off_pass_rate_L3, off_rush_yds_L3, off_pass_yds_L3,
        off_epa_L3, off_rush_epa_L3, off_pass_epa_L3,
        def_rush_yds_allowed_L3, def_pass_yds_allowed_L3,
        def_epa_L3, def_rush_epa_L3, def_pass_epa_L3
    """
    off_rolling = build_offense_rolling(pbp)
    def_rolling = build_defense_rolling(pbp)

    result = pbp.merge(off_rolling, on=["posteam", "season", "week"], how="left")
    result = result.merge(def_rolling, on=["defteam", "season", "week"], how="left")

    return result
