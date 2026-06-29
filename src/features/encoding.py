"""
Categorical encoding functions.

Converts raw string columns to numeric/binary features:
    surface      → is_turf     (1 = turf, 0 = grass)
    roof         → is_indoor   (1 = dome/closed, 0 = outdoors/open)
    season_type  → is_playoffs (1 = POST, 0 = REG)
    posteam_type → is_home     (1 = home, 0 = away)
"""
import numpy as np
import pandas as pd


_TURF_VARIANTS = frozenset({
    "fieldturf", "matrixturf", "sportturf", "astroturf", "a_turf",
    "fieldturf360", "astroplay", "fieldturf classicfield",
})
_INDOOR_ROOF = frozenset({"dome", "closed"})
_OUTDOOR_ROOF = frozenset({"outdoors", "open"})


def encode_surface(df: pd.DataFrame) -> pd.DataFrame:
    """
    is_turf: 1 = artificial turf, 0 = grass, NaN = unknown/blank.

    Known turf variants: fieldturf, matrixturf, sportturf, astroturf, a_turf.
    Blank or unrecognised surface strings return NaN.
    """
    def _map(s: str) -> int | float:
        if pd.isna(s):
            return np.nan
        s = s.lower().strip()
        if not s:
            return np.nan
        if s == "grass":
            return 0
        if s in _TURF_VARIANTS or any(t in s for t in _TURF_VARIANTS):
            return 1
        return np.nan

    df = df.copy()
    df["is_turf"] = df["surface"].map(_map).astype("Int8")
    return df


def encode_roof(df: pd.DataFrame) -> pd.DataFrame:
    """
    is_indoor: 1 = dome/closed (no weather), 0 = outdoors/open, NaN = unknown.

    Indoor games have structural temp/wind missingness, which is handled
    downstream by weather_features.build_weather_features().
    """
    def _map(s: str) -> int | float:
        if pd.isna(s):
            return np.nan
        s = s.lower().strip()
        if s in _INDOOR_ROOF:
            return 1
        if s in _OUTDOOR_ROOF:
            return 0
        return np.nan

    df = df.copy()
    df["is_indoor"] = df["roof"].map(_map).astype("Int8")
    return df


def encode_season_type(df: pd.DataFrame) -> pd.DataFrame:
    """is_playoffs: 1 = POST season, 0 = REG season."""
    df = df.copy()
    df["is_playoffs"] = (
        df["season_type"].str.strip().str.upper() == "POST"
    ).astype("Int8")
    return df


def encode_home(df: pd.DataFrame) -> pd.DataFrame:
    """is_home: 1 = possession team is the home team, 0 = away."""
    df = df.copy()
    df["is_home"] = (
        df["posteam_type"].str.strip().str.lower() == "home"
    ).astype("Int8")
    return df


def build_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all four categorical encodings in sequence."""
    df = encode_surface(df)
    df = encode_roof(df)
    df = encode_season_type(df)
    df = encode_home(df)
    return df
