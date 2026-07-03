"""
Usage (from project root):
    python3 -m src.features.feature_pipeline

Prerequisite:
    python3 -m src.preprocessing.clean_pbp    # produces pbp_clean.parquet
    (no other prerequisites)

Input:
    data/interim/pbp_clean.parquet

Output:
    data/processed/model_dataset.parquet

Pipeline:
    1. Load pbp_clean
    2. Compute team rolling features inline (no cross-table join required)
       - offense: off_pass_rate_L3, off_rush_yds_L3, off_pass_yds_L3
       - defense: def_rush_yds_allowed_L3, def_pass_yds_allowed_L3
    3. Formation features
       - is_heavy_formation, is_spread_formation, box_advantage
    4. Situational features
       - score_differential, time_adjusted_score_diff, red_zone, backed_up,
         two_minute_drill
    5. Categorical encodings
       - is_turf, is_indoor, is_playoffs, is_home

Team stats (posteam, defteam) are non-null on every offensive play, so all
rows receive a feature value. Only week 1 of each season is NaN — no prior
games exist to roll over.
"""
import pandas as pd
from pathlib import Path

from src.features.team_features import build_team_features
from src.features.formation_features import build_formation_features
from src.features.situational_features import build_situational_features
from src.features.encoding import build_encodings


PBP_PATH = Path("data/interim/pbp_clean.parquet")
OUT_PATH = Path("data/processed/model_dataset.parquet")


def build_feature_dataset(pbp_path: Path = PBP_PATH) -> pd.DataFrame:
    """
    Load pbp_clean and attach the engineered feature groups.

    Returns the fully-featured PBP DataFrame with 17 new columns:
        team:        off_pass_rate_L3, off_rush_yds_L3, off_pass_yds_L3,
                     def_rush_yds_allowed_L3, def_pass_yds_allowed_L3
        formation:   is_heavy_formation, is_spread_formation, box_advantage
        situational: score_differential, time_adjusted_score_diff, red_zone,
                     backed_up, two_minute_drill
        encoding:    is_turf, is_indoor, is_playoffs, is_home
    """
    pbp = pd.read_parquet(pbp_path)
    print(f"[INFO] Loaded pbp_clean: {pbp.shape}")

    pbp = build_team_features(pbp)
    pbp = build_formation_features(pbp)
    pbp = build_situational_features(pbp)
    pbp = build_encodings(pbp)
    print(f"[INFO] Final feature dataset: {pbp.shape}")
    return pbp


def save_feature_dataset(df: pd.DataFrame, out_path: Path = OUT_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[DONE] Saved → {out_path}")
    print(f"[DONE] Columns ({len(df.columns)}): {df.columns.tolist()}")


def main():
    df = build_feature_dataset()
    save_feature_dataset(df)


if __name__ == "__main__":
    main()
