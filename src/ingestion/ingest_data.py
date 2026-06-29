"""
Data ingestion pipeline.

Usage (from project root):
    python3 -m src.ingestion.ingest_data --start-year 2018 --end-year 2025
    python3 -m src.ingestion.ingest_data --start-year 2018 --end-year 2025 --datasets pbp
    python3 -m src.ingestion.ingest_data --start-year 2018 --end-year 2025 --datasets players

Output:
    data/raw/pbp/season=<year>.parquet
    data/raw/players/season=<year>.parquet
"""
import argparse
from pathlib import Path
import nfl_data_py as nfl
import pandas as pd


RAW_DIR = Path("data/raw")


def ensure_dirs():
    (RAW_DIR / "pbp").mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "players").mkdir(parents=True, exist_ok=True)


def save_parquet(df: pd.DataFrame, path: Path):
    df.to_parquet(path, index=False)


def ingest_pbp(years):
    print(f"[INFO] Downloading PBP data for years: {years}")

    df = nfl.import_pbp_data(years=years)

    if df is None or df.empty:
        raise ValueError("Play-by-play data download failed or returned empty DataFrame")

    for year in years:
        df_year = df[df["season"] == year].copy()

        out_path = RAW_DIR / "pbp" / f"season={year}.parquet"
        save_parquet(df_year, out_path)

        print(f"[INFO] Saved PBP {year}: {df_year.shape} → {out_path}")


def ingest_players(years):
    # Weekly data lags behind — cap at 2024 until upstream is updated
    years = [y for y in years if y <= 2024]
    if not years:
        print("[WARN] No valid years for weekly player data")
        return

    print(f"[INFO] Downloading weekly player data for years: {years}")

    df = nfl.import_weekly_data(years=years)

    if df is None or df.empty:
        print("[WARN] No weekly player data returned")
        return

    for year in years:
        df_year = df[df["season"] == year].copy()

        out_path = RAW_DIR / "players" / f"season={year}.parquet"
        save_parquet(df_year, out_path)

        print(f"[INFO] Saved players {year}: {df_year.shape} → {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="NFLVerse data ingestion pipeline")

    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["pbp", "players"],
        help="Which datasets to ingest"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    years = list(range(args.start_year, args.end_year + 1))

    ensure_dirs()

    if "pbp" in args.datasets:
        ingest_pbp(years)

    if "players" in args.datasets:
        ingest_players(years)

    print("[DONE] Ingestion complete.")


if __name__ == "__main__":
    main()
