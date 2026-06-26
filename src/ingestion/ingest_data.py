import argparse
from pathlib import Path
import nfl_data_py as nfl
import pandas as pd


RAW_DIR = Path("data/raw")


def ensure_dirs():
    """Create raw data directories if they don't exist."""
    (RAW_DIR / "pbp").mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "rosters").mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "schedules").mkdir(parents=True, exist_ok=True)


def save_parquet(df: pd.DataFrame, path: Path):
    """Save dataframe as parquet with consistent formatting."""
    df.to_parquet(path, index=False)


def ingest_pbp(years):
    """Ingest play-by-play data."""
    print(f"[INFO] Downloading PBP data for years: {years}")

    df = nfl.import_pbp_data(years=years)

    # Basic sanity check
    if df is None or df.empty:
        raise ValueError("Play-by-play data download failed or returned empty DataFrame")

    for year in years:
        df_year = df[df["season"] == year].copy()

        out_path = RAW_DIR / "pbp" / f"season={year}.parquet"
        save_parquet(df_year, out_path)

        print(f"[INFO] Saved PBP {year}: {df_year.shape} → {out_path}")


def ingest_rosters(years):
    """Ingest roster data."""
    print(f"[INFO] Downloading roster data for years: {years}")

    df = nfl.import_seasonal_rosters(years=years)

    if df is None or df.empty:
        print("[WARN] No roster data returned")
        return

    for year in years:
        df_year = df[df["season"] == year].copy()

        out_path = RAW_DIR / "rosters" / f"season={year}.parquet"
        save_parquet(df_year, out_path)

        print(f"[INFO] Saved rosters {year}: {df_year.shape} → {out_path}")


def ingest_schedules(years):
    """Ingest schedule data."""
    print(f"[INFO] Downloading schedule data for years: {years}")

    df = nfl.import_schedules(years=years)

    if df is None or df.empty:
        print("[WARN] No schedule data returned")
        return

    for year in years:
        df_year = df[df["season"] == year].copy()

        out_path = RAW_DIR / "schedules" / f"season={year}.parquet"
        save_parquet(df_year, out_path)

        print(f"[INFO] Saved schedules {year}: {df_year.shape} → {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="NFLVerse data ingestion pipeline")

    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["pbp", "rosters", "schedules"],
        help="Which datasets to ingest"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    years = list(range(args.start_year, args.end_year + 1))

    ensure_dirs()

    if "pbp" in args.datasets:
        ingest_pbp(years)

    if "rosters" in args.datasets:
        ingest_rosters(years)

    if "schedules" in args.datasets:
        ingest_schedules(years)

    print("[DONE] Ingestion complete.")


if __name__ == "__main__":
    main()