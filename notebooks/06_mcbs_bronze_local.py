"""
06_mcbs_bronze_local.py — MCBS Bronze Layer (Raw Ingest)

Reads extracted MCBS data from CMS ZIP subdirectories and converts to
per-year Parquet files. Handles the actual CMS distribution format:

  Survey File:      SFPUF{YEAR}_Data/sfpuf{year}_1_fall.csv  (3 seasonal rounds)
                    SFPUF{YEAR}_Data/sfpuf{year}_2_winter.csv
                    SFPUF{YEAR}_Data/sfpuf{year}_3_summer.csv
                    -> Survey rounds are MERGED on PUF_ID (same benes, diff questions)

  Cost Supplement:  CSPUF{YEAR}_Data/cspuf{year}.csv  (single file)

IMPORTANT: Survey File and Cost Supplement PUF_IDs do NOT overlap.
They are independent datasets and cannot be joined at the beneficiary level.
The Cost Supplement includes its own demographics (CSP_AGE, CSP_SEX, etc.).

Input:  data/mcbs/SFPUF*_Data/, data/mcbs/CSPUF*_Data/
Output: local_pipeline/mcbs_bronze/survey_{YEAR}.parquet
        local_pipeline/mcbs_bronze/cost_{YEAR}.parquet

Usage:
    python notebooks/06_mcbs_bronze_local.py
    python notebooks/06_mcbs_bronze_local.py --input-dir data/mcbs
"""

import os
import re
import glob
import argparse
import pandas as pd

_PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT   = os.path.join(_PROJECT_ROOT, "data", "mcbs")
DEFAULT_OUTPUT  = os.path.join(_PROJECT_ROOT, "local_pipeline", "mcbs_bronze")


def ingest_survey_year(year_dir: str, year: int, output_dir: str):
    """Merge 3 seasonal survey CSV rounds on PUF_ID into one Parquet."""
    out_path = os.path.join(output_dir, f"survey_{year}.parquet")
    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  survey_{year}: Already exists ({size_mb:.1f} MB) — skipping")
        return

    # Find CSVs — handle naming variants: sfpuf2022_1_fall.csv OR sfpuf_2017_1_fall.csv
    csv_files = sorted(glob.glob(os.path.join(year_dir, "*.csv")))
    if not csv_files:
        print(f"  survey_{year}: No CSV files found in {year_dir}")
        return

    print(f"  survey_{year}: {len(csv_files)} round(s)")
    rounds = []
    for csv_path in csv_files:
        fname = os.path.basename(csv_path).lower()
        # Determine round label from filename
        if "fall" in fname:
            round_label = "fall"
        elif "winter" in fname:
            round_label = "winter"
        elif "summer" in fname:
            round_label = "summer"
        else:
            print(f"    [WARN] Skipping unrecognized CSV: {fname}")
            continue

        df = pd.read_csv(csv_path, dtype=str, low_memory=False)
        print(f"    {round_label}: {len(df):,} rows, {len(df.columns)} cols")
        rounds.append((round_label, df))

    if not rounds:
        print(f"  survey_{year}: No valid round CSVs found")
        return

    # Merge rounds on PUF_ID (winter/summer are subsets of fall)
    # Start with the largest round (fall), then left-join others
    rounds.sort(key=lambda x: len(x[1]), reverse=True)
    merged = rounds[0][1]
    for round_label, rdf in rounds[1:]:
        # Drop duplicate columns (PUF_ID, SURVEYYR, VERSION exist in all)
        shared_cols = set(merged.columns) & set(rdf.columns)
        join_col = "PUF_ID"
        if join_col not in shared_cols:
            print(f"    [WARN] No PUF_ID in {round_label} — concatenating instead")
            continue
        rdf_unique = rdf.drop(columns=[c for c in shared_cols if c != join_col], errors="ignore")
        if len(rdf_unique.columns) <= 1:
            continue  # nothing new to add
        merged = merged.merge(rdf_unique, on=join_col, how="left")
        print(f"    merged {round_label}: now {len(merged.columns)} cols")

    # Provenance
    merged["_src_type"] = "survey"
    merged["_mcbs_year"] = str(year)
    n_benes = merged["PUF_ID"].nunique() if "PUF_ID" in merged.columns else len(merged)

    merged.to_parquet(out_path, index=False)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"    -> survey_{year}.parquet: {len(merged):,} benes, {len(merged.columns)} cols ({size_mb:.1f} MB)")


def ingest_cost_year(year_dir: str, year: int, output_dir: str):
    """Read single cost supplement CSV to Parquet."""
    out_path = os.path.join(output_dir, f"cost_{year}.parquet")
    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  cost_{year}: Already exists ({size_mb:.1f} MB) — skipping")
        return

    csv_files = sorted(glob.glob(os.path.join(year_dir, "*.csv")))
    if not csv_files:
        print(f"  cost_{year}: No CSV files found in {year_dir}")
        return

    # Cost supplement has a single CSV
    csv_path = csv_files[0]
    print(f"  cost_{year}: reading {os.path.basename(csv_path)}...", end=" ")
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)

    # Provenance
    df["_src_type"] = "cost"
    df["_mcbs_year"] = str(year)
    n_benes = df["PUF_ID"].nunique() if "PUF_ID" in df.columns else len(df)

    df.to_parquet(out_path, index=False)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"{len(df):,} benes, {len(df.columns)} cols -> cost_{year}.parquet ({size_mb:.1f} MB)")


def ingest(input_dir: str, output_dir: str):
    """Discover MCBS year directories and ingest all to Bronze Parquet."""
    input_dir = os.path.abspath(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Discover subdirectories: SFPUF{YEAR}_Data, CSPUF{YEAR}_Data
    survey_dirs = {}
    cost_dirs = {}
    for entry in sorted(os.listdir(input_dir)):
        full_path = os.path.join(input_dir, entry)
        if not os.path.isdir(full_path):
            continue
        match = re.match(r"SFPUF(\d{4})_Data", entry, re.IGNORECASE)
        if match:
            survey_dirs[int(match.group(1))] = full_path
            continue
        match = re.match(r"CSPUF(\d{4})_Data", entry, re.IGNORECASE)
        if match:
            cost_dirs[int(match.group(1))] = full_path

    if not survey_dirs and not cost_dirs:
        raise FileNotFoundError(
            f"No MCBS data directories found in '{input_dir}'\n"
            f"Expected directories like: SFPUF2022_Data/, CSPUF2022_Data/\n"
            f"Download and extract ZIP files from CMS, or run pull_mcbs_data.py."
        )

    print(f"MCBS Bronze Ingest")
    print(f"  Survey dirs: {sorted(survey_dirs.keys())}")
    print(f"  Cost dirs:   {sorted(cost_dirs.keys())}")
    print(f"  Output: {output_dir}\n")

    # Ingest survey files
    for year in sorted(survey_dirs.keys()):
        try:
            ingest_survey_year(survey_dirs[year], year, output_dir)
        except Exception as e:
            print(f"  survey_{year}: ERROR — {e}")

    # Ingest cost files
    for year in sorted(cost_dirs.keys()):
        try:
            ingest_cost_year(cost_dirs[year], year, output_dir)
        except Exception as e:
            print(f"  cost_{year}: ERROR — {e}")

    print(f"\nBronze ingest complete: {len(survey_dirs)} survey + {len(cost_dirs)} cost years")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCBS Bronze layer: ingest CMS data directories to Parquet."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT,
                        help=f"Directory with MCBS subdirs (default: {DEFAULT_INPUT})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT,
                        help=f"Output directory (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()
    ingest(args.input_dir, args.output_dir)
