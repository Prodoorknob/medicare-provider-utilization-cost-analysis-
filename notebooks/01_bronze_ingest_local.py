"""
01_bronze_ingest_local.py — Local Bronze Ingest
Walks the partitioned_data/ tree (STATE/PROVIDER_TYPE.parquet),
consolidates all parquets into a single Bronze parquet at
local_pipeline/bronze/bronze.parquet.

Usage:
    python notebooks/01_bronze_ingest_local.py
    python notebooks/01_bronze_ingest_local.py --input-dir partitioned_data --output-dir local_pipeline/bronze
"""

import os
import glob
import argparse
import pandas as pd

# partitioned_data/ sits one level above the project root (sibling of this repo)
_PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT  = os.path.join(_PROJECT_ROOT, "..", "partitioned_data")
DEFAULT_OUTPUT = os.path.join(_PROJECT_ROOT, "local_pipeline", "bronze")
BRONZE_FILE    = "bronze.parquet"


def ingest(input_dir: str, output_dir: str):
    input_dir = os.path.abspath(input_dir)

    # Structure: {input_dir}/{STATE}/*.parquet  (one level deep, no further nesting)
    pattern = os.path.join(input_dir, "*", "*.parquet")
    files   = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No parquet files found under '{input_dir}'.\n"
            f"  Expected layout: {input_dir}/{{STATE}}/{{PROVIDER_TYPE}}.parquet\n"
            f"  Run partition_medicare_data.py + csv_to_parquet.py first."
        )

    print(f"Found {len(files)} partition files across "
          f"{len(set(os.path.basename(os.path.dirname(f)) for f in files))} states — consolidating...")

    chunks = []
    for i, f in enumerate(files, 1):
        df = pd.read_parquet(f)
        # Tag provenance from the STATE/PROVIDER_TYPE.parquet path structure
        df["_src_state"]    = os.path.basename(os.path.dirname(f))
        df["_src_provider"] = os.path.splitext(os.path.basename(f))[0]
        chunks.append(df)
        if i % 200 == 0:
            print(f"  Loaded {i}/{len(files)} files...")

    bronze = pd.concat(chunks, ignore_index=True)
    assert "year" in bronze.columns, (
        "Missing 'year' column in source parquets. "
        "Re-run partition_medicare_data.py + csv_to_parquet.py to inject year from filenames."
    )
    print(f"Total rows ingested: {len(bronze):,}  |  Columns: {len(bronze.columns)}")
    print(f"  Years present: {sorted(bronze['year'].unique())}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, BRONZE_FILE)
    bronze.to_parquet(out_path, index=False)
    print(f"Bronze parquet written -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate partitioned parquets into Bronze layer.")
    parser.add_argument("--input-dir",  default=DEFAULT_INPUT,
                        help=f"Root of partitioned_data/ tree (default: {DEFAULT_INPUT})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT,
                        help=f"Output directory for bronze parquet (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()
    ingest(args.input_dir, args.output_dir)
