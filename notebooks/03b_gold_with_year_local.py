"""
03b_gold_with_year_local.py — Regenerate Gold parquets WITH year column preserved.

Outputs to local_pipeline/gold_year/ so existing gold/ is untouched.
Upload gold_year/*.parquet to Drive for V2_10 (derived feature channels).

The existing 03_gold_features_local.py already supports year — this wrapper
just targets a separate output directory and validates year is present.

Usage:
    python notebooks/03b_gold_with_year_local.py
    python notebooks/03b_gold_with_year_local.py --provider-data-dir data/
    python notebooks/03b_gold_with_year_local.py --no-gpu
"""

import os
import sys
import argparse
import pyarrow.parquet as pq

# Import the existing gold pipeline
_NOTEBOOKS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _NOTEBOOKS_DIR)
from importlib.util import spec_from_file_location, module_from_spec

spec = spec_from_file_location(
    "gold_features",
    os.path.join(_NOTEBOOKS_DIR, "03_gold_features_local.py"),
)
gold_mod = module_from_spec(spec)
spec.loader.exec_module(gold_mod)

_PROJECT_ROOT = os.path.dirname(_NOTEBOOKS_DIR)
DEFAULT_SILVER    = os.path.join(_PROJECT_ROOT, "local_pipeline", "silver")
DEFAULT_GOLD_YEAR = os.path.join(_PROJECT_ROOT, "local_pipeline", "gold_year")
DEFAULT_PROV      = os.path.join(_PROJECT_ROOT, "..", "data")


def validate_silver_has_year(silver_dir: str):
    """Check that Silver parquets contain a 'year' column."""
    import glob
    silver_files = sorted(glob.glob(os.path.join(silver_dir, "*.parquet")))
    if not silver_files:
        raise FileNotFoundError(f"No silver parquets in {silver_dir}")

    schema = pq.read_schema(silver_files[0])
    if "year" not in schema.names:
        print("=" * 70)
        print("ERROR: Silver parquets do NOT contain 'year' column!")
        print("=" * 70)
        print()
        print("Available columns:", schema.names)
        print()
        print("Fix: Re-run the partition + silver pipeline to inject year:")
        print("  python partition_medicare_data.py --input-dir data --output-dir partitioned_data")
        print("  python csv_to_parquet.py --dir partitioned_data")
        print("  python notebooks/01_bronze_ingest_local.py")
        print("  python notebooks/02_silver_clean_local.py")
        print()
        print("Then re-run this script.")
        sys.exit(1)

    print(f"✓ Silver has 'year' column (verified in {os.path.basename(silver_files[0])})")
    print(f"  Silver columns: {schema.names}")


def validate_gold_has_year(gold_dir: str):
    """Post-run check: verify Gold output includes year."""
    import glob
    gold_files = sorted(glob.glob(os.path.join(gold_dir, "*.parquet")))
    if not gold_files:
        print("[WARN] No gold parquets produced!")
        return False

    schema = pq.read_schema(gold_files[0])
    if "year" not in schema.names:
        print("[WARN] Gold parquets still missing 'year' — check engineer_df logic")
        return False

    print(f"\n✓ Gold output has 'year' column ({len(gold_files)} state files)")
    print(f"  Gold columns: {schema.names}")

    # Quick row count
    total = sum(pq.read_metadata(f).num_rows for f in gold_files)
    print(f"  Total rows: {total:,}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate Gold parquets WITH year column -> gold_year/"
    )
    parser.add_argument("--silver-dir", default=DEFAULT_SILVER,
                        help=f"Silver parquets directory (default: {DEFAULT_SILVER})")
    parser.add_argument("--gold-dir", default=DEFAULT_GOLD_YEAR,
                        help=f"Output directory (default: {DEFAULT_GOLD_YEAR})")
    parser.add_argument("--provider-data-dir", default=DEFAULT_PROV,
                        help=f"Provider CSV directory (default: {DEFAULT_PROV})")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU, use pandas CPU")
    args = parser.parse_args()

    print("=" * 70)
    print("03b: Gold with Year — outputs to gold_year/")
    print("=" * 70)

    # Pre-check
    validate_silver_has_year(args.silver_dir)

    # Run existing pipeline with different output dir
    print(f"\nOutput: {args.gold_dir}/")
    gold_mod.engineer(
        silver_dir=args.silver_dir,
        gold_dir=args.gold_dir,
        provider_data_dir=args.provider_data_dir,
        force_cpu=args.no_gpu,
    )

    # Post-check
    validate_gold_has_year(args.gold_dir)

    print("\nDone! Upload gold_year/*.parquet to Drive (AllowanceMap/V2/gold_year/)")
    print("Then update V2_10 to use GOLD_DIR = f'{DRIVE_ROOT}/gold_year'")
