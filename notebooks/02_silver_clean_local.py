"""
02_silver_clean_local.py — Local Silver Cleaning (batch / memory-safe)
Reads directly from partitioned_data/{STATE}/*.parquet (skips bronze consolidation).
Processes one state at a time to stay within memory, writing per-state silver parquets
to local_pipeline/silver/{STATE}.parquet.

IQR outlier bounds are computed globally in a cheap first pass that loads only the
target column, so the filter is consistent across states.

Usage:
    python notebooks/02_silver_clean_local.py
    python notebooks/02_silver_clean_local.py --input-dir ../partitioned_data --output-dir local_pipeline/silver
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np

_PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT  = os.path.join(_PROJECT_ROOT, "..", "partitioned_data")
DEFAULT_OUTPUT = os.path.join(_PROJECT_ROOT, "local_pipeline", "silver")

NUMERIC_COLS  = [
    "Tot_Benes", "Tot_Srvcs", "Tot_Bene_Day_Srvcs",
    "Avg_Sbmtd_Chrg", "Avg_Mdcr_Alowd_Amt", "Avg_Mdcr_Pymt_Amt",
    "Avg_Mdcr_Stdzd_Amt",
]
REQUIRED_COLS = ["Rndrng_NPI", "HCPCS_Cd", "Avg_Mdcr_Alowd_Amt"]
TARGET        = "Avg_Mdcr_Alowd_Amt"


# ── Pass 1: compute global IQR bounds ────────────────────────────────────────

def compute_global_iqr_bounds(state_dirs: list[str]) -> tuple[float, float]:
    """
    Load only the TARGET column from every partition file and compute
    global Q1/Q3. Cheap — one column per file, no full load.
    """
    print("Pass 1/2: Computing global IQR bounds on target column...")
    series_parts = []
    for state_dir in state_dirs:
        for f in glob.glob(os.path.join(state_dir, "*.parquet")):
            try:
                col = pd.read_parquet(f, columns=[TARGET])[TARGET]
                col = pd.to_numeric(col, errors="coerce").dropna()
                series_parts.append(col)
            except Exception as e:
                print(f"  [WARN] Could not read {f}: {e}")

    all_values = pd.concat(series_parts, ignore_index=True)
    q1 = all_values.quantile(0.25)
    q3 = all_values.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
    print(f"  Global IQR bounds: lower={lower:.2f}, upper={upper:.2f}  "
          f"(Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f})")
    return lower, upper


# ── Per-file cleaning ─────────────────────────────────────────────────────────

def clean_df(df: pd.DataFrame, iqr_lower: float, iqr_upper: float) -> pd.DataFrame:
    # Cast numerics
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing required columns
    present_required = [c for c in REQUIRED_COLS if c in df.columns]
    df = df.dropna(subset=present_required)

    # Apply global IQR filter
    df = df[df[TARGET].between(iqr_lower, iqr_upper)]

    # Normalize provider type casing
    if "Rndrng_Prvdr_Type" in df.columns:
        df["Rndrng_Prvdr_Type"] = df["Rndrng_Prvdr_Type"].str.strip().str.title()

    return df


# ── Pass 2: process state by state ───────────────────────────────────────────

def process_state(state_dir: str, output_dir: str, iqr_lower: float, iqr_upper: float) -> int:
    """Clean all provider parquets for one state, write a single silver parquet."""
    state = os.path.basename(state_dir)
    files = glob.glob(os.path.join(state_dir, "*.parquet"))
    if not files:
        return 0

    chunks = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            df["_src_state"]    = state
            df["_src_provider"] = os.path.splitext(os.path.basename(f))[0]
            chunks.append(clean_df(df, iqr_lower, iqr_upper))
        except Exception as e:
            print(f"  [WARN] Skipping {f}: {e}")

    if not chunks:
        return 0

    silver_df = pd.concat(chunks, ignore_index=True)
    # 'year' column (from partition_medicare_data.py) passes through as-is — needed for LSTM
    if "year" not in silver_df.columns:
        print(f"  [WARN] {state}: missing 'year' column — re-run partition_medicare_data.py")
    out_path  = os.path.join(output_dir, f"{state}.parquet")
    silver_df.to_parquet(out_path, index=False)
    return len(silver_df)


# ── Entry point ───────────────────────────────────────────────────────────────

def clean(input_dir: str, output_dir: str):
    input_dir = os.path.abspath(input_dir)
    state_dirs = sorted(
        d for d in glob.glob(os.path.join(input_dir, "*"))
        if os.path.isdir(d)
    )
    if not state_dirs:
        raise FileNotFoundError(
            f"No state subdirectories found under '{input_dir}'.\n"
            f"  Expected: {input_dir}/{{STATE}}/{{PROVIDER}}.parquet"
        )
    print(f"Found {len(state_dirs)} state directories under '{input_dir}'")

    iqr_lower, iqr_upper = compute_global_iqr_bounds(state_dirs)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nPass 2/2: Cleaning and writing per-state silver parquets → {output_dir}/")
    total_rows = 0
    for i, state_dir in enumerate(state_dirs, 1):
        state     = os.path.basename(state_dir)
        n         = process_state(state_dir, output_dir, iqr_lower, iqr_upper)
        total_rows += n
        print(f"  [{i:>2}/{len(state_dirs)}] {state:<6}  {n:>8,} rows")

    print(f"\nSilver layer complete — {total_rows:,} total rows across {len(state_dirs)} states")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean partitioned_data into Silver layer, one state at a time."
    )
    parser.add_argument(
        "--input-dir", default=DEFAULT_INPUT,
        help=f"Root of partitioned_data/ tree (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT,
        help=f"Directory to write per-state silver parquets (default: {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()
    clean(args.input_dir, args.output_dir)
