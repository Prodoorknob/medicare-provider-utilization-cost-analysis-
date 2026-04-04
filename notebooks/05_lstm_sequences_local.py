"""
05_lstm_sequences_local.py — LSTM Sequence Preparation

Reads per-state gold parquets and creates LSTM-ready time-series sequences.
Groups data by (provider_type × hcpcs_bucket × state) and for each group
produces a year-ordered vector of mean allowed amounts.

Output: local_pipeline/lstm/sequences.parquet
  - Group key columns (float64)
  - years: list of years present (sorted)
  - target_seq: list of mean Avg_Mdcr_Alowd_Amt per year (aligned with years)
  - n_years: number of years with data (int)

This is a PREPARATION step — the LSTM model training (Phase 3) consumes
this output. Groups with fewer years produce shorter sequences; filtering
by minimum sequence length is deferred to the LSTM training script.

Usage:
    python notebooks/05_lstm_sequences_local.py
    python notebooks/05_lstm_sequences_local.py --gold-dir local_pipeline/gold
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GOLD    = os.path.join(_PROJECT_ROOT, "local_pipeline", "gold")
DEFAULT_OUTPUT  = os.path.join(_PROJECT_ROOT, "local_pipeline", "lstm")

GROUP_KEYS = [
    "Rndrng_Prvdr_Type_idx",
    "hcpcs_bucket",
    "Rndrng_Prvdr_State_Abrvtn_idx",
]
TARGET = "Avg_Mdcr_Alowd_Amt"
LOAD_COLS = GROUP_KEYS + ["year", TARGET]


def build_sequences(gold_dir: str, output_dir: str):
    gold_dir = os.path.abspath(gold_dir)
    parquet_files = sorted(glob.glob(os.path.join(gold_dir, "*.parquet")))

    if not parquet_files:
        raise FileNotFoundError(f"No gold parquets found in '{gold_dir}'")

    print(f"Loading {len(parquet_files)} gold state parquets...")

    # Accumulate group-year means across all states
    agg_parts = []
    total_rows = 0

    for i, f in enumerate(parquet_files, 1):
        state = os.path.splitext(os.path.basename(f))[0]
        try:
            df = pd.read_parquet(f, columns=LOAD_COLS).dropna()
            if df.empty:
                continue

            # Compute mean target per group per year within this state
            grouped = (
                df.groupby(GROUP_KEYS + ["year"])[TARGET]
                .mean()
                .reset_index()
            )
            agg_parts.append(grouped)
            total_rows += len(df)

            if i % 10 == 0:
                print(f"  Processed {i}/{len(parquet_files)} states...")
        except Exception as e:
            print(f"  [WARN] Skipping {state}: {e}")

    if not agg_parts:
        raise RuntimeError("No data collected from gold parquets.")

    print(f"Aggregating across {len(agg_parts)} states ({total_rows:,} source rows)...")

    # Combine all states and re-aggregate (same group may span multiple state files)
    combined = pd.concat(agg_parts, ignore_index=True)
    year_means = (
        combined.groupby(GROUP_KEYS + ["year"])[TARGET]
        .mean()
        .reset_index()
    )

    print(f"Unique group-year combinations: {len(year_means):,}")

    # Pivot into sequences: one row per group, with year-ordered target lists
    def make_sequence(grp):
        grp_sorted = grp.sort_values("year")
        return pd.Series({
            "years":      grp_sorted["year"].astype(int).tolist(),
            "target_seq": grp_sorted[TARGET].tolist(),
            "n_years":    len(grp_sorted),
        })

    print("Building year-ordered sequences per group...")
    sequences = year_means.groupby(GROUP_KEYS).apply(make_sequence).reset_index()

    # Summary stats
    n_groups = len(sequences)
    year_counts = sequences["n_years"]
    print(f"\nSequence stats:")
    print(f"  Total groups: {n_groups:,}")
    print(f"  Avg years/group: {year_counts.mean():.1f}")
    print(f"  Groups with all 11 years: {(year_counts == 11).sum():,}")
    print(f"  Groups with < 3 years: {(year_counts < 3).sum():,}")

    # Write output
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "sequences.parquet")
    sequences.to_parquet(out_path, index=False)
    print(f"\nLSTM sequences written → {out_path} ({n_groups:,} groups)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare LSTM-ready time-series sequences from gold parquets."
    )
    parser.add_argument("--gold-dir", default=DEFAULT_GOLD,
                        help=f"Gold directory with per-state parquets (default: {DEFAULT_GOLD})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT,
                        help=f"Output directory for sequences (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()
    build_sequences(args.gold_dir, args.output_dir)
