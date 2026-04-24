"""
compute_benchmarks.py -- Phase A, spec section 3.2

Aggregate npi_profiles.parquet into three benchmark tables for
percentile/z-score comparisons in Phase B outlier detection.

Input:  local_pipeline/anomaly/npi_profiles.parquet
Output:
    local_pipeline/anomaly/specialty_benchmarks.parquet       (specialty x year)
    local_pipeline/anomaly/state_specialty_benchmarks.parquet (specialty x state x year)
    local_pipeline/anomaly/national_benchmarks.parquet        (year)

Usage:
    python anomaly/compute_benchmarks.py
    python anomaly/compute_benchmarks.py --profiles local_pipeline/anomaly/npi_profiles_VT.parquet
"""

import os
import argparse
import numpy as np
import pandas as pd

_PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR  = os.path.join(_PROJECT_ROOT, "local_pipeline", "anomaly")
DEFAULT_PROFILES = os.path.join(DEFAULT_OUT_DIR, "npi_profiles.parquet")

METRICS = [
    "total_services",
    "total_beneficiaries",
    "total_billing",
    "total_allowed",
    "srvcs_per_bene",
    "avg_charge",
    "avg_allowed",
    "charge_to_allowed_ratio",
    "n_unique_hcpcs",
    "herfindahl_index",
    "facility_pct",
    "yoy_volume_change",
    "yoy_billing_change",
    "risk_score",
]

PERCENTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


def compute_stats(group: pd.DataFrame, metrics: list[str]) -> pd.Series:
    """Compute mean + P5/P25/P50/P75/P95 for each metric within a group."""
    out = {}
    out["n_providers"] = len(group)
    for m in metrics:
        if m not in group.columns:
            continue
        s = group[m].dropna()
        if s.empty:
            for stat in ["mean", "p5", "p25", "p50", "p75", "p95"]:
                out[f"{m}_{stat}"] = np.nan
            continue
        out[f"{m}_mean"] = s.mean()
        q = s.quantile(PERCENTILES).values
        out[f"{m}_p5"]  = q[0]
        out[f"{m}_p25"] = q[1]
        out[f"{m}_p50"] = q[2]
        out[f"{m}_p75"] = q[3]
        out[f"{m}_p95"] = q[4]
    return pd.Series(out)


def aggregate_by(profiles: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    """Group profiles by keys and compute benchmark stats per group."""
    grouped = profiles.groupby(keys, observed=True, sort=False)
    benchmarks = grouped.apply(lambda g: compute_stats(g, metrics), include_groups=False).reset_index()
    return benchmarks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default=DEFAULT_PROFILES)
    ap.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading profiles: {args.profiles}")
    profiles = pd.read_parquet(args.profiles)
    print(f"  {len(profiles):,} NPI-year rows, "
          f"{profiles['specialty'].nunique()} specialties, "
          f"{profiles['state'].nunique()} states, "
          f"{profiles['year'].nunique()} years")

    available_metrics = [m for m in METRICS if m in profiles.columns]
    print(f"  metrics: {available_metrics}")

    # 1. Specialty x year
    print("\n[1/3] Specialty x year benchmarks...")
    sp_bench = aggregate_by(profiles, ["specialty", "year"], available_metrics)
    sp_path = os.path.join(args.output_dir, "specialty_benchmarks.parquet")
    sp_bench.to_parquet(sp_path, index=False, compression="snappy")
    print(f"  {len(sp_bench):,} rows -> {sp_path}")

    # 2. Specialty x state x year
    print("\n[2/3] Specialty x state x year benchmarks...")
    ss_bench = aggregate_by(profiles, ["specialty", "state", "year"], available_metrics)
    ss_path = os.path.join(args.output_dir, "state_specialty_benchmarks.parquet")
    ss_bench.to_parquet(ss_path, index=False, compression="snappy")
    print(f"  {len(ss_bench):,} rows -> {ss_path}")

    # 3. National x year
    print("\n[3/3] National x year benchmarks...")
    nat_bench = aggregate_by(profiles, ["year"], available_metrics)
    nat_path = os.path.join(args.output_dir, "national_benchmarks.parquet")
    nat_bench.to_parquet(nat_path, index=False, compression="snappy")
    print(f"  {len(nat_bench):,} rows -> {nat_path}")

    print("\n-- Summary --")
    print(f"  Specialty benchmarks:       {len(sp_bench):,} groups (expected ~{131*11} max)")
    print(f"  State-specialty benchmarks: {len(ss_bench):,} groups")
    print(f"  Thin groups (<10 providers): "
          f"{(ss_bench['n_providers'] < 10).sum():,} of {len(ss_bench):,} "
          f"({(ss_bench['n_providers'] < 10).mean()*100:.1f}%)")
    print(f"  National benchmarks:        {len(nat_bench):,} years")


if __name__ == "__main__":
    main()
