"""
08_mcbs_crosswalk_local.py — Provider × MCBS Crosswalk (Dual Mode)

Creates a bridge table linking provider-level Medicare allowed amounts
to MCBS beneficiary-level OOP patterns for Stage 2 modeling.

Two modes:
  --mode puf (default):
    Uses real MCBS PUF Cost Supplement data (no region, national level).
    Join on year only. Provider aggregation: (specialty, bucket, year).

  --mode synthetic:
    Uses synthetic LDS data from generate_synthetic_mcbs.py (has region).
    Join on (region, year). Provider aggregation: (region, specialty, bucket, year).
    Enables the full per-service regional OOP prediction pipeline.

Input:
  puf mode:       local_pipeline/gold/*.parquet + local_pipeline/mcbs_silver/*.parquet
  synthetic mode: local_pipeline/gold/*.parquet + local_pipeline/mcbs_synthetic/synthetic_oop.parquet

Output:
  - local_pipeline/mcbs_crosswalk/crosswalk.parquet

Usage:
    python notebooks/08_mcbs_crosswalk_local.py                  # puf mode (default)
    python notebooks/08_mcbs_crosswalk_local.py --mode synthetic  # synthetic/LDS mode
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd

_PROJECT_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GOLD      = os.path.join(_PROJECT_ROOT, "local_pipeline", "gold")
DEFAULT_MCBS      = os.path.join(_PROJECT_ROOT, "local_pipeline", "mcbs_silver")
DEFAULT_SYNTHETIC = os.path.join(_PROJECT_ROOT, "local_pipeline", "mcbs_synthetic", "synthetic_oop.parquet")
DEFAULT_OUTPUT    = os.path.join(_PROJECT_ROOT, "local_pipeline", "mcbs_crosswalk")


def load_label_encoders(gold_dir: str) -> dict:
    """Load label encoders for human-readable specialty names."""
    enc_path = os.path.join(gold_dir, "label_encoders.json")
    if not os.path.exists(enc_path):
        print(f"  [WARN] label_encoders.json not found at {enc_path}")
        return {}
    with open(enc_path) as f:
        return json.load(f)


def aggregate_provider_data(gold_dir: str) -> pd.DataFrame:
    """Aggregate provider gold data by (provider_type, hcpcs_bucket, year)."""
    gold_files = sorted(glob.glob(os.path.join(gold_dir, "*.parquet")))
    if not gold_files:
        raise FileNotFoundError(f"No gold parquets in {gold_dir}")

    load_cols = [
        "year", "Rndrng_Prvdr_Type_idx", "hcpcs_bucket",
        "Avg_Mdcr_Alowd_Amt", "Bene_Avg_Risk_Scre", "Avg_Sbmtd_Chrg",
    ]

    print(f"  Loading {len(gold_files)} gold state parquets...")
    chunks = []
    for i, f in enumerate(gold_files, 1):
        try:
            df = pd.read_parquet(f, columns=load_cols).dropna()
            chunks.append(df)
        except Exception as e:
            print(f"    [WARN] Skipping {os.path.basename(f)}: {e}")
        if i % 10 == 0:
            print(f"    {i}/{len(gold_files)} files loaded...")

    if not chunks:
        raise RuntimeError("No provider data loaded")

    combined = pd.concat(chunks, ignore_index=True)
    print(f"  Provider rows: {len(combined):,}")

    # Aggregate by (specialty, bucket, year) — national level
    group_cols = ["Rndrng_Prvdr_Type_idx", "hcpcs_bucket", "year"]
    agg = combined.groupby(group_cols).agg(
        mean_allowed_amt=("Avg_Mdcr_Alowd_Amt", "mean"),
        median_allowed_amt=("Avg_Mdcr_Alowd_Amt", "median"),
        std_allowed_amt=("Avg_Mdcr_Alowd_Amt", "std"),
        mean_risk_score=("Bene_Avg_Risk_Scre", "mean"),
        mean_submitted_chrg=("Avg_Sbmtd_Chrg", "mean"),
        provider_obs_count=("Avg_Mdcr_Alowd_Amt", "count"),
    ).reset_index()

    print(f"  Provider aggregated: {len(agg):,} (specialty x bucket x year) groups")
    return agg


def aggregate_mcbs_data(mcbs_dir: str) -> pd.DataFrame:
    """Aggregate MCBS Cost Supplement data by year (national level)."""
    mcbs_files = sorted(glob.glob(os.path.join(mcbs_dir, "*.parquet")))
    if not mcbs_files:
        raise FileNotFoundError(f"No MCBS silver parquets in {mcbs_dir}")

    print(f"  Loading {len(mcbs_files)} MCBS silver parquets...")
    chunks = []
    for f in mcbs_files:
        try:
            df = pd.read_parquet(f)
            chunks.append(df)
            print(f"    {os.path.basename(f)}: {len(df):,} benes")
        except Exception as e:
            print(f"    [WARN] Skipping {os.path.basename(f)}: {e}")

    if not chunks:
        raise RuntimeError("No MCBS data loaded")

    combined = pd.concat(chunks, ignore_index=True)
    print(f"  MCBS total benes: {len(combined):,}")

    # Aggregate by year (national level — no region available in PUF)
    agg_specs = {}
    if "pay_oop" in combined.columns:
        agg_specs["mcbs_mean_oop"]    = ("pay_oop", "mean")
        agg_specs["mcbs_median_oop"]  = ("pay_oop", "median")
        agg_specs["mcbs_std_oop"]     = ("pay_oop", "std")
        agg_specs["mcbs_p25_oop"]     = ("pay_oop", lambda x: x.quantile(0.25))
        agg_specs["mcbs_p75_oop"]     = ("pay_oop", lambda x: x.quantile(0.75))
    if "exp_total" in combined.columns:
        agg_specs["mcbs_mean_exp"]    = ("exp_total", "mean")
    if "oop_share" in combined.columns:
        agg_specs["mcbs_mean_oop_share"] = ("oop_share", "mean")
    if "age" in combined.columns:
        agg_specs["mcbs_mean_age"]    = ("age", "mean")
    if "chronic_count" in combined.columns:
        agg_specs["mcbs_mean_chronic"] = ("chronic_count", "mean")
    if "income" in combined.columns:
        agg_specs["mcbs_mean_income"] = ("income", "mean")
    if "has_medicaid" in combined.columns:
        agg_specs["mcbs_pct_medicaid"] = ("has_medicaid", "mean")
    if "has_private_ins" in combined.columns:
        agg_specs["mcbs_pct_private"]  = ("has_private_ins", "mean")

    agg_specs["mcbs_bene_count"] = ("puf_id" if "puf_id" in combined.columns else combined.columns[0], "count")

    agg = combined.groupby("year").agg(**agg_specs).reset_index()
    print(f"  MCBS aggregated: {len(agg)} year-level rows")
    for _, row in agg.iterrows():
        yr = int(row["year"])
        n = int(row.get("mcbs_bene_count", 0))
        oop = row.get("mcbs_mean_oop", float("nan"))
        print(f"    {yr}: {n:,} benes, mean OOP=${oop:,.0f}")

    return agg


def build_crosswalk_synthetic(gold_dir: str, synthetic_path: str, output_dir: str):
    """Build regional crosswalk from synthetic/LDS per-service OOP data."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "crosswalk.parquet")

    print("=== MCBS Crosswalk Builder (SYNTHETIC / LDS mode) ===\n")

    label_encoders = load_label_encoders(gold_dir)
    ptype_classes = label_encoders.get("Rndrng_Prvdr_Type", [])
    ptype_map = {float(i): name for i, name in enumerate(ptype_classes)}

    # Load synthetic data (already has census_region + per_service_oop)
    print("[1/3] Loading synthetic/LDS OOP data...")
    syn = pd.read_parquet(synthetic_path)
    print(f"  {len(syn):,} records, regions: {sorted(syn['census_region'].unique())}")

    # Aggregate provider side by (region, specialty, bucket, year)
    print("\n[2/3] Aggregating provider side (by region)...")
    provider_group = ["census_region", "Rndrng_Prvdr_Type_idx", "hcpcs_bucket", "year"]
    provider_agg = syn.groupby(provider_group).agg(
        mean_allowed_amt=("Avg_Mdcr_Alowd_Amt", "mean"),
        median_allowed_amt=("Avg_Mdcr_Alowd_Amt", "median"),
        std_allowed_amt=("Avg_Mdcr_Alowd_Amt", "std"),
        mean_risk_score=("Bene_Avg_Risk_Scre", "mean"),
        provider_obs_count=("Avg_Mdcr_Alowd_Amt", "count"),
    ).reset_index()
    print(f"  Provider groups: {len(provider_agg):,}")

    # Aggregate MCBS/OOP side by (region, year)
    print("\n[3/3] Aggregating OOP side (by region)...")
    mcbs_group = ["census_region", "year"]
    mcbs_agg = syn.groupby(mcbs_group).agg(
        mcbs_mean_oop=("per_service_oop", "mean"),
        mcbs_median_oop=("per_service_oop", "median"),
        mcbs_std_oop=("per_service_oop", "std"),
        mcbs_p25_oop=("per_service_oop", lambda x: x.quantile(0.25)),
        mcbs_p75_oop=("per_service_oop", lambda x: x.quantile(0.75)),
        mcbs_mean_age=("age", "mean"),
        mcbs_mean_chronic=("chronic_count", "mean"),
        mcbs_pct_dual=("dual_eligible", "mean"),
        mcbs_pct_supplemental=("has_supplemental", "mean"),
        mcbs_bene_count=("per_service_oop", "count"),
    ).reset_index()
    print(f"  MCBS region-year groups: {len(mcbs_agg)}")

    # Join on (region, year)
    crosswalk = provider_agg.merge(mcbs_agg, on=["census_region", "year"], how="left")

    REGION_INT_TO_NAME = {1: "NORTHEAST", 2: "MIDWEST", 3: "SOUTH", 4: "WEST"}
    crosswalk["census_region_name"] = crosswalk["census_region"].map(REGION_INT_TO_NAME)

    if ptype_map:
        crosswalk["specialty_name"] = crosswalk["Rndrng_Prvdr_Type_idx"].map(ptype_map)

    crosswalk["year"] = crosswalk["year"].astype("int16")
    crosswalk["_data_source"] = "synthetic"

    # Summary
    print(f"\n  Crosswalk: {len(crosswalk):,} rows")
    print(f"  Regions: {sorted(crosswalk['census_region'].unique())}")
    print(f"  Years: {sorted(crosswalk['year'].unique())}")
    print(f"  Specialties: {crosswalk['Rndrng_Prvdr_Type_idx'].nunique()}")

    crosswalk.to_parquet(out_path, index=False)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\n  -> crosswalk.parquet ({size_mb:.1f} MB)")
    print("\nDone.")


def build_crosswalk_puf(gold_dir: str, mcbs_dir: str, output_dir: str):
    """Build national crosswalk from real MCBS PUF data (no region)."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "crosswalk.parquet")

    print("=== MCBS Crosswalk Builder (PUF mode — national) ===\n")

    label_encoders = load_label_encoders(gold_dir)
    ptype_classes = label_encoders.get("Rndrng_Prvdr_Type", [])
    ptype_map = {float(i): name for i, name in enumerate(ptype_classes)}

    # Aggregate provider data
    print("[1/3] Aggregating provider data (national by specialty x bucket x year)...")
    provider_agg = aggregate_provider_data(gold_dir)

    # Aggregate MCBS data
    print("\n[2/3] Aggregating MCBS Cost Supplement (national by year)...")
    mcbs_agg = aggregate_mcbs_data(mcbs_dir)

    # Join on year
    print("\n[3/3] Joining provider x MCBS on year...")
    crosswalk = provider_agg.merge(mcbs_agg, on="year", how="left")

    if ptype_map:
        crosswalk["specialty_name"] = crosswalk["Rndrng_Prvdr_Type_idx"].map(ptype_map)

    crosswalk["year"] = crosswalk["year"].astype("int16")
    crosswalk["_data_source"] = "puf"

    # Summary
    n_with_mcbs = crosswalk["mcbs_mean_oop"].notna().sum() if "mcbs_mean_oop" in crosswalk.columns else 0
    print(f"\n  Crosswalk: {len(crosswalk):,} rows")
    print(f"  Years: {sorted(crosswalk['year'].unique())}")
    print(f"  Specialties: {crosswalk['Rndrng_Prvdr_Type_idx'].nunique()}")
    print(f"  Rows with MCBS OOP data: {n_with_mcbs:,} / {len(crosswalk):,}")

    mcbs_years = set(mcbs_agg["year"].unique()) if len(mcbs_agg) else set()
    provider_years = set(provider_agg["year"].unique())
    missing = provider_years - mcbs_years
    if missing:
        print(f"  Provider years without MCBS match: {sorted(missing)}")

    crosswalk.to_parquet(out_path, index=False)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\n  -> crosswalk.parquet ({size_mb:.1f} MB)")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Provider x MCBS crosswalk (dual mode: puf or synthetic)."
    )
    parser.add_argument("--mode", choices=["puf", "synthetic"], default="puf",
                        help="Data source: 'puf' (real, national) or 'synthetic' (with region)")
    parser.add_argument("--gold-dir", default=DEFAULT_GOLD,
                        help=f"Provider gold directory (default: {DEFAULT_GOLD})")
    parser.add_argument("--mcbs-dir", default=DEFAULT_MCBS,
                        help=f"MCBS silver directory — puf mode only (default: {DEFAULT_MCBS})")
    parser.add_argument("--synthetic-path", default=DEFAULT_SYNTHETIC,
                        help=f"Synthetic OOP parquet — synthetic mode only (default: {DEFAULT_SYNTHETIC})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT,
                        help=f"Output directory (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    if args.mode == "synthetic":
        build_crosswalk_synthetic(args.gold_dir, args.synthetic_path, args.output_dir)
    else:
        build_crosswalk_puf(args.gold_dir, args.mcbs_dir, args.output_dir)
