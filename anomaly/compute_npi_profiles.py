"""
compute_npi_profiles.py — Phase A, spec §3.1

Aggregate silver parquets (row grain: NPI × HCPCS × Place_Of_Srvc × year)
into NPI × year profiles for downstream anomaly detection.

Input:  local_pipeline/silver/{STATE}.parquet
        data/provider_summary_*.csv  (for Bene_Avg_Risk_Scre)
Output: local_pipeline/anomaly/npi_profiles.parquet

Usage:
    python anomaly/compute_npi_profiles.py
    python anomaly/compute_npi_profiles.py --states CA,TX,FL
    python anomaly/compute_npi_profiles.py --sample 0.1
"""

import os
import re
import glob
import argparse
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SILVER  = os.path.join(_PROJECT_ROOT, "local_pipeline", "silver")
DEFAULT_OUT_DIR = os.path.join(_PROJECT_ROOT, "local_pipeline", "anomaly")
DEFAULT_OUT     = os.path.join(DEFAULT_OUT_DIR, "npi_profiles.parquet")
DEFAULT_PROV    = os.path.join(_PROJECT_ROOT, "data")

SILVER_COLS = [
    "Rndrng_NPI",
    "Rndrng_Prvdr_Type",
    "Rndrng_Prvdr_State_Abrvtn",
    "HCPCS_Cd",
    "Place_Of_Srvc",
    "Tot_Benes",
    "Tot_Srvcs",
    "Avg_Sbmtd_Chrg",
    "Avg_Mdcr_Alowd_Amt",
    "year",
]


def hcpcs_to_bucket(hcpcs_series: pd.Series) -> pd.Series:
    """Mirror of notebooks/03_gold_features_local.py hcpcs_to_bucket."""
    s = hcpcs_series.astype(str).str.strip()
    is_alpha = s.str[0].str.isalpha().fillna(False)
    numeric = pd.to_numeric(s, errors="coerce")
    buckets = pd.Series(np.nan, index=s.index, dtype="float64")
    buckets[is_alpha] = 5.0
    buckets[(numeric >= 100)   & (numeric <= 1999)]  = 0.0
    buckets[(numeric >= 10000) & (numeric <= 69999)] = 1.0
    buckets[(numeric >= 70000) & (numeric <= 79999)] = 2.0
    buckets[(numeric >= 80000) & (numeric <= 89999)] = 3.0
    buckets[(numeric >= 90000) & (numeric <= 99999)] = 4.0
    return buckets


def load_provider_risk_scores(provider_data_dir: str) -> pd.DataFrame:
    """Read CMS 'by Provider' CSVs → (Rndrng_NPI, year, Bene_Avg_Risk_Scre)."""
    provider_data_dir = os.path.abspath(provider_data_dir)
    csv_files = sorted(glob.glob(os.path.join(provider_data_dir, "provider_summary_*.csv")))
    if not csv_files:
        print(f"  [WARN] No provider_summary_*.csv in '{provider_data_dir}' -- risk_score will be NaN")
        return pd.DataFrame(columns=["Rndrng_NPI", "year", "Bene_Avg_Risk_Scre"])

    parts = []
    for f in csv_files:
        year_match = re.search(r"(\d{4})", os.path.basename(f))
        if not year_match:
            continue
        year = int(year_match.group(1))
        try:
            df = pd.read_csv(f, usecols=["Rndrng_NPI", "Bene_Avg_Risk_Scre"], dtype=str)
            df["year"] = year
            df["Bene_Avg_Risk_Scre"] = pd.to_numeric(df["Bene_Avg_Risk_Scre"], errors="coerce")
            df["Rndrng_NPI"] = df["Rndrng_NPI"].astype(str).str.strip()
            df = df.dropna(subset=["Bene_Avg_Risk_Scre"])
            parts.append(df)
            print(f"  risk scores {year}: {len(df):,} NPIs")
        except Exception as e:
            print(f"  [WARN] {f}: {e}")

    if not parts:
        return pd.DataFrame(columns=["Rndrng_NPI", "year", "Bene_Avg_Risk_Scre"])
    return pd.concat(parts, ignore_index=True)


def aggregate_state(df: pd.DataFrame) -> pd.DataFrame:
    """Produce per-(NPI, year) profile rows from one state's silver dataframe."""
    df = df.dropna(subset=["Rndrng_NPI", "HCPCS_Cd", "Tot_Srvcs"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int16")
    df = df.dropna(subset=["year"])
    df["Rndrng_NPI"] = df["Rndrng_NPI"].astype(str).str.strip()
    df["HCPCS_Cd"]   = df["HCPCS_Cd"].astype(str).str.strip()

    # Derived row-level fields
    df["hcpcs_bucket"]   = hcpcs_to_bucket(df["HCPCS_Cd"]).fillna(5.0)
    df["row_billing"]    = df["Tot_Srvcs"] * df["Avg_Sbmtd_Chrg"]
    df["row_allowed"]    = df["Tot_Srvcs"] * df["Avg_Mdcr_Alowd_Amt"]
    df["is_facility"]    = (df["Place_Of_Srvc"].astype(str) == "F").astype(np.int8)
    df["facility_srvcs"] = df["is_facility"] * df["Tot_Srvcs"]

    group_keys = ["Rndrng_NPI", "year"]

    # Main aggregates
    main = df.groupby(group_keys, observed=True, sort=False).agg(
        total_services      =("Tot_Srvcs", "sum"),
        total_beneficiaries =("Tot_Benes", "sum"),
        total_billing       =("row_billing", "sum"),
        total_allowed       =("row_allowed", "sum"),
        avg_charge          =("Avg_Sbmtd_Chrg", "mean"),
        avg_allowed         =("Avg_Mdcr_Alowd_Amt", "mean"),
        n_unique_hcpcs      =("HCPCS_Cd", "nunique"),
        facility_srvcs      =("facility_srvcs", "sum"),
        specialty           =("Rndrng_Prvdr_Type", "first"),
        state               =("Rndrng_Prvdr_State_Abrvtn", "first"),
    ).reset_index()

    main["srvcs_per_bene"]         = main["total_services"] / main["total_beneficiaries"].replace(0, np.nan)
    main["charge_to_allowed_ratio"] = main["avg_charge"]    / main["avg_allowed"].replace(0, np.nan)
    main["facility_pct"]           = main["facility_srvcs"] / main["total_services"].replace(0, np.nan)
    main = main.drop(columns=["facility_srvcs"])

    # Herfindahl index: first collapse to (NPI, year, HCPCS), then shares
    hcpcs_srvcs = df.groupby(group_keys + ["HCPCS_Cd"], observed=True, sort=False)["Tot_Srvcs"].sum().reset_index()
    hcpcs_srvcs["total_by_group"] = hcpcs_srvcs.groupby(group_keys, observed=True)["Tot_Srvcs"].transform("sum")
    hcpcs_srvcs["share_sq"] = (hcpcs_srvcs["Tot_Srvcs"] / hcpcs_srvcs["total_by_group"].replace(0, np.nan)) ** 2
    hhi = hcpcs_srvcs.groupby(group_keys, observed=True)["share_sq"].sum().rename("herfindahl_index").reset_index()
    main = main.merge(hhi, on=group_keys, how="left")

    # Bucket distribution: 6 columns with service-share per bucket
    bucket_srvcs = df.groupby(group_keys + ["hcpcs_bucket"], observed=True, sort=False)["Tot_Srvcs"].sum().unstack(fill_value=0.0)
    # Ensure all 6 bucket columns present
    for b in range(6):
        if float(b) not in bucket_srvcs.columns:
            bucket_srvcs[float(b)] = 0.0
    bucket_srvcs = bucket_srvcs[[float(b) for b in range(6)]]
    total_by_row = bucket_srvcs.sum(axis=1).replace(0, np.nan)
    bucket_pct = bucket_srvcs.div(total_by_row, axis=0).fillna(0.0)
    bucket_pct.columns = [f"bucket_{int(c)}_pct" for c in bucket_pct.columns]
    bucket_pct = bucket_pct.reset_index()
    main = main.merge(bucket_pct, on=group_keys, how="left")

    return main


def add_yoy_changes(profiles: pd.DataFrame) -> pd.DataFrame:
    """Compute year-over-year change metrics within each NPI."""
    profiles = profiles.sort_values(["Rndrng_NPI", "year"]).reset_index(drop=True)
    g = profiles.groupby("Rndrng_NPI", sort=False)

    prev_srvcs   = g["total_services"].shift(1)
    prev_billing = g["total_billing"].shift(1)
    prev_benes   = g["total_beneficiaries"].shift(1)

    profiles["yoy_volume_change"]  = (profiles["total_services"] - prev_srvcs) / prev_srvcs.replace(0, np.nan)
    profiles["yoy_billing_change"] = (profiles["total_billing"]  - prev_billing) / prev_billing.replace(0, np.nan)
    profiles["yoy_bene_change"]    = (profiles["total_beneficiaries"] - prev_benes) / prev_benes.replace(0, np.nan)
    return profiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver-dir",   default=DEFAULT_SILVER)
    ap.add_argument("--provider-dir", default=DEFAULT_PROV)
    ap.add_argument("--output",       default=DEFAULT_OUT)
    ap.add_argument("--states",       default="", help="Comma-separated state abbrevs to include (default: all)")
    ap.add_argument("--sample",       type=float, default=1.0, help="Row-level sample fraction per state for dev")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(args.silver_dir, "*.parquet")))
    if args.states:
        wanted = {s.strip().upper() for s in args.states.split(",") if s.strip()}
        all_files = [f for f in all_files if os.path.splitext(os.path.basename(f))[0] in wanted]
    if not all_files:
        raise SystemExit(f"No silver parquet files found in {args.silver_dir}")

    print(f"Silver files: {len(all_files)}  | sample={args.sample}")
    print(f"Output:       {args.output}")

    # Risk scores loaded once
    risk_df = load_provider_risk_scores(args.provider_dir)

    per_state_profiles = []
    t0 = time.time()
    for i, f in enumerate(all_files, 1):
        state = os.path.splitext(os.path.basename(f))[0]
        try:
            schema_cols = pq.read_schema(f).names
            use_cols = [c for c in SILVER_COLS if c in schema_cols]
            df = pd.read_parquet(f, columns=use_cols)
            if args.sample < 1.0:
                df = df.sample(frac=args.sample, random_state=42)
            prof = aggregate_state(df)
            per_state_profiles.append(prof)
            print(f"  [{i:2d}/{len(all_files)}] {state}: {len(df):>9,} rows -> {len(prof):>7,} NPI-years")
        except Exception as e:
            print(f"  [WARN] {state}: {e}")
            continue

    profiles = pd.concat(per_state_profiles, ignore_index=True)
    print(f"Concatenated profiles: {len(profiles):,} NPI-year rows")

    # Year-over-year changes
    profiles = add_yoy_changes(profiles)

    # Join risk scores
    if not risk_df.empty:
        risk_df["year"] = risk_df["year"].astype("Int16")
        profiles = profiles.merge(
            risk_df.rename(columns={"Bene_Avg_Risk_Scre": "risk_score"}),
            on=["Rndrng_NPI", "year"],
            how="left",
        )
        pct_matched = profiles["risk_score"].notna().mean() * 100
        print(f"Risk score coverage: {pct_matched:.1f}% of NPI-years")
    else:
        profiles["risk_score"] = np.nan

    # Cast dtypes for storage efficiency
    profiles["year"] = profiles["year"].astype("int16")
    for col in [
        "total_services", "total_beneficiaries", "total_billing", "total_allowed",
        "avg_charge", "avg_allowed", "srvcs_per_bene", "charge_to_allowed_ratio",
        "facility_pct", "herfindahl_index",
        "bucket_0_pct", "bucket_1_pct", "bucket_2_pct", "bucket_3_pct",
        "bucket_4_pct", "bucket_5_pct",
        "yoy_volume_change", "yoy_billing_change", "yoy_bene_change",
        "risk_score",
    ]:
        if col in profiles.columns:
            profiles[col] = profiles[col].astype("float32")
    profiles["n_unique_hcpcs"] = profiles["n_unique_hcpcs"].astype("int32")

    profiles.to_parquet(args.output, index=False, compression="snappy")
    print(f"Wrote {len(profiles):,} rows -> {args.output}  ({time.time()-t0:.1f}s)")

    # Quick summary
    print("\n-- Summary --")
    print(f"  Unique NPIs:       {profiles['Rndrng_NPI'].nunique():,}")
    print(f"  Years:             {sorted(profiles['year'].unique().tolist())}")
    print(f"  Unique specialties:{profiles['specialty'].nunique():,}")
    print(f"  Unique states:     {profiles['state'].nunique():,}")


if __name__ == "__main__":
    main()
