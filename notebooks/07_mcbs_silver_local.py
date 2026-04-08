"""
07_mcbs_silver_local.py — MCBS Silver Layer (Cleaning & Standardization)

Processes MCBS Bronze Parquets into analysis-ready Silver files.

IMPORTANT PUF CONSTRAINTS:
  - Survey File and Cost Supplement have DIFFERENT PUF_IDs — cannot be
    joined at the beneficiary level. They are independent samples.
  - No Census REGION column in the PUF (suppressed for privacy).
  - Cost Supplement has its own demographics (CSP_AGE, CSP_SEX, etc.).

Processing tracks:
  Survey (survey_{YEAR}.parquet):
    - Demographics: DEM_AGE, DEM_SEX, DEM_RACE
    - Coverage: ADM_DUAL_FLAG_YR, INS_* columns
    - Health: chronic conditions from survey questions

  Cost (cost_{YEAR}.parquet):  ** PRIMARY for Stage 2 OOP model **
    - Demographics: CSP_AGE, CSP_SEX, CSP_RACE, CSP_INCOME
    - Spending: PAMTOOP (OOP total), PAMTIP/OP/MP/HH/etc. (by service)
    - Health: CSP_NCHRNCND (chronic condition count)
    - Weights: CSPUFWGT (survey weight for population estimates)

Input:  local_pipeline/mcbs_bronze/survey_{YEAR}.parquet, cost_{YEAR}.parquet
Output: local_pipeline/mcbs_silver/{YEAR}.parquet (cost-based, Stage 2 ready)

Usage:
    python notebooks/07_mcbs_silver_local.py
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd

_PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BRONZE  = os.path.join(_PROJECT_ROOT, "local_pipeline", "mcbs_bronze")
DEFAULT_OUTPUT  = os.path.join(_PROJECT_ROOT, "local_pipeline", "mcbs_silver")

# ---------------------------------------------------------------------------
# Cost Supplement column mapping (actual CMS column names)
# These are consistent across 2018-2023 Cost Supplement PUF files.
# ---------------------------------------------------------------------------
COST_RENAME = {
    "PUF_ID":         "puf_id",
    "SURVEYYR":       "year",
    "CSP_AGE":        "age",
    "CSP_SEX":        "sex",
    "CSP_RACE":       "race",
    "CSP_INCOME":     "income",
    "CSP_NCHRNCND":   "chronic_count",
    # Expenditure by service type (all payers)
    "PAMTTOT":        "exp_total",
    "PAMTIP":         "exp_inpatient",
    "PAMTOP":         "exp_outpatient",
    "PAMTMP":         "exp_physician",
    "PAMTHH":         "exp_home_health",
    "PAMTDU":         "exp_dental",
    "PAMTVU":         "exp_vision",
    "PAMTHU":         "exp_hearing",
    "PAMTPM":         "exp_rx",
    # Payment sources
    "PAMTCARE":       "pay_medicare",
    "PAMTCAID":       "pay_medicaid",
    "PAMTMADV":       "pay_medicare_adv",
    "PAMTALPR":       "pay_private_ins",
    "PAMTOOP":        "pay_oop",          # ** OOP total — Stage 2 TARGET **
    "PAMTDISC":       "pay_discount",
    "PAMTOTH":        "pay_other",
    # Utilization (event counts by service)
    "PEVENTS":        "total_events",
    "IPAEVNTS":       "events_inpatient",
    "OPAEVNTS":       "events_outpatient",
    "MPAEVNTS":       "events_physician",
    "HHAEVNTS":       "events_home_health",
    "DUAEVNTS":       "events_dental",
    "VUAEVNTS":       "events_vision",
    "HUAEVNTS":       "events_hearing",
    "PMAEVNTS":       "events_rx",
    # Survey weight
    "CSPUFWGT":       "survey_weight",
}

# Numeric columns to cast after rename
COST_NUMERIC = [
    "age", "sex", "race", "income", "chronic_count",
    "exp_total", "exp_inpatient", "exp_outpatient", "exp_physician",
    "exp_home_health", "exp_dental", "exp_vision", "exp_hearing", "exp_rx",
    "pay_medicare", "pay_medicaid", "pay_medicare_adv", "pay_private_ins",
    "pay_oop", "pay_discount", "pay_other",
    "total_events", "events_inpatient", "events_outpatient", "events_physician",
    "events_home_health", "events_dental", "events_vision", "events_hearing",
    "events_rx", "survey_weight",
]

# Required columns — rows missing these are dropped
COST_REQUIRED = ["puf_id", "age", "pay_oop"]


def age_to_band(age: float) -> str:
    """Bin age into standard Medicare age bands."""
    if pd.isna(age):
        return "unknown"
    age = int(age)
    if age < 65:
        return "under_65"
    elif age < 70:
        return "65-69"
    elif age < 75:
        return "70-74"
    elif age < 80:
        return "75-79"
    elif age < 85:
        return "80-84"
    else:
        return "85+"


def compute_iqr_bounds(values: pd.Series) -> tuple[float, float]:
    """Compute 3x IQR bounds on non-zero values."""
    nonzero = values[values > 0].dropna()
    if len(nonzero) < 10:
        return 0.0, float("inf")
    q1 = nonzero.quantile(0.25)
    q3 = nonzero.quantile(0.75)
    iqr = q3 - q1
    return q1 - 3 * iqr, q3 + 3 * iqr


def process_cost_year(bronze_path: str, output_dir: str, year: int,
                      iqr_lower: float, iqr_upper: float):
    """Clean one year of Cost Supplement data."""
    out_path = os.path.join(output_dir, f"{year}.parquet")
    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  {year}: Already exists ({size_mb:.1f} MB) — skipping")
        return

    print(f"  {year}: ", end="")
    df = pd.read_parquet(bronze_path)
    print(f"{len(df):,} rows, {len(df.columns)} cols")

    # Rename to canonical names
    rename_map = {k: v for k, v in COST_RENAME.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    print(f"    Mapped {len(rename_map)} columns")

    # Drop replicate weight columns (CSPUF001-CSPUF100) and provenance
    drop_cols = [c for c in df.columns
                 if re.match(r"CSPUF\d{3}", c) or c.startswith("_")]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Type cast numeric columns
    for col in COST_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing required columns
    present_required = [c for c in COST_REQUIRED if c in df.columns]
    before = len(df)
    df = df.dropna(subset=present_required)
    dropped = before - len(df)
    if dropped:
        print(f"    Dropped {dropped:,} rows with missing required cols")

    # Derive: age_band
    if "age" in df.columns:
        df["age_band"] = df["age"].apply(age_to_band)

    # Derive: oop_share (what fraction of total expenditure is OOP)
    if "pay_oop" in df.columns and "exp_total" in df.columns:
        df["oop_share"] = np.where(
            df["exp_total"] > 0,
            df["pay_oop"] / df["exp_total"],
            0.0,
        )

    # Derive: has_medicaid (dual eligible proxy)
    if "pay_medicaid" in df.columns:
        df["has_medicaid"] = (df["pay_medicaid"] > 0).astype("int16")

    # Derive: has_private_ins
    if "pay_private_ins" in df.columns:
        df["has_private_ins"] = (df["pay_private_ins"] > 0).astype("int16")

    # Ensure year column is int16
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int16")
    else:
        df["year"] = np.int16(year)

    # IQR outlier removal on pay_oop
    if "pay_oop" in df.columns:
        before = len(df)
        df = df[(df["pay_oop"] >= iqr_lower) & (df["pay_oop"] <= iqr_upper)]
        removed = before - len(df)
        if removed:
            print(f"    IQR removed {removed:,} OOP outliers ({removed/max(before,1)*100:.1f}%)")

    # Write
    df.to_parquet(out_path, index=False)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"    -> {year}.parquet: {len(df):,} rows, {len(df.columns)} cols ({size_mb:.1f} MB)")


def clean_all(bronze_dir: str, output_dir: str):
    """Process all MCBS Cost Supplement Bronze files to Silver."""
    bronze_dir = os.path.abspath(bronze_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Discover cost supplement files (these are the primary Stage 2 input)
    cost_files = {}
    for f in sorted(glob.glob(os.path.join(bronze_dir, "cost_*.parquet"))):
        match = re.match(r"cost_(\d{4})\.parquet", os.path.basename(f))
        if match:
            cost_files[int(match.group(1))] = f

    if not cost_files:
        raise FileNotFoundError(
            f"No cost supplement Bronze parquets in '{bronze_dir}'\n"
            f"Run notebooks/06_mcbs_bronze_local.py first."
        )

    years = sorted(cost_files.keys())
    print(f"MCBS Silver Cleaning (Cost Supplement)")
    print(f"  Years: {years}")
    print(f"  Output: {output_dir}\n")

    # Compute global IQR bounds on pay_oop across all years
    print("Computing global IQR bounds on OOP spending...")
    oop_samples = []
    for year, fpath in sorted(cost_files.items()):
        try:
            cdf = pd.read_parquet(fpath)
            if "PAMTOOP" in cdf.columns:
                vals = pd.to_numeric(cdf["PAMTOOP"], errors="coerce").dropna()
                oop_samples.append(vals)
        except Exception as e:
            print(f"  [WARN] Skipping {year} for IQR: {e}")

    if oop_samples:
        all_oop = pd.concat(oop_samples, ignore_index=True)
        iqr_lower, iqr_upper = compute_iqr_bounds(all_oop)
        print(f"  IQR bounds (non-zero OOP): [{iqr_lower:.2f}, {iqr_upper:.2f}]")
        print(f"  Total OOP values sampled: {len(all_oop):,}\n")
    else:
        iqr_lower, iqr_upper = 0.0, float("inf")
        print("  No OOP data found — skipping IQR bounds\n")

    # Process each year
    for year in years:
        try:
            process_cost_year(cost_files[year], output_dir, year, iqr_lower, iqr_upper)
        except Exception as e:
            print(f"  {year}: ERROR — {e}")

    # Also note which survey files are available (for reference, not processed into Silver)
    survey_files = sorted(glob.glob(os.path.join(bronze_dir, "survey_*.parquet")))
    if survey_files:
        survey_years = [re.match(r"survey_(\d{4})", os.path.basename(f)).group(1)
                        for f in survey_files]
        print(f"\n  Note: Survey Bronze files available for {survey_years}")
        print(f"  (Survey data kept separate — different PUF_IDs from Cost Supplement)")

    print(f"\nSilver cleaning complete: {len(years)} cost year(s) processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCBS Silver layer: clean Cost Supplement data for Stage 2 OOP prediction."
    )
    parser.add_argument("--bronze-dir", default=DEFAULT_BRONZE,
                        help=f"MCBS Bronze directory (default: {DEFAULT_BRONZE})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT,
                        help=f"Output directory (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()
    clean_all(args.bronze_dir, args.output_dir)
