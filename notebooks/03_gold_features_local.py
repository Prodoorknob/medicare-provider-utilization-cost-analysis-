"""
03_gold_features_local.py — Local Gold Feature Engineering (GPU-accelerated, memory-safe)
Reads per-state silver parquets from local_pipeline/silver/{STATE}.parquet.

Outputs per-state gold parquets to local_pipeline/gold/{STATE}.parquet
plus a label_encoders.json for consistent encoding across training scripts.

Features aligned with project spec (Phase 1-2):
  - Target: Avg_Mdcr_Alowd_Amt (Medicare allowed amount)
  - New: hcpcs_bucket (CPT range clinical category)
  - New: place_of_srvc_flag (facility=1, office=0)
  - New: Bene_Avg_Risk_Scre (HCC risk from "by Provider" dataset, NPI join)
  - New: log_srvcs, log_benes (log-transformed counts)
  - Kept: Avg_Sbmtd_Chrg, srvcs_per_bene, year, encoded categoricals

Uses NVIDIA cuDF (RAPIDS) for GPU-accelerated dataframe ops when available,
falling back to pandas automatically if cuDF is not installed.

Usage:
    python notebooks/03_gold_features_local.py
    python notebooks/03_gold_features_local.py --provider-data-dir data/
    python notebooks/03_gold_features_local.py --no-gpu
"""

import os
import re
import json
import glob
import argparse
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import LabelEncoder

_PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SILVER = os.path.join(_PROJECT_ROOT, "local_pipeline", "silver")
DEFAULT_GOLD   = os.path.join(_PROJECT_ROOT, "local_pipeline", "gold")
DEFAULT_PROV   = os.path.join(_PROJECT_ROOT, "..", "data")

CAT_COLS = ["Rndrng_Prvdr_Type", "Rndrng_Prvdr_State_Abrvtn", "HCPCS_Cd"]

FEATURE_COLS = [
    "year",                            # temporal metadata (not a model feature)
    "Rndrng_Prvdr_Type_idx",           # encoded categoricals
    "Rndrng_Prvdr_State_Abrvtn_idx",
    "HCPCS_Cd_idx",
    "hcpcs_bucket",                    # coarse clinical category (0-5)
    "place_of_srvc_flag",              # binary: facility=1, office=0
    "Bene_Avg_Risk_Scre",              # NPI-level HCC risk score
    "log_srvcs",                       # log1p(Tot_Srvcs)
    "log_benes",                       # log1p(Tot_Benes)
    "Avg_Sbmtd_Chrg",                  # submitted charge
    "srvcs_per_bene",                  # Tot_Srvcs / Tot_Benes ratio
    "Avg_Mdcr_Alowd_Amt",             # TARGET
]

GOLD_SCHEMA = pa.schema([
    pa.field("year",                           pa.int16()),
    pa.field("Rndrng_Prvdr_Type_idx",          pa.float64()),
    pa.field("Rndrng_Prvdr_State_Abrvtn_idx",  pa.float64()),
    pa.field("HCPCS_Cd_idx",                   pa.float64()),
    pa.field("hcpcs_bucket",                   pa.float64()),
    pa.field("place_of_srvc_flag",             pa.float64()),
    pa.field("Bene_Avg_Risk_Scre",             pa.float64()),
    pa.field("log_srvcs",                      pa.float64()),
    pa.field("log_benes",                      pa.float64()),
    pa.field("Avg_Sbmtd_Chrg",                pa.float64()),
    pa.field("srvcs_per_bene",                 pa.float64()),
    pa.field("Avg_Mdcr_Alowd_Amt",            pa.float64()),
])


# ── GPU / CPU backend selection ───────────────────────────────────────────────

def _load_backend(force_cpu: bool):
    if not force_cpu:
        try:
            import cudf
            print("cuDF available — using GPU")
            return cudf, True
        except ImportError:
            print("cuDF not found — falling back to pandas (CPU)")
    else:
        print("--no-gpu set — using pandas (CPU)")
    return pd, False


# ── HCPCS bucketing ──────────────────────────────────────────────────────────

def hcpcs_to_bucket(hcpcs_series) -> pd.Series:
    """
    Map HCPCS codes to 6 coarse clinical categories by CPT range.
    Returns float64 Series (NaN for unmappable codes, dropped later).

    0 = Anesthesia     (00100–01999)
    1 = Surgery        (10000–69999)
    2 = Radiology      (70000–79999)
    3 = Lab/Pathology  (80000–89999)
    4 = Medicine/E&M   (90000–99999)
    5 = HCPCS Level II (A–V prefix)
    """
    s = hcpcs_series.astype(str).str.strip()

    # Alpha-prefix codes → HCPCS Level II
    is_alpha = s.str[0].str.isalpha().fillna(False)

    # Numeric codes → parse and bucket
    numeric = pd.to_numeric(s, errors="coerce")
    buckets = pd.Series(np.nan, index=s.index, dtype="float64")

    buckets[is_alpha] = 5.0
    buckets[(numeric >= 100)   & (numeric <= 1999)]  = 0.0
    buckets[(numeric >= 10000) & (numeric <= 69999)] = 1.0
    buckets[(numeric >= 70000) & (numeric <= 79999)] = 2.0
    buckets[(numeric >= 80000) & (numeric <= 89999)] = 3.0
    buckets[(numeric >= 90000) & (numeric <= 99999)] = 4.0

    return buckets


# ── Provider risk score loading ──────────────────────────────────────────────

def load_provider_risk_scores(provider_data_dir: str) -> pd.DataFrame:
    """
    Read CMS 'by Provider' CSVs and extract (Rndrng_NPI, year, Bene_Avg_Risk_Scre).
    Returns a DataFrame for left-joining into the gold pipeline.
    """
    provider_data_dir = os.path.abspath(provider_data_dir)
    csv_files = sorted(glob.glob(os.path.join(provider_data_dir, "provider_summary_*.csv")))

    if not csv_files:
        print(f"  [WARN] No provider_summary_*.csv files found in '{provider_data_dir}'")
        print(f"         Run pull_provider_data.py first. Risk scores will be NaN.")
        return pd.DataFrame(columns=["Rndrng_NPI", "year", "Bene_Avg_Risk_Scre"])

    print(f"Loading provider risk scores from {len(csv_files)} files...")
    parts = []
    for f in csv_files:
        year_match = re.search(r'(\d{4})', os.path.basename(f))
        year = int(year_match.group(1)) if year_match else 0

        try:
            df = pd.read_csv(f, usecols=["Rndrng_NPI", "Bene_Avg_Risk_Scre"], dtype=str)
            df["year"] = year
            df["Bene_Avg_Risk_Scre"] = pd.to_numeric(df["Bene_Avg_Risk_Scre"], errors="coerce")
            df["Rndrng_NPI"] = df["Rndrng_NPI"].astype(str).str.strip()
            df = df.dropna(subset=["Bene_Avg_Risk_Scre"])
            parts.append(df)
            print(f"  {year}: {len(df):,} NPIs with risk scores")
        except Exception as e:
            print(f"  [WARN] Could not read {f}: {e}")

    if not parts:
        return pd.DataFrame(columns=["Rndrng_NPI", "year", "Bene_Avg_Risk_Scre"])

    risk_df = pd.concat(parts, ignore_index=True)
    print(f"  Total risk score records: {len(risk_df):,}")
    return risk_df


# ── Pass 1: fit global label encoders ────────────────────────────────────────

def fit_encoders(silver_files: list[str]) -> dict[str, LabelEncoder]:
    print("Pass 1/2: Collecting unique category values for LabelEncoder fitting...")

    unique: dict[str, set] = {col: set() for col in CAT_COLS}
    for f in silver_files:
        try:
            cols_present = [c for c in CAT_COLS if c in pq.read_schema(f).names]
            if not cols_present:
                continue
            chunk = pd.read_parquet(f, columns=cols_present)
            for col in cols_present:
                unique[col].update(chunk[col].fillna("UNKNOWN").astype(str).unique())
        except Exception as e:
            print(f"  [WARN] Could not read {f}: {e}")

    encoders: dict[str, LabelEncoder] = {}
    for col in CAT_COLS:
        if unique[col]:
            le = LabelEncoder()
            le.fit(sorted(unique[col]))
            encoders[col] = le
            print(f"  {col}: {len(le.classes_):,} unique values")

    return encoders


def save_encoders(encoders: dict[str, LabelEncoder], gold_dir: str):
    mapping = {col: le.classes_.tolist() for col, le in encoders.items()}
    path = os.path.join(gold_dir, "label_encoders.json")
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"  Encoder classes saved → {path}")


# ── Feature engineering ──────────────────────────────────────────────────────

def engineer_df(df, encoders: dict[str, LabelEncoder], xp, using_gpu: bool):
    """
    df    — cudf.DataFrame (GPU) or pandas.DataFrame (CPU)
    xp    — cudf or pandas module
    """
    # Place of service flag: F=1 (facility), O=0 (office)
    if "Place_Of_Srvc" in df.columns:
        pos = df["Place_Of_Srvc"].fillna("O").astype(str).str.strip().str.upper().str[0]
        df["place_of_srvc_flag"] = (pos == "F").astype(float)
    else:
        df["place_of_srvc_flag"] = 0.0

    # HCPCS bucket (coarse clinical category)
    if "HCPCS_Cd" in df.columns:
        if using_gpu:
            hcpcs_pd = df["HCPCS_Cd"].to_pandas() if hasattr(df["HCPCS_Cd"], "to_pandas") else df["HCPCS_Cd"]
            df["hcpcs_bucket"] = hcpcs_to_bucket(hcpcs_pd).values
        else:
            df["hcpcs_bucket"] = hcpcs_to_bucket(df["HCPCS_Cd"])

    # Log-transformed counts
    for raw_col, log_col in [("Tot_Srvcs", "log_srvcs"), ("Tot_Benes", "log_benes")]:
        if raw_col in df.columns:
            vals = pd.to_numeric(df[raw_col] if not using_gpu else df[raw_col].to_pandas(),
                                 errors="coerce").fillna(0)
            df[log_col] = np.log1p(vals.values)

    # Services per beneficiary ratio
    if "Tot_Srvcs" in df.columns and "Tot_Benes" in df.columns:
        tot_srvcs = pd.to_numeric(df["Tot_Srvcs"] if not using_gpu else df["Tot_Srvcs"].to_pandas(),
                                   errors="coerce")
        tot_benes = pd.to_numeric(df["Tot_Benes"] if not using_gpu else df["Tot_Benes"].to_pandas(),
                                   errors="coerce")
        ratio = (tot_srvcs / tot_benes).where(tot_benes > 0, other=float("nan"))
        df["srvcs_per_bene"] = ratio.values

    # Label encoding via vectorized merge
    for col, le in encoders.items():
        if col not in df.columns:
            continue
        fallback = len(le.classes_)
        lookup = xp.DataFrame({
            col:          le.classes_,
            f"{col}_idx": np.arange(len(le.classes_), dtype=np.float64),
        })
        df[col] = df[col].fillna("UNKNOWN").astype(str)
        df = df.merge(lookup, on=col, how="left")
        df[f"{col}_idx"] = df[f"{col}_idx"].fillna(float(fallback))

    # Cast year to int16
    if "year" in df.columns:
        if using_gpu:
            df["year"] = df["year"].astype("int16")
        else:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int16")

    present = [c for c in FEATURE_COLS if c in df.columns]
    df_out = df[present].dropna()

    if using_gpu:
        df_out = df_out.to_pandas()

    for col in df_out.columns:
        if col == "year":
            df_out[col] = df_out[col].astype("int16")
        else:
            df_out[col] = df_out[col].astype("float64")

    return df_out


# ── Pass 2: process state-by-state ───────────────────────────────────────────

def engineer(silver_dir: str, gold_dir: str, provider_data_dir: str, force_cpu: bool = False):
    silver_dir   = os.path.abspath(silver_dir)
    silver_files = sorted(glob.glob(os.path.join(silver_dir, "*.parquet")))

    if not silver_files:
        raise FileNotFoundError(
            f"No silver parquets found under '{silver_dir}'.\n"
            f"  Run 02_silver_clean_local.py first."
        )
    print(f"Found {len(silver_files)} silver state files in '{silver_dir}'")

    xp, using_gpu = _load_backend(force_cpu)
    encoders      = fit_encoders(silver_files)

    os.makedirs(gold_dir, exist_ok=True)
    save_encoders(encoders, gold_dir)

    # Load provider risk scores for NPI join
    risk_df = load_provider_risk_scores(provider_data_dir)
    has_risk = len(risk_df) > 0
    if has_risk:
        global_median_risk = risk_df["Bene_Avg_Risk_Scre"].median()
    else:
        global_median_risk = 1.0  # CMS national average is ~1.0

    print(f"\nPass 2/2: Engineering features ({'GPU' if using_gpu else 'CPU'}) → {gold_dir}/")
    total_rows = 0

    for i, f in enumerate(silver_files, 1):
        state = os.path.splitext(os.path.basename(f))[0]
        try:
            if using_gpu:
                import cudf
                df = cudf.read_parquet(f)
            else:
                df = pd.read_parquet(f)

            # Join risk scores on NPI + year
            if has_risk and "Rndrng_NPI" in df.columns and "year" in df.columns:
                if using_gpu:
                    df_pd = df.to_pandas()
                else:
                    df_pd = df

                df_pd["Rndrng_NPI"] = df_pd["Rndrng_NPI"].astype(str).str.strip()
                df_pd["year_int"] = pd.to_numeric(df_pd["year"], errors="coerce").astype("Int64")
                risk_join = risk_df.copy()
                risk_join["year"] = risk_join["year"].astype("Int64")
                df_pd = df_pd.merge(
                    risk_join.rename(columns={"year": "year_int"}),
                    on=["Rndrng_NPI", "year_int"],
                    how="left",
                )
                df_pd.drop(columns=["year_int"], inplace=True)
                # Fill missing risk scores with global median
                df_pd["Bene_Avg_Risk_Scre"] = df_pd["Bene_Avg_Risk_Scre"].fillna(global_median_risk)

                if using_gpu:
                    df = cudf.from_pandas(df_pd)
                else:
                    df = df_pd
            else:
                df["Bene_Avg_Risk_Scre"] = global_median_risk

            df_gold = engineer_df(df, encoders, xp, using_gpu)

            if df_gold.empty:
                print(f"  [{i:>2}/{len(silver_files)}] {state:<6}  (no rows after engineering)")
                continue

            table    = pa.Table.from_pandas(df_gold, schema=GOLD_SCHEMA, preserve_index=False)
            out_path = os.path.join(gold_dir, f"{state}.parquet")
            pq.write_table(table, out_path)

            total_rows += len(df_gold)
            print(f"  [{i:>2}/{len(silver_files)}] {state:<6}  {len(df_gold):>8,} rows")

        except Exception as e:
            print(f"  [WARN] Skipping {state}: {e}")

    print(f"\nGold layer complete — {total_rows:,} rows across {len(silver_files)} states → {gold_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Engineer features from Silver parquets into Gold (GPU-accelerated)."
    )
    parser.add_argument("--silver-dir", default=DEFAULT_SILVER,
                        help=f"Directory of per-state silver parquets (default: {DEFAULT_SILVER})")
    parser.add_argument("--gold-dir", default=DEFAULT_GOLD,
                        help=f"Output directory for per-state gold parquets (default: {DEFAULT_GOLD})")
    parser.add_argument("--provider-data-dir", default=DEFAULT_PROV,
                        help=f"Directory with provider_summary_*.csv files (default: {DEFAULT_PROV})")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration and use pandas (CPU) instead")
    args = parser.parse_args()
    engineer(args.silver_dir, args.gold_dir, args.provider_data_dir, force_cpu=args.no_gpu)
