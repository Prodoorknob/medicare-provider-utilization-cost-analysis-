"""
upload_all.py — Pre-compute aggregations and upload all data to Supabase.
Usage:
    set SUPABASE_URL=https://zdkoniqnvbklxtsviikl.supabase.co
    set SUPABASE_SERVICE_KEY=<service-role-key>
    python web/scripts/upload_all.py
"""
import os, sys, glob, json, numpy as np, pandas as pd
from supabase import create_client

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_PIPELINE = os.path.join(_PROJECT_ROOT, "local_pipeline")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    print("Set SUPABASE_URL and SUPABASE_SERVICE_KEY env vars"); sys.exit(1)

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

def batch_insert(table, rows, batch_size=500):
    total = len(rows)
    for i in range(0, total, batch_size):
        chunk = rows[i:i+batch_size]
        sb.table(table).insert(chunk).execute()
        done = min(i + batch_size, total)
        if done % 2000 == 0 or done >= total:
            print(f"  {done:,}/{total:,}")

def upload_forecasts():
    print("\n=== LSTM Forecasts ===")
    df = pd.read_parquet(os.path.join(LOCAL_PIPELINE, "lstm", "forecast_2024_2026.parquet"))
    print(f"  Loaded {len(df):,} rows")
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "specialty_idx": int(r["Rndrng_Prvdr_Type_idx"]),
            "hcpcs_bucket": int(r["hcpcs_bucket"]),
            "state_idx": int(r["Rndrng_Prvdr_State_Abrvtn_idx"]),
            "forecast_year": int(r["forecast_year"]),
            "forecast_mean": round(float(r["forecast_mean"]), 2),
            "forecast_std": round(float(r["forecast_std"]), 2) if pd.notna(r["forecast_std"]) else None,
            "forecast_p10": round(float(r["forecast_p10"]), 2),
            "forecast_p50": round(float(r["forecast_p50"]), 2),
            "forecast_p90": round(float(r["forecast_p90"]), 2),
            "last_known_year": int(r["last_known_year"]) if pd.notna(r["last_known_year"]) else None,
            "last_known_value": round(float(r["last_known_value"]), 2) if pd.notna(r["last_known_value"]) else None,
            "n_history_years": int(r["n_history_years"]) if pd.notna(r["n_history_years"]) else None,
        })
    batch_insert("lstm_forecasts", rows)
    print(f"  Done: {len(rows):,}")

def upload_stage1():
    print("\n=== Stage 1 Allowed Amounts ===")
    gold_dir = os.path.join(LOCAL_PIPELINE, "gold")
    files = sorted(glob.glob(os.path.join(gold_dir, "*.parquet")))
    COLS = ["Rndrng_Prvdr_Type_idx","hcpcs_bucket","Rndrng_Prvdr_State_Abrvtn_idx",
            "place_of_srvc_flag","Avg_Mdcr_Alowd_Amt","Avg_Sbmtd_Chrg","Bene_Avg_Risk_Scre"]
    GROUP = COLS[:4]
    parts = []
    for i, f in enumerate(files, 1):
        bn = os.path.splitext(os.path.basename(f))[0]
        if bn == "label_encoders": continue
        try:
            d = pd.read_parquet(f, columns=COLS).dropna(subset=["Avg_Mdcr_Alowd_Amt"])
            if d.empty: continue
            agg = d.groupby(GROUP).agg(
                n=("Avg_Mdcr_Alowd_Amt","count"), mean_a=("Avg_Mdcr_Alowd_Amt","mean"),
                med_a=("Avg_Mdcr_Alowd_Amt","median"),
                p10_a=("Avg_Mdcr_Alowd_Amt", lambda x: np.percentile(x,10)),
                p90_a=("Avg_Mdcr_Alowd_Amt", lambda x: np.percentile(x,90)),
                mean_c=("Avg_Sbmtd_Chrg","mean"), mean_r=("Bene_Avg_Risk_Scre","mean"),
            ).reset_index()
            parts.append(agg)
        except Exception as e:
            print(f"  WARN {bn}: {e}")
        if i % 10 == 0: print(f"  Processed {i}/{len(files)} states...")
    s1 = pd.concat(parts, ignore_index=True)
    s1 = s1[s1["n"] >= 5]
    print(f"  Total groups (n>=5): {len(s1):,}")
    rows = []
    for _, r in s1.iterrows():
        rows.append({
            "specialty_idx": int(r["Rndrng_Prvdr_Type_idx"]),
            "hcpcs_bucket": int(r["hcpcs_bucket"]),
            "state_idx": int(r["Rndrng_Prvdr_State_Abrvtn_idx"]),
            "place_of_service": int(r["place_of_srvc_flag"]),
            "n_records": int(r["n"]),
            "mean_allowed": round(float(r["mean_a"]),2),
            "median_allowed": round(float(r["med_a"]),2),
            "p10_allowed": round(float(r["p10_a"]),2),
            "p90_allowed": round(float(r["p90_a"]),2),
            "mean_charge": round(float(r["mean_c"]),2) if pd.notna(r["mean_c"]) else None,
            "mean_risk_score": round(float(r["mean_r"]),4) if pd.notna(r["mean_r"]) else None,
        })
    batch_insert("stage1_allowed_amounts", rows)
    print(f"  Done: {len(rows):,}")

def upload_stage2():
    print("\n=== Stage 2 OOP Estimates ===")
    df = pd.read_parquet(os.path.join(LOCAL_PIPELINE, "mcbs_synthetic", "synthetic_oop.parquet"))
    print(f"  Loaded {len(df):,} rows")
    GRP = ["Rndrng_Prvdr_Type_idx","hcpcs_bucket","census_region",
           "dual_eligible","has_supplemental","age","income"]
    agg = df.groupby(GRP).agg(
        n=("per_service_oop","count"),
        p10=("per_service_oop", lambda x: np.percentile(x,10)),
        p50=("per_service_oop", lambda x: np.percentile(x,50)),
        p90=("per_service_oop", lambda x: np.percentile(x,90)),
        ma=("Avg_Mdcr_Alowd_Amt","mean"),
    ).reset_index()
    agg = agg[agg["n"] >= 10]
    print(f"  Total groups (n>=10): {len(agg):,}")
    rows = []
    for _, r in agg.iterrows():
        rows.append({
            "specialty_idx": int(r["Rndrng_Prvdr_Type_idx"]),
            "hcpcs_bucket": int(r["hcpcs_bucket"]),
            "census_region": int(r["census_region"]),
            "dual_eligible": int(r["dual_eligible"]),
            "has_supplemental": int(r["has_supplemental"]),
            "age_group": int(r["age"]),
            "income_bracket": int(r["income"]),
            "n_records": int(r["n"]),
            "oop_p10": round(float(r["p10"]),2),
            "oop_p50": round(float(r["p50"]),2),
            "oop_p90": round(float(r["p90"]),2),
            "mean_allowed": round(float(r["ma"]),2) if pd.notna(r["ma"]) else None,
        })
    batch_insert("stage2_oop_estimates", rows)
    print(f"  Done: {len(rows):,}")

if __name__ == "__main__":
    # Clear any partial data first
    print("Clearing partial data...")
    sb.table("lstm_forecasts").delete().neq("id", -1).execute()
    sb.table("stage1_allowed_amounts").delete().neq("id", -1).execute()
    sb.table("stage2_oop_estimates").delete().neq("id", -1).execute()

    upload_forecasts()
    upload_stage1()
    upload_stage2()
    print("\n=== All uploads complete ===")
