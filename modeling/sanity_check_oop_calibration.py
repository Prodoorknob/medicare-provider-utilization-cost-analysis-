"""sanity_check_oop_calibration.py — sweep top-5 / bottom-5 specialties by service volume.

For each specialty, runs predict_stage2 with that specialty's median Stage 1 allowed
amount and the most-common hcpcs_bucket as inputs, holding all beneficiary inputs
constant. Reports raw vs calibrated P10/P50/P90 side-by-side.
"""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "api"))
os.environ.setdefault("SUPABASE_URL", "http://localhost")
os.environ.setdefault("SUPABASE_ANON_KEY", "placeholder")

from models.loader import load_all_models  # noqa: E402
from services.prediction import predict_stage2  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SILVER_GLOB = os.path.join(PROJECT_ROOT, "local_pipeline", "silver", "*.parquet")

# HCPCS to bucket mapping — mirrors notebooks/03_gold_features_local.py
def hcpcs_bucket(code: str) -> int:
    if not isinstance(code, str) or not code:
        return 4
    c = code.strip().upper()
    if c[0].isdigit():
        n = int(c[:5]) if c[:5].isdigit() else 0
        if    100 <= n <= 1999:  return 0  # Anesthesia
        if  10000 <= n <= 69999: return 1  # Surgery
        if  70000 <= n <= 79999: return 2  # Radiology
        if  80000 <= n <= 89999: return 3  # Lab
        if  90000 <= n <= 99999: return 4  # Medicine / E&M
    return 5  # HCPCS Level II


def aggregate_specialties() -> pd.DataFrame:
    """Aggregate total Tot_Srvcs and median allowed by specialty across silver."""
    pieces = []
    for path in sorted(glob.glob(SILVER_GLOB)):
        df = pd.read_parquet(
            path,
            columns=["Rndrng_Prvdr_Type", "HCPCS_Cd", "Place_Of_Srvc",
                     "Tot_Srvcs", "Avg_Mdcr_Alowd_Amt"],
        )
        df["hcpcs_bucket"] = df["HCPCS_Cd"].apply(hcpcs_bucket)
        df["place_flag"] = (df["Place_Of_Srvc"].astype(str).str.upper() == "F").astype(int)
        pieces.append(df)
    full = pd.concat(pieces, ignore_index=True)

    grp = full.groupby("Rndrng_Prvdr_Type")
    summary = pd.DataFrame({
        "total_services": grp["Tot_Srvcs"].sum(),
        "median_allowed": grp["Avg_Mdcr_Alowd_Amt"].median(),
        "modal_bucket": grp["hcpcs_bucket"].agg(lambda s: int(s.mode().iat[0])),
        "modal_pos_facility": grp["place_flag"].agg(lambda s: int(s.mode().iat[0])),
        "n_rows": grp.size(),
    }).reset_index()
    return summary.sort_values("total_services", ascending=False)


def run_specialty(art, row, q_lo, q_hi):
    """Predict raw and calibrated bounds for one specialty row."""
    # Calibrated
    art.oop_q_lo, art.oop_q_hi = q_lo, q_hi
    pc10, pc50, pc90, _ = predict_stage2(
        art,
        allowed_amount=float(row["median_allowed"]),
        risk_score=None,
        provider_type=row["Rndrng_Prvdr_Type"],
        hcpcs_bucket=int(row["modal_bucket"]),
        place_of_service=int(row["modal_pos_facility"]),
        state="TX",  # holding state constant for fair comparison
        age=70, sex=0, income=1, chronic_count=2,
        dual_eligible=0, has_supplemental=0,
    )
    # Raw
    art.oop_q_lo, art.oop_q_hi = 0.0, 0.0
    pr10, pr50, pr90, _ = predict_stage2(
        art,
        allowed_amount=float(row["median_allowed"]),
        risk_score=None,
        provider_type=row["Rndrng_Prvdr_Type"],
        hcpcs_bucket=int(row["modal_bucket"]),
        place_of_service=int(row["modal_pos_facility"]),
        state="TX",
        age=70, sex=0, income=1, chronic_count=2,
        dual_eligible=0, has_supplemental=0,
    )
    return pr10, pr50, pr90, pc10, pc50, pc90


def main():
    print("Loading production artifacts...")
    art = load_all_models(os.path.join(PROJECT_ROOT, "api", "models", "artifacts"))
    q_lo, q_hi = art.oop_q_lo, art.oop_q_hi
    print()

    print("Aggregating specialties from silver layer...")
    summary = aggregate_specialties()
    print(f"  {len(summary)} unique specialties\n")

    top5 = summary.head(5)
    bot5 = summary.tail(5).iloc[::-1]

    def render_block(title, df):
        print("=" * 110)
        print(title)
        print("=" * 110)
        header = (
            f"{'specialty':<48} {'svcs (M)':>10} {'allowed':>9} "
            f"{'bkt':>4} {'P90 raw':>10} {'P90 cal':>10} {'delta':>8}"
        )
        print(header)
        print("-" * 110)
        for _, row in df.iterrows():
            pr10, pr50, pr90, pc10, pc50, pc90 = run_specialty(art, row, q_lo, q_hi)
            print(
                f"{row['Rndrng_Prvdr_Type']:<48} "
                f"{row['total_services']/1e6:>10.2f} "
                f"${row['median_allowed']:>8.2f} "
                f"{int(row['modal_bucket']):>4d} "
                f"${pr90:>9.2f} ${pc90:>9.2f} {pc90-pr90:>+7.2f}"
            )
            print(
                f"{'':<48} "
                f"{'P10:':>10} ${pr10:>7.2f} -> ${pc10:.2f}    "
                f"P50: ${pr50:.2f}"
            )
        print()

    render_block("TOP 5 SPECIALTIES BY TOTAL SERVICES", top5)
    render_block("BOTTOM 5 SPECIALTIES BY TOTAL SERVICES", bot5)

    print(f"Calibration constants from sidecar:  q_lo={q_lo:+.4f}   q_hi={q_hi:+.4f}")
    print("State held at TX, beneficiary inputs constant (age=70, income=1, no dual/suppl)")


if __name__ == "__main__":
    main()
