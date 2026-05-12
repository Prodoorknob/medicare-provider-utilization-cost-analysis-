"""
revenue_residuals.py -- LightGBM-driven residual sidecar for REVENUE_DEVIATION rule

For each silver row (NPI x HCPCS x POS x year) build the same 12-feature vector
the API uses at inference, score with the production lgbm_v2_no_charge model,
back-transform via expm1, then aggregate residuals to the (NPI, year) grain.

The rule signal is:
    revenue_ratio = actual_total_allowed / expected_total_allowed
where each "total allowed" is a service-weighted sum across the NPI-year's
silver rows. A revenue_ratio meaningfully above 1.0 indicates the NPI
realizes more allowed dollars per service than peers on the same specific mix
of code, place-of-service, and patient acuity -- a signal that catches
modifier abuse and non-E&M code-family upcoding the per-specialty rules miss.

Feature parity with api/services/prediction.build_stage1_features is critical;
any drift here biases every residual.

Inputs
------
    local_pipeline/silver/{STATE}.parquet
    data/provider_summary_*.csv               (for Bene_Avg_Risk_Scre)
    api/models/artifacts/lgbm_v2_no_charge.txt
    api/models/artifacts/label_encoders.json
    api/models/artifacts/hcpcs_target_enc.json

Output
------
    local_pipeline/anomaly/revenue_residuals.parquet
        keyed by (Rndrng_NPI, year)

Usage
-----
    python anomaly/rules/revenue_residuals.py
    python anomaly/rules/revenue_residuals.py --states CA,TX,FL
    python anomaly/rules/revenue_residuals.py --sample 0.1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


_PROJECT_ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SILVER    = os.path.join(_PROJECT_ROOT, "local_pipeline", "silver")
DEFAULT_OUT_DIR   = os.path.join(_PROJECT_ROOT, "local_pipeline", "anomaly")
DEFAULT_OUTPUT    = os.path.join(DEFAULT_OUT_DIR, "revenue_residuals.parquet")
DEFAULT_PROV      = os.path.join(_PROJECT_ROOT, "data")
DEFAULT_ARTIFACTS = os.path.join(_PROJECT_ROOT, "api", "models", "artifacts")

SILVER_COLS = [
    "Rndrng_NPI",
    "Rndrng_Prvdr_Type",
    "Rndrng_Prvdr_State_Abrvtn",
    "HCPCS_Cd",
    "Place_Of_Srvc",
    "Tot_Benes",
    "Tot_Srvcs",
    "Avg_Mdcr_Alowd_Amt",
    "year",
]

# Project defaults for missing continuous features (mirrors api/config.py).
DEFAULT_RISK_SCORE = 1.0
DEFAULT_TOT_SRVCS  = 50.0
DEFAULT_TOT_BENES  = 20.0


def hcpcs_to_bucket(s: pd.Series) -> pd.Series:
    """Mirror of notebooks/03_gold_features_local.py and api/services/prediction.py."""
    s = s.astype(str).str.strip()
    is_alpha = s.str[0].str.isalpha().fillna(False)
    numeric = pd.to_numeric(s, errors="coerce")
    buckets = pd.Series(4, index=s.index, dtype="int64")  # default Medicine/E&M
    buckets[is_alpha] = 5
    buckets[(numeric >= 100)   & (numeric <= 1999)]  = 0
    buckets[(numeric >= 10000) & (numeric <= 69999)] = 1
    buckets[(numeric >= 70000) & (numeric <= 79999)] = 2
    buckets[(numeric >= 80000) & (numeric <= 89999)] = 3
    buckets[(numeric >= 90000) & (numeric <= 99999)] = 4
    return buckets


def load_risk_scores(provider_dir: str) -> pd.DataFrame:
    """Read CMS 'by Provider' CSVs -> (Rndrng_NPI, year, risk_score)."""
    csv_files = sorted(glob.glob(os.path.join(provider_dir, "provider_summary_*.csv")))
    if not csv_files:
        print(f"  [WARN] No provider_summary_*.csv in '{provider_dir}'; risk_score will use default {DEFAULT_RISK_SCORE}")
        return pd.DataFrame(columns=["Rndrng_NPI", "year", "risk_score"])

    parts = []
    for f in csv_files:
        m = re.search(r"(\d{4})", os.path.basename(f))
        if not m:
            continue
        yr = int(m.group(1))
        try:
            df = pd.read_csv(f, usecols=["Rndrng_NPI", "Bene_Avg_Risk_Scre"], dtype=str)
            df["year"] = yr
            df["risk_score"] = pd.to_numeric(df["Bene_Avg_Risk_Scre"], errors="coerce")
            df["Rndrng_NPI"] = df["Rndrng_NPI"].astype(str).str.strip()
            df = df.dropna(subset=["risk_score"])[["Rndrng_NPI", "year", "risk_score"]]
            parts.append(df)
        except Exception as e:
            print(f"  [WARN] {f}: {e}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["Rndrng_NPI", "year", "risk_score"]
    )


def build_feature_matrix(
    df: pd.DataFrame,
    specialty_to_idx: dict[str, int],
    state_to_idx: dict[str, int],
    hcpcs_to_idx: dict[str, int],
    hcpcs_target_enc: dict[str, float],
    hcpcs_global_mean: float,
) -> np.ndarray:
    """Vectorized version of api/services/prediction.build_stage1_features.

    Returns an (n_rows, 12) float64 matrix in the same column order the
    no-charge model was trained on. NO Avg_Sbmtd_Chrg slot (the no-charge
    variant omits it; lgbm_v2_no_charge.txt's feature_name() reflects this).
    """
    n_specialties = len(specialty_to_idx)
    n_states = len(state_to_idx)
    n_hcpcs  = len(hcpcs_to_idx)

    # Categorical indices with unknown-fallback (matches API)
    specialty_idx = df["Rndrng_Prvdr_Type"].map(specialty_to_idx).fillna(n_specialties).astype("int64")
    state_idx     = df["Rndrng_Prvdr_State_Abrvtn"].map(state_to_idx).fillna(n_states).astype("int64")
    hcpcs_idx     = df["HCPCS_Cd"].map(hcpcs_to_idx).fillna(n_hcpcs).astype("int64")

    hcpcs_bucket = hcpcs_to_bucket(df["HCPCS_Cd"]).astype("int64")
    # POS: silver uses 'F' (facility) / 'O' (office). Cast to 1/0.
    pos_flag = (df["Place_Of_Srvc"].astype(str).str.upper() == "F").astype("int64")

    # Risk score: prefer joined value, fall back to project default 1.0
    risk = df["risk_score"].fillna(DEFAULT_RISK_SCORE).astype("float64")

    # Volume features: default project values when missing
    srvcs = pd.to_numeric(df["Tot_Srvcs"], errors="coerce").fillna(DEFAULT_TOT_SRVCS).astype("float64")
    benes = pd.to_numeric(df["Tot_Benes"], errors="coerce").fillna(DEFAULT_TOT_BENES).astype("float64")
    log_srvcs = np.log1p(srvcs)
    log_benes = np.log1p(benes)

    # NOTE: API computes srvcs_per_bene = log_srvcs / log_benes (not the natural ratio).
    # Replicate verbatim or residuals will be systematically biased.
    srvcs_per_bene = np.where(log_benes > 0, log_srvcs / log_benes, 0.0)

    specialty_bucket = specialty_idx.to_numpy() * 6.0 + hcpcs_bucket.to_numpy()
    pos_bucket       = pos_flag.to_numpy()     * 6.0 + hcpcs_bucket.to_numpy()

    # HCPCS target encoding lookup
    hcpcs_te = (
        df["HCPCS_Cd"].astype(str).str.strip()
          .map(hcpcs_target_enc)
          .fillna(hcpcs_global_mean)
          .astype("float64")
          .to_numpy()
    )

    X = np.column_stack([
        specialty_idx.to_numpy(dtype="float64"),    # Rndrng_Prvdr_Type_idx
        state_idx.to_numpy(dtype="float64"),        # Rndrng_Prvdr_State_Abrvtn_idx
        hcpcs_idx.to_numpy(dtype="float64"),        # HCPCS_Cd_idx
        hcpcs_bucket.to_numpy(dtype="float64"),     # hcpcs_bucket
        pos_flag.to_numpy(dtype="float64"),         # place_of_srvc_flag
        risk.to_numpy(dtype="float64"),             # Bene_Avg_Risk_Scre
        log_srvcs.to_numpy(dtype="float64"),        # log_srvcs
        log_benes.to_numpy(dtype="float64"),        # log_benes
        srvcs_per_bene.astype("float64"),           # srvcs_per_bene
        specialty_bucket.astype("float64"),         # specialty_bucket
        pos_bucket.astype("float64"),               # pos_bucket
        hcpcs_te,                                   # hcpcs_target_enc
    ])
    return X


def score_silver_state(
    df: pd.DataFrame,
    booster: lgb.Booster,
    encoders: dict,
    target_enc: dict[str, float],
    target_global_mean: float,
) -> pd.DataFrame:
    """Score one state's silver, return per-(NPI, year) residual rows."""
    df = df.dropna(subset=["Rndrng_NPI", "HCPCS_Cd", "Tot_Srvcs", "Avg_Mdcr_Alowd_Amt"]).copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int16")
    df = df.dropna(subset=["year"])
    df["Rndrng_NPI"] = df["Rndrng_NPI"].astype(str).str.strip()
    df["HCPCS_Cd"]   = df["HCPCS_Cd"].astype(str).str.strip()

    X = build_feature_matrix(
        df,
        specialty_to_idx={n: i for i, n in enumerate(encoders["Rndrng_Prvdr_Type"])},
        state_to_idx    ={n: i for i, n in enumerate(encoders["Rndrng_Prvdr_State_Abrvtn"])},
        hcpcs_to_idx    ={c: i for i, c in enumerate(encoders["HCPCS_Cd"])},
        hcpcs_target_enc=target_enc,
        hcpcs_global_mean=target_global_mean,
    )

    # Model trained on log1p(target) -> back-transform with expm1; floor at 0.
    log_pred = booster.predict(X)
    pred_per_service = np.maximum(0.0, np.expm1(log_pred)).astype("float64")

    df["row_actual_allowed"]   = df["Tot_Srvcs"] * df["Avg_Mdcr_Alowd_Amt"]
    df["row_expected_allowed"] = df["Tot_Srvcs"] * pred_per_service

    agg = df.groupby(["Rndrng_NPI", "year"], observed=True, sort=False).agg(
        actual_total_allowed   =("row_actual_allowed",   "sum"),
        expected_total_allowed =("row_expected_allowed", "sum"),
        n_rows_scored          =("HCPCS_Cd", "size"),
        specialty              =("Rndrng_Prvdr_Type",        "first"),
        state                  =("Rndrng_Prvdr_State_Abrvtn", "first"),
    ).reset_index()

    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver-dir",    default=DEFAULT_SILVER)
    ap.add_argument("--provider-dir",  default=DEFAULT_PROV)
    ap.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS)
    ap.add_argument("--model-file",    default="lgbm_v2_no_charge.txt")
    ap.add_argument("--output",        default=DEFAULT_OUTPUT)
    ap.add_argument("--states",        default="", help="Comma-separated state filter (default all)")
    ap.add_argument("--sample",        type=float, default=1.0,
                    help="Row-level sample fraction per state for dev")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load model + encodings once
    print(f"Loading model: {os.path.join(args.artifacts_dir, args.model_file)}")
    booster = lgb.Booster(model_file=os.path.join(args.artifacts_dir, args.model_file))
    feat_names = booster.feature_name()
    print(f"  trees={booster.num_trees()}, features={feat_names}")
    if "Avg_Sbmtd_Chrg" in feat_names:
        raise SystemExit(
            "ERROR: charge-using model selected but this scorer is no-charge only. "
            "Pass --model-file lgbm_v2_no_charge.txt."
        )

    with open(os.path.join(args.artifacts_dir, "label_encoders.json")) as f:
        encoders = json.load(f)
    with open(os.path.join(args.artifacts_dir, "hcpcs_target_enc.json")) as f:
        te = json.load(f)
    target_enc = te.get("codes", {})
    target_global_mean = float(te.get("global_mean", 76.34))
    print(f"  encoders: {len(encoders.get('Rndrng_Prvdr_Type', []))} specialties, "
          f"{len(encoders.get('Rndrng_Prvdr_State_Abrvtn', []))} states, "
          f"{len(encoders.get('HCPCS_Cd', []))} HCPCS codes")
    print(f"  hcpcs_target_enc: {len(target_enc):,} codes, global_mean={target_global_mean:.2f}")

    # Risk scores (joined per (NPI, year))
    risk_df = load_risk_scores(args.provider_dir)
    if not risk_df.empty:
        risk_df["year"] = risk_df["year"].astype("Int16")
        print(f"  risk scores: {len(risk_df):,} (NPI, year) rows")

    # Iterate silver files
    files = sorted(glob.glob(os.path.join(args.silver_dir, "*.parquet")))
    if args.states:
        wanted = {s.strip().upper() for s in args.states.split(",") if s.strip()}
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in wanted]
    if not files:
        raise SystemExit(f"No silver parquets in {args.silver_dir}")

    print(f"Silver files: {len(files)}  | sample={args.sample}")
    print(f"Output:       {args.output}\n")

    parts = []
    t0 = time.time()
    for i, f in enumerate(files, 1):
        state = os.path.splitext(os.path.basename(f))[0]
        try:
            schema_cols = pq.read_schema(f).names
            use_cols = [c for c in SILVER_COLS if c in schema_cols]
            df = pd.read_parquet(f, columns=use_cols)
            if args.sample < 1.0:
                df = df.sample(frac=args.sample, random_state=42)
            # Normalize join keys BEFORE the merge: silver's year is often object
            # dtype, risk_df's is Int16 -- mismatched dtypes cause merge to error.
            df["Rndrng_NPI"] = df["Rndrng_NPI"].astype(str).str.strip()
            df["year"]       = pd.to_numeric(df["year"], errors="coerce").astype("Int16")
            df = df.dropna(subset=["year"])
            # Join risk scores (left join; missing -> default in build_feature_matrix)
            if not risk_df.empty:
                df = df.merge(risk_df, on=["Rndrng_NPI", "year"], how="left")
            else:
                df["risk_score"] = np.nan

            agg = score_silver_state(df, booster, encoders, target_enc, target_global_mean)
            parts.append(agg)
            print(f"  [{i:2d}/{len(files)}] {state}: {len(df):>9,} rows -> "
                  f"{len(agg):>7,} (NPI, year) residuals")
        except Exception as e:
            print(f"  [WARN] {state}: {e}")
            continue

    out = pd.concat(parts, ignore_index=True)

    # Some NPIs span multiple states within a year (rare but real); collapse.
    # We keep the dominant-state row but sum the dollars so the ratio is correct.
    if out.duplicated(subset=["Rndrng_NPI", "year"]).any():
        before = len(out)
        out = (
            out.groupby(["Rndrng_NPI", "year"], observed=True, sort=False)
               .agg(
                   actual_total_allowed   =("actual_total_allowed",   "sum"),
                   expected_total_allowed =("expected_total_allowed", "sum"),
                   n_rows_scored          =("n_rows_scored",          "sum"),
                   specialty              =("specialty",              "first"),
                   state                  =("state",                  "first"),
               ).reset_index()
        )
        print(f"  collapsed cross-state duplicates: {before:,} -> {len(out):,}")

    # Compute final ratio. Guard against zero/near-zero expected (would explode).
    out["revenue_ratio"] = np.where(
        out["expected_total_allowed"] > 0,
        out["actual_total_allowed"] / out["expected_total_allowed"],
        np.nan,
    ).astype("float32")

    # Storage casts
    out["year"] = out["year"].astype("int16")
    out["actual_total_allowed"]   = out["actual_total_allowed"].astype("float64")
    out["expected_total_allowed"] = out["expected_total_allowed"].astype("float64")
    out["n_rows_scored"]          = out["n_rows_scored"].astype("int32")

    out.to_parquet(args.output, index=False, compression="snappy")
    print(f"\nWrote {len(out):,} rows -> {args.output}  ({time.time() - t0:.1f}s)")

    # Summary
    valid = out.dropna(subset=["revenue_ratio"])
    print("\n-- Summary --")
    print(f"  Unique NPIs:           {out['Rndrng_NPI'].nunique():,}")
    print(f"  (NPI, year) rows:      {len(out):,}")
    print(f"  Ratio quantiles (NPI-years with expected > $0):")
    for q in [0.01, 0.05, 0.50, 0.95, 0.99, 0.999]:
        print(f"    p{q*100:6.2f}: {valid['revenue_ratio'].quantile(q):.3f}")
    n_above_130 = int((valid["revenue_ratio"] > 1.30).sum())
    n_above_volume = int(
        ((valid["revenue_ratio"] > 1.30) &
         (valid["expected_total_allowed"] >= 50_000)).sum()
    )
    print(f"  ratio > 1.30:                                {n_above_130:,}")
    print(f"  ratio > 1.30 AND expected >= $50K (trigger): {n_above_volume:,}")


if __name__ == "__main__":
    main()
