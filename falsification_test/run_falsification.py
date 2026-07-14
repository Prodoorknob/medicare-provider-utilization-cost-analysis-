"""
Falsification test: Stage 1 model vs official PFS vs empirical Avg_Mdcr_Alowd_Amt.

Sample 50 rows from silver. For each (HCPCS, state, year):
  - empirical = silver row's Avg_Mdcr_Alowd_Amt
  - official  = state-area-weighted PFS rate from PPRRVU + GPCI
  - model     = LightGBM no-charge prediction via the API feature pipeline

Decision:
  model ~= empirical, both != official  -> model captures aggregation effects (legit)
  model ~= official, both != empirical  -> model just learned the fee schedule (lookup wins)
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd

# Make the API helpers importable so we use the *exact* production feature pipeline.
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(ROOT, "api"))

from models.loader import load_all_models  # noqa: E402
from services.prediction import build_stage1_features, hcpcs_code_to_bucket  # noqa: E402
import lightgbm as lgb  # noqa: E402

ART_DIR = os.path.join(ROOT, "api", "models", "artifacts")
SILVER_DIR = os.path.join(ROOT, "local_pipeline", "silver")
RVU_DIR = os.path.join(ROOT, "falsification_test", "rvu_extracted")

YEARS = list(range(2013, 2024))
N_SAMPLE = 80          # oversample so we hit 50 evaluable rows after PFS-coverage filter
SEED = 42

# ----------------------------------------------------------------------
# 1. Parse PPRRVU (RVUs + status + conversion factor) per year
# ----------------------------------------------------------------------

PPRRVU_GLOBS = {y: glob.glob(os.path.join(RVU_DIR, str(y), "PPRRVU*.csv")) for y in YEARS}

PPRRVU_COLS = {
    "HCPCS": 0, "MOD": 1, "STATUS_CODE": 3,
    "WORK_RVU": 5, "PE_RVU_NONFAC": 6, "PE_RVU_FAC": 8,
    "MP_RVU": 10, "CF": 24,
}

def load_pprrvu(year: int) -> pd.DataFrame:
    """The PPRRVU file has a 2-row split header that pandas can't merge cleanly,
    so we parse by column position (positions are stable 2013-2023)."""
    files = PPRRVU_GLOBS[year]
    if not files:
        raise FileNotFoundError(f"No PPRRVU file for {year}")
    path = files[0]

    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("latin-1").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("HCPCS,MOD,DESCRIPTION") or line.startswith("HCPCS,MODIFIER,DESCRIPTION"):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError(f"Could not locate HCPCS header row in {path}")

    from io import StringIO
    body = "\n".join(lines[header_idx + 1:])  # skip the (split) header line
    df = pd.read_csv(StringIO(body), header=None, low_memory=False, dtype=str)
    out = pd.DataFrame({k: df.iloc[:, v] for k, v in PPRRVU_COLS.items()})
    out = out[out["HCPCS"].notna() & out["HCPCS"].str.match(r"^[A-Z0-9]{5}$", na=False)]
    for c in ("WORK_RVU", "PE_RVU_NONFAC", "PE_RVU_FAC", "MP_RVU", "CF"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["MOD"] = out["MOD"].fillna("").str.strip()
    # Prefer unmodified row (MOD = "") for the canonical PFS rate
    out = out.sort_values(["HCPCS", "MOD"]).drop_duplicates(subset=["HCPCS"], keep="first")
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------
# 2. Parse GPCI per year and aggregate to state-area-weighted (= simple mean across localities)
# ----------------------------------------------------------------------

GPCI_GLOBS = {y: glob.glob(os.path.join(RVU_DIR, str(y), "*GPCI*.csv")) + glob.glob(os.path.join(RVU_DIR, str(y), "CY*GPCI*.csv")) for y in YEARS}

FULL_TO_ABBR = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "DISTRICT OF COLUMBIA": "DC", "DC + MD/VA SUBURBS": "DC", "DC PLUS MD/VA SUBURBS": "DC",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID",
    "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
    "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT",
    "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI", "WYOMING": "WY",
}

import re

def _state_from_locality_name(name: str) -> str | None:
    """Extract 2-letter state from GPCI locality name.

    Examples:
      'Alabama' -> 'AL'
      'Anaheim/Santa Ana, CA' -> 'CA'
      'Rest of New York' -> 'NY'
      'Manhattan, NY' -> 'NY'
      'NYC SUBURBS/LONG ISLAND' -> 'NY'   (special-case)
      'POUGHKEEPSIE/N NYC SUBURBS' -> 'NY'
    """
    if not isinstance(name, str): return None
    s = name.strip().rstrip("*").strip()
    if not s: return None
    # Trailing ", XX"
    m = re.search(r",\s*([A-Z]{2})\s*$", s)
    if m: return m.group(1)
    # "Rest of <full state>"
    m = re.match(r"REST\s+OF\s+(.+)$", s, re.IGNORECASE)
    if m:
        sub = m.group(1).strip().upper()
        if sub in FULL_TO_ABBR: return FULL_TO_ABBR[sub]
    # Bare full state name
    if s.upper() in FULL_TO_ABBR: return FULL_TO_ABBR[s.upper()]
    # Hard-coded NY/NJ multi-county locality names lacking a comma
    upper = s.upper()
    if "NYC" in upper or "MANHATTAN" in upper or "QUEENS" in upper or "BRONX" in upper or "LONG ISLAND" in upper or "POUGHKEEPSIE" in upper:
        return "NY"
    if "NORTHERN NJ" in upper or "NORTHERN NEW JERSEY" in upper:
        return "NJ"
    return None


def load_gpci(year: int) -> pd.DataFrame:
    """Parse GPCI by column position (cols 0-5) — header text varies year to year.
    A row is a data row iff cols 3-5 parse as floats. State is extracted from the
    locality name (col 2)."""
    files = GPCI_GLOBS[year]
    if not files:
        raise FileNotFoundError(f"No GPCI file for {year}")
    path = files[0]

    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("latin-1").replace("\r\n", "\n").replace("\r", "\n")
    from io import StringIO
    df_raw = pd.read_csv(StringIO(text), header=None, dtype=str, low_memory=False)

    # 2021+ inserted a State column; detect layout by looking for a 2-letter
    # uppercase state abbr in column 1 of any data-looking row.
    has_state_col = False
    for _, row in df_raw.head(20).iterrows():
        v = str(row.iloc[1] if len(row) > 1 else "").strip()
        if re.fullmatch(r"[A-Z]{2}", v) and v in set(FULL_TO_ABBR.values()):
            has_state_col = True
            break

    if has_state_col:
        df = pd.DataFrame({
            "MAC": df_raw.iloc[:, 0],
            "STATE_DIRECT": df_raw.iloc[:, 1],
            "LOC_NUM": df_raw.iloc[:, 2],
            "LOC_NAME": df_raw.iloc[:, 3],
            "PW_GPCI": pd.to_numeric(df_raw.iloc[:, 4], errors="coerce"),
            "PE_GPCI": pd.to_numeric(df_raw.iloc[:, 5], errors="coerce"),
            "MP_GPCI": pd.to_numeric(df_raw.iloc[:, 6], errors="coerce"),
        })
    else:
        df = pd.DataFrame({
            "MAC": df_raw.iloc[:, 0],
            "LOC_NUM": df_raw.iloc[:, 1],
            "LOC_NAME": df_raw.iloc[:, 2],
            "PW_GPCI": pd.to_numeric(df_raw.iloc[:, 3], errors="coerce"),
            "PE_GPCI": pd.to_numeric(df_raw.iloc[:, 4], errors="coerce"),
            "MP_GPCI": pd.to_numeric(df_raw.iloc[:, 5], errors="coerce"),
        })
        df["STATE_DIRECT"] = None
    df = df.dropna(subset=["PW_GPCI", "PE_GPCI", "MP_GPCI"])
    df = df[(df["PW_GPCI"] > 0.3) & (df["PW_GPCI"] < 2.0)]  # GPCI sanity

    # Prefer the explicit STATE column when present; fall back to parsing locality name.
    parsed_state = df["LOC_NAME"].apply(_state_from_locality_name)
    df["STATE"] = df["STATE_DIRECT"].where(df["STATE_DIRECT"].notna(), parsed_state)
    df = df.dropna(subset=["STATE"])
    df["STATE"] = df["STATE"].str.strip().str.upper()
    df = df[df["STATE"].str.match(r"^[A-Z]{2}$")]
    # Simple unweighted mean across localities (state-area-weighted approximation)
    state_gpci = df.groupby("STATE")[["PW_GPCI", "PE_GPCI", "MP_GPCI"]].mean().reset_index()
    return state_gpci


# ----------------------------------------------------------------------
# 3. Build the joined PFS lookup: (year, HCPCS, state) -> official $ rate
# ----------------------------------------------------------------------

def compute_official_pfs(year: int, hcpcs: str, state: str,
                         pprrvu_by_year: dict, gpci_by_year: dict,
                         place_of_service: int) -> tuple[float | None, str]:
    """Return (official_dollar_rate, status_code).
    Uses non-facility PE RVU for office (POS=0), facility PE RVU for facility (POS=1).
    Returns None if HCPCS not found or status indicates non-PFS payment."""
    rvu_df = pprrvu_by_year[year]
    gpci_df = gpci_by_year[year]

    rvu = rvu_df[rvu_df["HCPCS"] == hcpcs]
    if len(rvu) == 0:
        return None, "NOT_IN_PFS"
    r = rvu.iloc[0]
    status = str(r.get("STATUS_CODE", "")).strip().upper()
    if status not in ("A", "R", "T"):
        # I=invalid, N=non-covered, X=other-fee-schedule, B=bundled, C=carrier-priced
        return None, f"STATUS_{status}"

    pe_rvu = r["PE_RVU_FAC"] if place_of_service == 1 else r["PE_RVU_NONFAC"]
    if pd.isna(pe_rvu): pe_rvu = 0.0
    work_rvu = r["WORK_RVU"] if not pd.isna(r["WORK_RVU"]) else 0.0
    mp_rvu = r["MP_RVU"] if not pd.isna(r["MP_RVU"]) else 0.0
    cf = r["CF"] if not pd.isna(r["CF"]) else None
    if cf is None:
        return None, "NO_CF"

    g = gpci_df[gpci_df["STATE"] == state]
    if len(g) == 0:
        return None, "NO_GPCI"
    pw_g = g["PW_GPCI"].iloc[0]
    pe_g = g["PE_GPCI"].iloc[0]
    mp_g = g["MP_GPCI"].iloc[0]

    rate = (work_rvu * pw_g + pe_rvu * pe_g + mp_rvu * mp_g) * cf
    return float(rate), status


# ----------------------------------------------------------------------
# 4. Sample silver rows
# ----------------------------------------------------------------------

def sample_silver(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    state_files = glob.glob(os.path.join(SILVER_DIR, "*.parquet"))
    territories = {"AA", "AE", "AP", "AS", "FM", "GU", "MP", "PR", "PW", "VI"}
    state_files = [f for f in state_files if os.path.basename(f).split(".")[0] not in territories]

    cols = ["Rndrng_NPI", "HCPCS_Cd", "year", "Rndrng_Prvdr_State_Abrvtn", "Rndrng_Prvdr_Zip5",
            "Rndrng_Prvdr_Type", "Place_Of_Srvc", "Tot_Srvcs", "Tot_Benes",
            "Avg_Sbmtd_Chrg", "Avg_Mdcr_Alowd_Amt"]

    rng_files = list(state_files)
    rng.shuffle(rng_files)
    chosen = []
    for f in rng_files[:15]:
        df = pd.read_parquet(f, columns=cols)
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df = df[df["year"].isin(YEARS)]
        df = df[df["Avg_Mdcr_Alowd_Amt"].notna() & (df["Avg_Mdcr_Alowd_Amt"] > 0)]
        df = df[df["Avg_Sbmtd_Chrg"].notna() & (df["Avg_Sbmtd_Chrg"] > 0)]
        df = df[df["Tot_Srvcs"].notna() & df["Tot_Benes"].notna()]
        df["HCPCS_Cd"] = df["HCPCS_Cd"].astype(str)
        df = df[df["HCPCS_Cd"].str.match(r"^[A-Z0-9]{5}$", na=False)]
        if len(df) == 0:
            continue
        # Take ~25 random rows per state, stratified over the years that exist
        per_state = min(25, max(11, n // 4))
        chosen.append(df.sample(n=min(per_state, len(df)),
                                random_state=int(rng.integers(0, 1_000_000))))
        if sum(len(c) for c in chosen) >= n * 3:
            break

    if not chosen:
        raise RuntimeError("Sampling produced no rows — check silver dir")
    pool = pd.concat(chosen, ignore_index=True)
    sampled = pool.sample(n=min(n, len(pool)), random_state=seed).reset_index(drop=True)
    return sampled


# ----------------------------------------------------------------------
# 5. Run model inference using API helper
# ----------------------------------------------------------------------

def predict_row(artifacts, row) -> float:
    pos_flag = 1 if str(row["Place_Of_Srvc"]).strip().upper() == "F" else 0
    feats = build_stage1_features(
        artifacts,
        provider_type=row["Rndrng_Prvdr_Type"],
        state=row["Rndrng_Prvdr_State_Abrvtn"],
        hcpcs_code=row["HCPCS_Cd"],
        hcpcs_bucket=hcpcs_code_to_bucket(row["HCPCS_Cd"]),
        place_of_service=pos_flag,
        risk_score=None,                              # default ~1.0 (median-imputed at training)
        total_services=float(row["Tot_Srvcs"]),
        total_beneficiaries=float(row["Tot_Benes"]),
        avg_submitted_charge=float(row["Avg_Sbmtd_Chrg"]),  # ignored by no-charge model
    )
    log_pred = artifacts.lgbm_booster.predict(feats)[0]
    return float(np.expm1(log_pred))


# ----------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------

def main():
    print("Loading model artifacts...")
    artifacts = load_all_models(ART_DIR)
    if not artifacts.stage1_ready:
        raise RuntimeError("Stage 1 not loaded — check artifacts dir")

    print("Loading PPRRVU + GPCI tables for all 11 years...")
    pprrvu_by_year = {}
    gpci_by_year = {}
    for y in YEARS:
        pprrvu_by_year[y] = load_pprrvu(y)
        gpci_by_year[y] = load_gpci(y)
        print(f"  {y}: PPRRVU rows={len(pprrvu_by_year[y]):>6}  GPCI states={len(gpci_by_year[y])}")

    print(f"\nSampling {N_SAMPLE} silver rows (seed={SEED})...")
    sampled = sample_silver(N_SAMPLE, SEED)
    print(f"  {len(sampled)} candidates drawn")

    # Score every candidate
    rows_out = []
    for _, r in sampled.iterrows():
        pos_flag = 1 if str(r["Place_Of_Srvc"]).strip().upper() == "F" else 0
        official, status = compute_official_pfs(
            int(r["year"]), r["HCPCS_Cd"], r["Rndrng_Prvdr_State_Abrvtn"],
            pprrvu_by_year, gpci_by_year, pos_flag,
        )
        try:
            pred = predict_row(artifacts, r)
        except Exception as e:
            pred = None

        rows_out.append({
            "year": int(r["year"]),
            "state": r["Rndrng_Prvdr_State_Abrvtn"],
            "zip5": r["Rndrng_Prvdr_Zip5"],
            "hcpcs": r["HCPCS_Cd"],
            "specialty": r["Rndrng_Prvdr_Type"],
            "pos": "F" if pos_flag == 1 else "O",
            "tot_srvcs": float(r["Tot_Srvcs"]),
            "tot_benes": float(r["Tot_Benes"]),
            "submitted_charge": float(r["Avg_Sbmtd_Chrg"]),
            "empirical": float(r["Avg_Mdcr_Alowd_Amt"]),
            "official_pfs": official,
            "model_pred": pred,
            "pfs_status": status,
        })

    df = pd.DataFrame(rows_out)
    df["evaluable"] = df["official_pfs"].notna() & df["model_pred"].notna()

    print(f"\nCandidates: {len(df)} | evaluable (PFS-active): {df['evaluable'].sum()}")
    eval_df = df[df["evaluable"]].copy()
    if len(eval_df) > 50:
        eval_df = eval_df.sample(n=50, random_state=SEED).reset_index(drop=True)
    eval_df["err_emp_off"] = eval_df["empirical"] - eval_df["official_pfs"]
    eval_df["err_mod_off"] = eval_df["model_pred"] - eval_df["official_pfs"]
    eval_df["err_mod_emp"] = eval_df["model_pred"] - eval_df["empirical"]

    # Save raw + summary
    out_csv = os.path.join(ROOT, "falsification_test", "results.csv")
    eval_df.to_csv(out_csv, index=False)
    df.to_csv(os.path.join(ROOT, "falsification_test", "all_candidates.csv"), index=False)

    print("\n=== EVALUABLE ROWS (n={}) ===".format(len(eval_df)))
    print(eval_df[["year", "state", "hcpcs", "specialty", "pos",
                   "empirical", "official_pfs", "model_pred"]].to_string(index=False))

    def summarize(name, errs):
        print(f"\n  {name}:")
        print(f"    MAE   = ${errs.abs().mean():>8.2f}")
        print(f"    bias  = ${errs.mean():>+8.2f}")
        print(f"    P50|err| = ${errs.abs().median():>6.2f}")
        print(f"    P90|err| = ${errs.abs().quantile(0.9):>6.2f}")

    print("\n=== PAIRWISE COMPARISONS ===")
    summarize("empirical vs official", eval_df["err_emp_off"])
    summarize("model     vs official", eval_df["err_mod_off"])
    summarize("model     vs empirical", eval_df["err_mod_emp"])

    print("\n=== CORRELATIONS (Pearson) ===")
    print(f"  empirical vs official : {eval_df['empirical'].corr(eval_df['official_pfs']):.4f}")
    print(f"  model     vs official : {eval_df['model_pred'].corr(eval_df['official_pfs']):.4f}")
    print(f"  model     vs empirical: {eval_df['model_pred'].corr(eval_df['empirical']):.4f}")

    # Verdict
    mae_mo = eval_df["err_mod_off"].abs().mean()
    mae_me = eval_df["err_mod_emp"].abs().mean()
    mae_eo = eval_df["err_emp_off"].abs().mean()

    print("\n=== VERDICT ===")
    print(f"  MAE(model, empirical) = ${mae_me:.2f}")
    print(f"  MAE(model, official)  = ${mae_mo:.2f}")
    print(f"  MAE(empirical, official) = ${mae_eo:.2f}")
    if mae_eo < 1.0:
        print("\n  empirical ~= official (gap ${:.2f}). The training target IS the fee schedule —".format(mae_eo))
        print("  there's nothing for the model to add. Lookup wins by parsimony.")
    elif mae_me < mae_mo and mae_me < 0.6 * mae_eo:
        print("\n  model ~= empirical, both != official. Model captures aggregation effects")
        print("  (modifiers, multi-procedure discounts, etc.) that a flat PFS lookup misses.")
    elif mae_mo < mae_me and mae_mo < 0.6 * mae_eo:
        print("\n  model ~= official, model != empirical. Model collapsed onto the fee schedule;")
        print("  no NPI-conditional signal beyond what a lookup gives you.")
    else:
        print("\n  Mixed: model sits between official and empirical. No clear winner — see distribution.")

    print(f"\nResults saved to: {out_csv}")


if __name__ == "__main__":
    main()
