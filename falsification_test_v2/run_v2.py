"""
Stage 1 falsification test v2: corrected methodology.

Changes vs v1:
  1. Source of truth: live PFS Look-Up Tool's underlying CSVs (pfs.data.cms.gov),
     not the formula-derived PPRRVU + GPCI computation in v1.
  2. Modifier handling: for radiologist NPIs billing imaging codes with PCTC=1,
     pull the MOD-26 (professional component) row, not the unmodified Global row.
     This is the central correction.
  3. Per-locality resolution: state -> list of localities, compute price per
     locality, then average across the state. Both Option B (state-mean) and
     Option A (per-locality range) are reported.

Output: falsification_test_v2/{results_v2.csv, REPORT_v2.md, scatter_v2.png, methodology_diff.md}
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "falsification_test" / "data"
V2_DIR = ROOT / "falsification_test_v2"
V1_RESULTS = ROOT / "falsification_test" / "results.csv"

# Year -> (indicators_csv, localities_csv) per the user's mapping. 2024A/B and
# 2025 are kept available for the optional forecast extension; 2024B is the
# final 2024 release.
YEAR_FILES = {
    2013: ("indicators2013.csv", "localities2013.csv"),
    2014: ("indicators2014.csv", "localities2014.csv"),
    2015: ("indicators215b.csv", "localities215b.csv"),
    2016: ("indicators2016.csv", "localities2016.csv"),
    2017: ("indicators2017.csv", "localities2017.csv"),
    2018: ("indicators2018.csv", "localities2018.csv"),
    2019: ("indicators2019.csv", "localities2019.csv"),
    2020: ("indicators2020.csv", "localities2020.csv"),
    2021: ("indicators2021-10-21-2021.csv", "localities2021.csv"),
    2022: ("indicators2022-09-12.csv", "localities2022--02-08-22.csv"),
    2023: ("indicators2023-09-28.csv", "localities2023-06-20.csv"),
    2024: ("indicators2024B-09-18-2024.csv", "localities2024B.csv"),
    2025: ("indicators2025-09-23-2025.csv", "localities2025.csv"),
}

# mac_description -> 2-letter state abbreviation. "55 distinct values" verified
# stable 2013 through 2023. CA splits into NORTHERN/SOUTHERN, NY into 3 MACs,
# HI shares a MAC with the Pacific territories, PR shares with VI.
MAC_TO_STATE = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "DISTRICT OF COLUMBIA": "DC",
    "EMPIRE NEW YORK": "NY", "GHI/NEW YORK": "NY", "WESTERN NEW YORK": "NY",
    "FLORIDA": "FL", "GEORGIA": "GA",
    "HAWAII/GUAM/AMER SAMOA/N MARIANA": "HI",
    "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA",
    "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA",
    "MAINE": "ME", "MARYLAND": "MD", "MASSACHUSETTS": "MA",
    "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
    "MISSOURI": "MO", "MONTANA": "MT",
    "NEBRASKA": "NE", "NEVADA": "NV", "NEW HAMPSHIRE": "NH",
    "NEW JERSEY": "NJ", "NEW MEXICO": "NM",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND",
    "NORTHERN CALIFORNIA": "CA", "SOUTHERN CALIFORNIA": "CA",
    "OHIO": "OH", "OKLAHOMA": "OK", "OREGON": "OR",
    "PENNSYLVANIA": "PA",
    "PUERTO RICO/VIRGIN ISLANDS": "PR",
    "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN", "TEXAS": "TX",
    "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA",
    "WASHINGTON": "WA", "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI", "WYOMING": "WY",
}


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_files() -> None:
    missing = []
    for y, (ind, loc) in YEAR_FILES.items():
        if not (DATA_DIR / ind).exists(): missing.append((y, "indicators", ind))
        if not (DATA_DIR / loc).exists(): missing.append((y, "localities", loc))
    if missing:
        for y, t, f in missing: print(f"  MISSING {y} {t}: {f}")
        raise SystemExit(f"Missing {len(missing)} files in {DATA_DIR}")
    print(f"Verified {len(YEAR_FILES)} year pairs ({len(YEAR_FILES)*2} CSVs) in {DATA_DIR}")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

_IND_CACHE: dict[int, pd.DataFrame] = {}
_LOC_CACHE: dict[int, pd.DataFrame] = {}


def load_indicators(year: int) -> pd.DataFrame:
    if year in _IND_CACHE: return _IND_CACHE[year]
    fn = YEAR_FILES[year][0]
    df = pd.read_csv(DATA_DIR / fn, low_memory=False, encoding="latin-1")
    df["hcpc"] = df["hcpc"].astype(str).str.strip()
    df["modifier"] = df["modifier"].astype(str).str.strip()
    df.loc[df["modifier"].isin(["nan", "NaN", ""]), "modifier"] = ""
    for c in ("rvu_work", "trans_nfac_pe", "trans_fac_pe", "rvu_mp", "work_adjustor", "conv_fact", "pctc"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    _IND_CACHE[year] = df
    return df


def load_localities(year: int) -> pd.DataFrame:
    if year in _LOC_CACHE: return _LOC_CACHE[year]
    fn = YEAR_FILES[year][1]
    df = pd.read_csv(DATA_DIR / fn, low_memory=False, encoding="latin-1")
    for c in ("gpci_work", "gpci_pe", "gpci_mp"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["locality"] != 0].copy()
    df["state"] = df["mac_description"].map(MAC_TO_STATE)
    unmapped = df[df["state"].isna()]["mac_description"].unique()
    if len(unmapped) > 0:
        print(f"  WARN {year}: unmapped MACs: {list(unmapped)}")
    _LOC_CACHE[year] = df
    return df


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------

def price_per_locality(ind_row: pd.Series, loc_df: pd.DataFrame, pos_facility: bool) -> pd.DataFrame:
    """Apply the PFS pricing formula across all localities. Returns columns
    [locality, loc_description, state, price]."""
    work_adj = ind_row["work_adjustor"] if pd.notna(ind_row["work_adjustor"]) else 1.0
    rvu_w = ind_row["rvu_work"] if pd.notna(ind_row["rvu_work"]) else 0.0
    rvu_pe = (ind_row["trans_fac_pe"] if pos_facility else ind_row["trans_nfac_pe"])
    rvu_pe = rvu_pe if pd.notna(rvu_pe) else 0.0
    rvu_mp = ind_row["rvu_mp"] if pd.notna(ind_row["rvu_mp"]) else 0.0
    cf = ind_row["conv_fact"]

    out = loc_df[["locality", "loc_description", "state", "gpci_work", "gpci_pe", "gpci_mp"]].copy()
    out["price"] = (
        (rvu_w * work_adj * out["gpci_work"]) +
        (rvu_pe * out["gpci_pe"]) +
        (rvu_mp * out["gpci_mp"])
    ) * cf
    return out[["locality", "loc_description", "state", "price"]]


def lookup_indicator(ind_df: pd.DataFrame, hcpcs: str, modifier: str = "") -> pd.Series | None:
    """Find the (HCPCS, modifier) row. Returns None if missing or non-payable status."""
    hcpcs = hcpcs.strip()
    modifier = modifier.strip()
    sub = ind_df[(ind_df["hcpc"] == hcpcs) & (ind_df["modifier"] == modifier)]
    if len(sub) == 0:
        return None
    row = sub.iloc[0]
    if str(row.get("proc_stat", "")).strip().upper() not in ("A", "R", "T"):
        return None
    return row


def state_price(year: int, hcpcs: str, modifier: str, state: str, pos_facility: bool) -> dict:
    """Compute per-locality prices for (hcpcs, modifier) in the given state.
    Returns dict with keys: ok, status, mean, min, max, n_localities, mod_used."""
    ind_df = load_indicators(year)
    loc_df = load_localities(year)
    state_loc = loc_df[loc_df["state"] == state]
    if len(state_loc) == 0:
        return {"ok": False, "status": f"NO_LOCALITIES_FOR_STATE_{state}"}

    ind_row = lookup_indicator(ind_df, hcpcs, modifier)
    if ind_row is None:
        # Try to see if the row exists with a different status (informative)
        any_match = ind_df[(ind_df["hcpc"] == hcpcs) & (ind_df["modifier"] == modifier)]
        st = (str(any_match["proc_stat"].iloc[0]).strip() if len(any_match) else "MISSING")
        return {"ok": False, "status": f"INACTIVE_OR_MISSING_{st}", "modifier": modifier}

    prices = price_per_locality(ind_row, state_loc, pos_facility)
    return {
        "ok": True,
        "status": "OK",
        "mean": float(prices["price"].mean()),
        "min": float(prices["price"].min()),
        "max": float(prices["price"].max()),
        "n_localities": int(len(prices)),
        "mod_used": modifier or "(global)",
    }


# --------------------------------------------------------------------------
# Modifier policy
# --------------------------------------------------------------------------

def choose_modifier(specialty: str, hcpcs: str, year: int) -> tuple[str, str]:
    """Return (modifier, reason). Modifier is "" for global, "26" for PC.

    Policy:
      - If specialty is Diagnostic Radiology AND HCPCS has PCTC=1 -> "26"
      - Else -> "" (global)
      - "Mass Immunizer" and "Independent Diagnostic Testing Facility" carry the
        TC component but those specialties were not in the v1 sample.

    Reason is returned for audit columns.
    """
    ind_df = load_indicators(year)
    sub = ind_df[ind_df["hcpc"] == hcpcs.strip()]
    pctc_vals = sub["pctc"].dropna().unique()
    has_pctc = (1 in pctc_vals) or (1.0 in pctc_vals)

    spec = (specialty or "").strip().lower()
    if has_pctc and "diagnostic radiology" in spec:
        return "26", "RADIOLOGY_PCTC1_MOD26"
    if has_pctc:
        return "", "PCTC1_NON_RADIOLOGY_DEFAULT_GLOBAL_REVIEW"
    return "", "NO_PCTC_GLOBAL"


# --------------------------------------------------------------------------
# State to localities resolver: unit tests
# --------------------------------------------------------------------------

def unit_test_state_resolver():
    print("\n=== State to localities resolver unit tests (year 2023) ===")
    loc23 = load_localities(2023)
    expectations = {
        "CA": (32, ["LOS ANGELES", "SAN FRANCISCO", "ANAHEIM", "REST OF CALIFORNIA"]),
        "NY": (5,  ["MANHATTAN", "QUEENS", "REST OF NEW YORK"]),
        "FL": (3,  ["MIAMI", "FORT LAUDERDALE", "REST OF FLORIDA"]),
        "IL": (4,  ["CHICAGO", "REST OF ILLINOIS"]),
        "TX": (8,  ["HOUSTON", "DALLAS", "AUSTIN"]),
    }
    for st, (expected_n_min, expected_substrs) in expectations.items():
        rows = loc23[loc23["state"] == st]
        ok_n = len(rows) >= expected_n_min - 1  # allow off-by-one for year drift
        descs = " | ".join(rows["loc_description"].astype(str).str.upper())
        hits = sum(1 for s in expected_substrs if s in descs)
        passed = ok_n and hits >= max(2, len(expected_substrs) - 1)
        print(f"  {st}: {len(rows)} localities (>= {expected_n_min - 1}? {ok_n}); "
              f"substr hits {hits}/{len(expected_substrs)}: {'PASS' if passed else 'FAIL'}")


# --------------------------------------------------------------------------
# 2013 Suburban Chicago 70549 MOD-26 spot check (user's reproducibility anchor)
# --------------------------------------------------------------------------

def spot_check_2013():
    print("\n=== Spot check: 2013 70549 MOD-26 Suburban Chicago ===")
    ind = load_indicators(2013)
    loc = load_localities(2013)
    ind_row = lookup_indicator(ind, "70549", "26")
    sub_chi = loc[loc["loc_description"].astype(str).str.contains("Suburban Chicago", case=False, na=False)]
    if len(sub_chi) == 0 or ind_row is None:
        print("  FAIL: row not found"); return
    prices = price_per_locality(ind_row, sub_chi, pos_facility=True)
    p = prices["price"].iloc[0]
    print(f"  Computed: ${p:.2f}  (expected ~$92.28)")
    if abs(p - 92.28) > 0.05:
        print("  FAIL: mismatch with user's spot check")
    else:
        print("  PASS")


# --------------------------------------------------------------------------
# Model loader: reuse from v1
# --------------------------------------------------------------------------

def load_model_artifacts():
    sys.path.insert(0, str(ROOT / "api"))
    from models.loader import load_all_models
    return load_all_models(str(ROOT / "api" / "models" / "artifacts"))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    validate_files()
    print(f"\nLoading 2023 smoke test...")
    ind23 = load_indicators(2023)
    loc23 = load_localities(2023)
    print(f"  indicators2023: {len(ind23)} rows, {ind23['hcpc'].nunique()} unique HCPCS")
    print(f"  localities2023: {len(loc23)} rows, {loc23['state'].nunique()} unique states")

    spot_check_2013()
    unit_test_state_resolver()

    print(f"\nLoading v1 results from {V1_RESULTS}")
    v1 = pd.read_csv(V1_RESULTS)
    print(f"  v1 rows: {len(v1)}  columns: {list(v1.columns)}")

    out = []
    for i, r in v1.iterrows():
        year = int(r["year"])
        state = str(r["state"]).strip()
        hcpcs = str(r["hcpcs"]).strip()
        spec = str(r["specialty"])
        pos_fac = (str(r["pos"]).strip().upper() == "F")

        # Option B: state-mean Global rate (replicates v1 methodology under new data source)
        global_res = state_price(year, hcpcs, "", state, pos_fac)

        # Modifier-aware: choose modifier per policy
        chosen_mod, reason = choose_modifier(spec, hcpcs, year)
        modaware_res = state_price(year, hcpcs, chosen_mod, state, pos_fac)

        # If modifier-aware lookup fell through (e.g., MOD-26 row missing), fall
        # back to global so we still get a number to compare against.
        if not modaware_res["ok"] and chosen_mod != "":
            fallback = state_price(year, hcpcs, "", state, pos_fac)
            if fallback["ok"]:
                modaware_res = fallback
                reason += "_FALLBACK_TO_GLOBAL"

        row_out = dict(r)
        row_out["official_global_v2"] = global_res["mean"] if global_res["ok"] else np.nan
        row_out["official_global_v2_min"] = global_res.get("min", np.nan) if global_res["ok"] else np.nan
        row_out["official_global_v2_max"] = global_res.get("max", np.nan) if global_res["ok"] else np.nan
        row_out["official_global_v2_n"] = global_res.get("n_localities", 0) if global_res["ok"] else 0
        row_out["official_global_v2_status"] = global_res["status"]

        row_out["official_modifier_aware"] = modaware_res["mean"] if modaware_res["ok"] else np.nan
        row_out["official_modifier_aware_min"] = modaware_res.get("min", np.nan) if modaware_res["ok"] else np.nan
        row_out["official_modifier_aware_max"] = modaware_res.get("max", np.nan) if modaware_res["ok"] else np.nan
        row_out["applied_modifier"] = chosen_mod or "(global)"
        row_out["modifier_reason"] = reason
        row_out["modaware_status"] = modaware_res["status"]

        # Convenience errors for scoring
        emp = r["empirical"]
        mp = r["model_pred"]
        og = row_out["official_global_v2"]
        oma = row_out["official_modifier_aware"]
        row_out["err_model_vs_official_global"] = (mp - og) if pd.notna(og) else np.nan
        row_out["err_model_vs_official_modifier_aware"] = (mp - oma) if pd.notna(oma) else np.nan
        row_out["err_emp_vs_official_modifier_aware"] = (emp - oma) if pd.notna(oma) else np.nan
        row_out["empirical_in_state_range"] = (
            (pd.notna(oma) and pd.notna(modaware_res.get("min")) and pd.notna(modaware_res.get("max"))
             and modaware_res["min"] <= emp <= modaware_res["max"])
        )

        out.append(row_out)
        if (i + 1) % 10 == 0:
            print(f"  row {i+1}/{len(v1)}: {year} {state} {hcpcs} {spec} "
                  f"mod={chosen_mod or 'GBL'} global=${og or 'NA'} mod_aware=${oma or 'NA'} "
                  f"emp=${emp} model=${mp:.2f}")

    df = pd.DataFrame(out)
    out_csv = V2_DIR / "results_v2.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    print(f"  rows: {len(df)}  modaware coverage: {df['official_modifier_aware'].notna().sum()}/{len(df)}")

    # Headline numbers
    print("\n=== Headline (n_evaluable_modaware = {}) ===".format(df["official_modifier_aware"].notna().sum()))
    eval_df = df[df["official_modifier_aware"].notna() & df["official_global_v2"].notna()].copy()
    eval_df["err_mod_emp"] = eval_df["model_pred"] - eval_df["empirical"]

    def stats(name, errs):
        print(f"  {name}:")
        print(f"    n={len(errs)}  MAE=${errs.abs().mean():.2f}  bias=${errs.mean():+.2f}  "
              f"P50|err|=${errs.abs().median():.2f}  P90|err|=${errs.abs().quantile(0.9):.2f}")

    stats("model vs official_global_v2 (Option B)", eval_df["err_model_vs_official_global"])
    stats("model vs official_modifier_aware",        eval_df["err_model_vs_official_modifier_aware"])
    stats("empirical vs official_modifier_aware",    eval_df["err_emp_vs_official_modifier_aware"])
    stats("model vs empirical (sanity, unchanged)",  eval_df["err_mod_emp"])

    print("\n  Pearson correlations:")
    print(f"    model vs official_global_v2     : {eval_df['model_pred'].corr(eval_df['official_global_v2']):.4f}")
    print(f"    model vs official_modifier_aware: {eval_df['model_pred'].corr(eval_df['official_modifier_aware']):.4f}")
    print(f"    empirical vs modifier_aware     : {eval_df['empirical'].corr(eval_df['official_modifier_aware']):.4f}")

    return df


if __name__ == "__main__":
    df = main()
