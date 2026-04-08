"""
generate_synthetic_mcbs.py — Synthetic Per-Service OOP Data Generator

Generates statistically plausible per-service out-of-pocket cost records
WITH Census region, mimicking what the MCBS Limited Data Set (LDS) would
provide. Enables the full per-service Stage 2 OOP prediction pipeline.

    *** THIS PRODUCES SYNTHETIC DATA — NOT REAL PATIENT RECORDS ***

    All synthetic values are derived from real distributions:
      - Beneficiary demographics sampled from real MCBS Cost Supplement
      - Service-level data drawn from real provider gold parquets
      - OOP amounts modeled from real national OOP-to-allowed ratios

    Users with access to the actual MCBS LDS ($600/module, DUA required)
    can replace this output with real per-service data and run the same
    Stage 2 pipeline unchanged.

Data sources:
  - local_pipeline/mcbs_silver/{YEAR}.parquet  (real OOP distributions)
  - local_pipeline/gold/{STATE}.parquet        (real provider data)
  - local_pipeline/gold/label_encoders.json    (state→region mapping)

Output:
  - local_pipeline/mcbs_synthetic/synthetic_oop.parquet

Usage:
    python generate_synthetic_mcbs.py
    python generate_synthetic_mcbs.py --sample 0.1 --seed 42
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd

_PROJECT_ROOT     = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GOLD      = os.path.join(_PROJECT_ROOT, "local_pipeline", "gold")
DEFAULT_MCBS      = os.path.join(_PROJECT_ROOT, "local_pipeline", "mcbs_silver")
DEFAULT_OUTPUT    = os.path.join(_PROJECT_ROOT, "local_pipeline", "mcbs_synthetic")
DEFAULT_ENCODERS  = os.path.join(DEFAULT_GOLD, "label_encoders.json")

CENSUS_REGIONS = {
    "NORTHEAST": ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
    "SOUTH":     ["DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV",
                  "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"],
    "MIDWEST":   ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO",
                  "NE", "ND", "SD"],
    "WEST":      ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY",
                  "AK", "CA", "HI", "OR", "WA"],
}
REGION_NAME_TO_INT = {"NORTHEAST": 1, "MIDWEST": 2, "SOUTH": 3, "WEST": 4}

# Provider gold columns to carry forward as Stage 2 features
PROVIDER_COLS = [
    "year", "Rndrng_Prvdr_Type_idx", "Rndrng_Prvdr_State_Abrvtn_idx",
    "hcpcs_bucket", "place_of_srvc_flag", "Avg_Mdcr_Alowd_Amt",
    "Bene_Avg_Risk_Scre",
]


def build_state_region_map(label_encoders: dict) -> dict[int, int]:
    """Map state_idx -> census_region_int from label encoders."""
    state_classes = label_encoders.get("Rndrng_Prvdr_State_Abrvtn", [])
    state_to_region = {}
    for region_name, states in CENSUS_REGIONS.items():
        region_int = REGION_NAME_TO_INT[region_name]
        for st in states:
            state_to_region[st.upper()] = region_int

    idx_to_region = {}
    for idx, abbrev in enumerate(state_classes):
        r = state_to_region.get(abbrev.upper())
        if r is not None:
            idx_to_region[idx] = r
    return idx_to_region


def extract_mcbs_distributions(mcbs_dir: str) -> dict:
    """Extract real demographic & OOP distributions from MCBS Silver by year."""
    mcbs_files = sorted(glob.glob(os.path.join(mcbs_dir, "*.parquet")))
    if not mcbs_files:
        raise FileNotFoundError(f"No MCBS Silver files in {mcbs_dir}. Run the MCBS pipeline first.")

    yearly_stats = {}
    for f in mcbs_files:
        df = pd.read_parquet(f)
        year = int(df["year"].iloc[0]) if "year" in df.columns else int(os.path.basename(f).replace(".parquet", ""))

        stats = {"n_benes": len(df)}

        # OOP share: what fraction of allowed amount becomes OOP
        if "oop_share" in df.columns:
            oop_share = df["oop_share"].dropna()
            stats["oop_share_mean"] = float(oop_share.mean())
            stats["oop_share_std"]  = float(oop_share.std())
        elif "pay_oop" in df.columns and "exp_total" in df.columns:
            mask = df["exp_total"] > 0
            share = df.loc[mask, "pay_oop"] / df.loc[mask, "exp_total"]
            stats["oop_share_mean"] = float(share.mean())
            stats["oop_share_std"]  = float(share.std())
        else:
            stats["oop_share_mean"] = 0.20  # fallback national average
            stats["oop_share_std"]  = 0.15

        # Demographic distributions (for sampling)
        for col, key in [("age", "age"), ("sex", "sex"), ("income", "income"),
                         ("chronic_count", "chronic_count")]:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(vals) > 0:
                    stats[f"{key}_values"] = vals.values.astype(float)

        # Binary rates
        for col, key in [("has_medicaid", "medicaid_rate"), ("has_private_ins", "supplemental_rate")]:
            if col in df.columns:
                stats[key] = float(pd.to_numeric(df[col], errors="coerce").mean())

        yearly_stats[year] = stats
        print(f"  {year}: {len(df):,} benes, OOP share={stats['oop_share_mean']:.3f}")

    return yearly_stats


def load_provider_sample(gold_dir: str, sample_frac: float, seed: int) -> pd.DataFrame:
    """Load a sample of provider gold data."""
    gold_files = sorted(glob.glob(os.path.join(gold_dir, "*.parquet")))
    gold_files = [f for f in gold_files if not f.endswith("label_encoders.json")]
    if not gold_files:
        raise FileNotFoundError(f"No gold parquets in {gold_dir}")

    print(f"  Loading {len(gold_files)} gold state parquets (sample={sample_frac})...")
    chunks = []
    rng = np.random.RandomState(seed)
    for i, f in enumerate(gold_files, 1):
        try:
            df = pd.read_parquet(f, columns=PROVIDER_COLS).dropna()
            if sample_frac < 1.0:
                df = df.sample(frac=sample_frac, random_state=rng)
            chunks.append(df)
        except Exception as e:
            print(f"    [WARN] Skipping {os.path.basename(f)}: {e}")
        if i % 10 == 0:
            print(f"    {i}/{len(gold_files)} files...")

    combined = pd.concat(chunks, ignore_index=True)
    print(f"  Provider sample: {len(combined):,} rows")
    return combined


def generate(gold_dir: str, mcbs_dir: str, encoders_path: str,
             output_dir: str, sample_frac: float, seed: int):
    """Generate synthetic per-service OOP records."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    print("=== Synthetic MCBS LDS Generator ===\n")

    # 1. Load label encoders for state→region mapping
    print("[1/4] Loading label encoders...")
    with open(encoders_path) as f:
        label_encoders = json.load(f)
    idx_to_region = build_state_region_map(label_encoders)
    print(f"  Mapped {len(idx_to_region)} state indices to 4 Census regions\n")

    # 2. Extract real MCBS distributions
    print("[2/4] Extracting MCBS distributions...")
    yearly_stats = extract_mcbs_distributions(mcbs_dir)
    mcbs_years = sorted(yearly_stats.keys())
    print(f"  MCBS years: {mcbs_years}\n")

    # 3. Load provider sample
    print("[3/4] Loading provider gold sample...")
    provider = load_provider_sample(gold_dir, sample_frac, seed)

    # Map state to census region
    provider["census_region"] = provider["Rndrng_Prvdr_State_Abrvtn_idx"].astype(int).map(idx_to_region)
    before = len(provider)
    provider = provider.dropna(subset=["census_region"])
    provider["census_region"] = provider["census_region"].astype(int)
    print(f"  After region mapping: {len(provider):,} rows ({before - len(provider)} territories dropped)\n")

    # 4. Synthesize beneficiary demographics + per-service OOP
    print("[4/4] Generating synthetic beneficiary records...")
    n = len(provider)

    # For each row, find the nearest MCBS year for distribution sampling
    provider_years = provider["year"].astype(int).values
    nearest_mcbs = np.array([min(mcbs_years, key=lambda y: abs(y - py)) for py in provider_years])

    # Pre-allocate arrays
    syn_age      = np.zeros(n, dtype=np.int16)
    syn_sex      = np.zeros(n, dtype=np.int16)
    syn_income   = np.zeros(n, dtype=np.int16)
    syn_chronic  = np.zeros(n, dtype=np.int16)
    syn_dual     = np.zeros(n, dtype=np.int16)
    syn_suppl    = np.zeros(n, dtype=np.int16)
    syn_oop      = np.zeros(n, dtype=np.float64)

    # Process by MCBS year cohort for efficiency
    for mcbs_year in mcbs_years:
        mask = nearest_mcbs == mcbs_year
        count = mask.sum()
        if count == 0:
            continue

        stats = yearly_stats[mcbs_year]

        # Sample age
        if "age_values" in stats:
            syn_age[mask] = rng.choice(stats["age_values"], size=count).astype(np.int16)
        else:
            syn_age[mask] = rng.normal(75, 8, size=count).clip(18, 105).astype(np.int16)

        # Sample sex (1=M, 2=F, roughly 44/56 split in Medicare)
        if "sex_values" in stats:
            syn_sex[mask] = rng.choice(stats["sex_values"], size=count).astype(np.int16)
        else:
            syn_sex[mask] = rng.choice([1, 2], size=count, p=[0.44, 0.56]).astype(np.int16)

        # Sample income
        if "income_values" in stats:
            syn_income[mask] = rng.choice(stats["income_values"], size=count).astype(np.int16)
        else:
            syn_income[mask] = rng.choice([1, 2, 3, 4, 5], size=count,
                                          p=[0.25, 0.25, 0.20, 0.15, 0.15]).astype(np.int16)

        # Sample chronic count
        if "chronic_count_values" in stats:
            syn_chronic[mask] = rng.choice(stats["chronic_count_values"], size=count).astype(np.int16)
        else:
            syn_chronic[mask] = rng.poisson(2.5, size=count).clip(0, 10).astype(np.int16)

        # Sample dual eligible
        medicaid_rate = stats.get("medicaid_rate", 0.20)
        syn_dual[mask] = rng.binomial(1, medicaid_rate, size=count).astype(np.int16)

        # Sample supplemental insurance
        suppl_rate = stats.get("supplemental_rate", 0.35)
        syn_suppl[mask] = rng.binomial(1, suppl_rate, size=count).astype(np.int16)

        # Compute per-service OOP
        allowed = provider.loc[mask, "Avg_Mdcr_Alowd_Amt"].values
        oop_share_mean = stats["oop_share_mean"]
        oop_share_std  = stats["oop_share_std"]

        # Base OOP: allowed × sampled share
        shares = rng.normal(oop_share_mean, max(oop_share_std, 0.01), size=count)
        shares = np.clip(shares, 0.0, 1.0)
        base_oop = allowed * shares

        # Demographic modulation
        dual_factor = np.where(syn_dual[mask] == 1, 0.15, 1.0)     # Medicaid covers ~85% of OOP
        suppl_factor = np.where(syn_suppl[mask] == 1, 0.65, 1.0)   # supplemental covers ~35%
        chronic_factor = 1.0 + (syn_chronic[mask] - 2.5) * 0.03    # slight increase per condition
        chronic_factor = np.clip(chronic_factor, 0.8, 1.3)

        modulated_oop = base_oop * dual_factor * suppl_factor * chronic_factor

        # Add log-normal noise for realism
        noise = rng.lognormal(0, 0.3, size=count)
        noisy_oop = modulated_oop * noise

        # Clip: OOP can't be negative or exceed allowed amount
        syn_oop[mask] = np.clip(noisy_oop, 0.0, allowed)

    # Build output DataFrame
    result = pd.DataFrame({
        "year":                     provider["year"].values.astype(np.int16),
        "census_region":            provider["census_region"].values.astype(np.int16),
        "Rndrng_Prvdr_Type_idx":    provider["Rndrng_Prvdr_Type_idx"].values,
        "hcpcs_bucket":             provider["hcpcs_bucket"].values,
        "place_of_srvc_flag":       provider["place_of_srvc_flag"].values,
        "Avg_Mdcr_Alowd_Amt":      provider["Avg_Mdcr_Alowd_Amt"].values,
        "Bene_Avg_Risk_Scre":       provider["Bene_Avg_Risk_Scre"].values,
        "age":                      syn_age,
        "sex":                      syn_sex,
        "income":                   syn_income,
        "chronic_count":            syn_chronic,
        "dual_eligible":            syn_dual,
        "has_supplemental":         syn_suppl,
        "per_service_oop":          syn_oop,
    })

    # Derive age_band
    def age_band(a):
        if a < 65:  return "under_65"
        if a < 70:  return "65-69"
        if a < 75:  return "70-74"
        if a < 80:  return "75-79"
        if a < 85:  return "80-84"
        return "85+"
    result["age_band"] = result["age"].apply(age_band)

    # Summary stats
    print(f"\n  Generated: {len(result):,} synthetic records")
    print(f"  Years: {sorted(result['year'].unique())}")
    print(f"  Regions: {sorted(result['census_region'].unique())}")
    print(f"  Per-service OOP: mean=${result['per_service_oop'].mean():,.2f}, "
          f"median=${result['per_service_oop'].median():,.2f}")
    print(f"  Dual eligible rate: {result['dual_eligible'].mean():.1%}")
    print(f"  Supplemental rate: {result['has_supplemental'].mean():.1%}")

    # Write
    out_path = os.path.join(output_dir, "synthetic_oop.parquet")
    result.to_parquet(out_path, index=False)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\n  -> synthetic_oop.parquet ({size_mb:.1f} MB)")

    # Write metadata
    meta = {
        "description": "SYNTHETIC per-service OOP data — NOT real patient records",
        "source": "Generated from real MCBS PUF distributions + provider gold data",
        "mcbs_years_used": mcbs_years,
        "provider_sample_frac": sample_frac,
        "seed": seed,
        "n_records": len(result),
        "replacement_instructions": (
            "Replace this file with real MCBS LDS data to run Stage 2 on actual data. "
            "The LDS requires a Data Use Agreement with CMS ($600/module). "
            "Ensure the replacement file has the same column schema."
        ),
    }
    meta_path = os.path.join(output_dir, "synthetic_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  -> synthetic_metadata.json")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic per-service OOP data (mimics MCBS LDS)."
    )
    parser.add_argument("--gold-dir", default=DEFAULT_GOLD,
                        help=f"Provider gold directory (default: {DEFAULT_GOLD})")
    parser.add_argument("--mcbs-dir", default=DEFAULT_MCBS,
                        help=f"MCBS silver directory (default: {DEFAULT_MCBS})")
    parser.add_argument("--label-encoders", default=DEFAULT_ENCODERS,
                        help=f"Path to label_encoders.json (default: {DEFAULT_ENCODERS})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT,
                        help=f"Output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--sample", type=float, default=0.1,
                        help="Fraction of provider gold rows to sample (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()
    generate(args.gold_dir, args.mcbs_dir, args.label_encoders,
             args.output_dir, args.sample, args.seed)
