"""
pull_external_covariates.py — Download external covariates for V2 forecasting

Pulls:
  1. Medicare Conversion Factor (CF) — hardcoded from CMS PFS final rules
  2. Medical Care CPI — BLS API (Series CUUR0000SAM), fallback to hardcoded
  3. Sequestration Rate — hardcoded from CMS published rates
  4. MACRA/MIPS Adjustment — hardcoded from QPP final rules
  5. COVID Indicators — hardcoded binary flags

Output: CSVs in --output-dir (default: local_pipeline/external/)

Usage:
    python pull_external_covariates.py
    python pull_external_covariates.py --output-dir local_pipeline/external/
    python pull_external_covariates.py --bls-key YOUR_API_KEY  # optional, higher rate limit
"""

import os
import csv
import argparse
import requests
import time

# ── Medicare Conversion Factor ─────────────────────────────────────────────
# Source: CMS Physician Fee Schedule Final Rule fact sheets
# https://www.cms.gov/medicare/payment/fee-schedules/physician/pfs-relative-value-files
CONVERSION_FACTORS = {
    2013: 34.0230,
    2014: 35.8013,
    2015: 35.7547,
    2016: 35.8043,
    2017: 35.8887,
    2018: 35.9996,
    2019: 36.0391,
    2020: 36.0896,
    2021: 34.8931,
    2022: 33.5983,
    2023: 33.0607,
    2024: 32.7442,
    2025: 32.3540,
    2026: 31.9200,  # CY 2026 PFS Final Rule (estimated)
}

# ── Medical Care CPI (hardcoded fallback) ──────────────────────────────────
# Source: BLS Series CUUR0000SAM — Medical care, U.S. city average, not seasonally adjusted
# https://data.bls.gov/timeseries/CUUR0000SAM
# Annual averages; 2025-2026 projected from recent trend (~3% annual growth)
MEDICAL_CPI_FALLBACK = {
    2013: 425.1,
    2014: 435.3,
    2015: 446.8,
    2016: 453.0,
    2017: 457.2,
    2018: 463.8,
    2019: 471.4,
    2020: 479.8,
    2021: 490.6,
    2022: 516.0,
    2023: 536.7,
    2024: 551.8,
    2025: 568.3,  # projected
    2026: 585.4,  # projected
}

# ── Sequestration Rate ─────────────────────────────────────────────────────
# Source: https://www.cms.gov/medicare/payment/claims-based-sequestration
# Budget Control Act of 2011; suspended during COVID
SEQUESTRATION_RATES = {
    2013: 0.020,  # effective Apr 1 (prorated ~0.2% for partial year; using 2%)
    2014: 0.020,
    2015: 0.020,
    2016: 0.020,
    2017: 0.020,
    2018: 0.020,
    2019: 0.020,
    2020: 0.010,  # suspended May 1 (CARES Act) — ~half year
    2021: 0.000,  # fully suspended through Dec 31
    2022: 0.010,  # phased back: 1% Jan-Mar, 2% Apr onward — blended ~1%
    2023: 0.020,  # fully restored
    2024: 0.020,
    2025: 0.020,
    2026: 0.020,
}

# ── MACRA/MIPS Payment Adjustments ─────────────────────────────────────────
# Source: https://qpp.cms.gov/resources/all-resources
# Mean payment adjustment factor from QPP final score distributions
# Pre-MACRA (2013-2018): 0.0 (MIPS didn't exist or no payment impact yet)
# MIPS adjustments are small (±few %), this captures the national mean
MACRA_MIPS = {
    2013: 0.000,
    2014: 0.000,
    2015: 0.000,
    2016: 0.000,
    2017: 0.000,
    2018: 0.000,  # MIPS Year 1 (PY2017), adjustments applied CY2019
    2019: 0.002,  # PY2017 adjustments: most providers positive, small
    2020: 0.004,  # PY2018 adjustments
    2021: 0.005,  # PY2019 adjustments
    2022: 0.000,  # PY2020: COVID automatic extreme hardship exception
    2023: 0.005,  # PY2021 adjustments
    2024: 0.006,  # PY2022 adjustments
    2025: 0.005,  # PY2023 adjustments (estimated)
    2026: 0.005,  # PY2024 adjustments (estimated)
}

# ── COVID Indicators ───────────────────────────────────────────────────────
# Binary: 1 = pandemic year with significant utilization disruption
COVID_INDICATORS = {yr: (1 if yr in (2020, 2021) else 0) for yr in range(2013, 2027)}


def _write_csv(filepath, header, data_dict):
    """Write a {year: value} dict to CSV."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for year in sorted(data_dict.keys()):
            writer.writerow([year, data_dict[year]])
    print(f"  Wrote {filepath} ({len(data_dict)} rows)")


def pull_bls_cpi(api_key=None, start_year=2013, end_year=2025):
    """
    Pull Medical Care CPI from BLS Public Data API v2.

    API docs: https://www.bls.gov/developers/api_signature_v2.htm
    Series:   CUUR0000SAM (Medical care in U.S. city average)

    Without API key: 25 requests/day, 10-year max span.
    With API key:    500 requests/day, 20-year max span.
    Register at: https://data.bls.gov/registrationEngine/
    """
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    headers = {"Content-Type": "application/json"}

    payload = {
        "seriesid": ["CUUR0000SAM"],
        "startyear": str(start_year),
        "endyear": str(end_year),
        "annualaverage": True,
    }
    if api_key:
        payload["registrationkey"] = api_key

    print(f"  Fetching BLS CPI Medical (CUUR0000SAM) {start_year}-{end_year}...")
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        if result.get("status") != "REQUEST_SUCCEEDED":
            msg = result.get("message", [])
            print(f"  [WARN] BLS API error: {msg}")
            return None

        series_data = result["Results"]["series"][0]["data"]
        annual = {}
        for entry in series_data:
            if entry["period"] == "M13":  # M13 = annual average
                annual[int(entry["year"])] = float(entry["value"])
            elif entry["periodName"] == "Annual":
                annual[int(entry["year"])] = float(entry["value"])

        if annual:
            print(f"  BLS API returned {len(annual)} annual values")
            return annual
        else:
            # No annual average rows — compute from monthly
            monthly = {}
            for entry in series_data:
                if entry["period"].startswith("M") and entry["period"] != "M13":
                    yr = int(entry["year"])
                    monthly.setdefault(yr, []).append(float(entry["value"]))
            for yr, vals in monthly.items():
                annual[yr] = round(sum(vals) / len(vals), 1)
            if annual:
                print(f"  BLS API: computed annual avg from monthly for {len(annual)} years")
                return annual

            print("  [WARN] BLS API returned data but no annual averages found")
            return None

    except requests.RequestException as e:
        print(f"  [WARN] BLS API request failed: {e}")
        return None
    except (KeyError, IndexError, ValueError) as e:
        print(f"  [WARN] BLS API parse error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Pull external covariates for V2 forecasting")
    parser.add_argument("--output-dir", default=os.path.join("local_pipeline", "external"),
                        help="Output directory (default: local_pipeline/external/)")
    parser.add_argument("--bls-key", default=None,
                        help="BLS API registration key (optional, increases rate limit)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}\n")

    # 1. Conversion factors
    print("[1/5] Medicare Conversion Factor")
    _write_csv(
        os.path.join(args.output_dir, "conversion_factors.csv"),
        ["year", "conversion_factor"],
        CONVERSION_FACTORS,
    )

    # 2. Medical Care CPI — try BLS API, fall back to hardcoded
    print("[2/5] Medical Care CPI")
    bls_data = pull_bls_cpi(api_key=args.bls_key)
    if bls_data:
        # Merge: BLS actuals + fallback projections for missing years
        cpi_merged = dict(MEDICAL_CPI_FALLBACK)
        cpi_merged.update(bls_data)  # API values override fallback
        source = "BLS API + projected"
    else:
        print("  Using hardcoded fallback values")
        cpi_merged = dict(MEDICAL_CPI_FALLBACK)
        source = "hardcoded fallback"
    _write_csv(
        os.path.join(args.output_dir, "medical_cpi.csv"),
        ["year", "cpi_medical"],
        cpi_merged,
    )
    print(f"  Source: {source}")

    # 3. Sequestration rates
    print("[3/5] Sequestration Rate")
    _write_csv(
        os.path.join(args.output_dir, "sequestration_rates.csv"),
        ["year", "sequestration_rate"],
        SEQUESTRATION_RATES,
    )

    # 4. MACRA/MIPS adjustments
    print("[4/5] MACRA/MIPS Adjustment")
    _write_csv(
        os.path.join(args.output_dir, "macra_mips.csv"),
        ["year", "mips_adjustment"],
        MACRA_MIPS,
    )

    # 5. COVID indicators
    print("[5/5] COVID Indicators")
    _write_csv(
        os.path.join(args.output_dir, "covid_indicators.csv"),
        ["year", "covid_indicator"],
        COVID_INDICATORS,
    )

    print(f"\nDone. All covariates written to {args.output_dir}/")
    print("Upload this directory to Google Drive: My Drive/medicare_v2/external/")


if __name__ == "__main__":
    main()
