"""
pull_mcbs_data.py — Download MCBS Public Use File data from CMS

Downloads the Medicare Current Beneficiary Survey (MCBS) Survey File and
Cost Supplement PUF ZIP archives from data.cms.gov, then extracts CSVs.

Available years (no Data Use Agreement required):
  - Survey File:      2015-2022
  - Cost Supplement:  2018-2022

Usage:
    python pull_mcbs_data.py
    python pull_mcbs_data.py --type survey --year 2022
    python pull_mcbs_data.py --type cost --year 2021
    python pull_mcbs_data.py --type both
    python pull_mcbs_data.py --url https://custom-url.zip --output-dir data/mcbs
"""

import os
import argparse
import zipfile
import requests

# Known download URLs from data.cms.gov
# Pattern: SFPUF{year}_Data.zip (survey), CSPUF{year}_Data.zip (cost supplement)
# NOTE: CMS occasionally updates these URLs. If a download fails with 404,
#       visit https://data.cms.gov/medicare-current-beneficiary-survey-mcbs
#       to find the current URL, then pass it via --url.
SURVEY_URLS = {
    2022: "https://data.cms.gov/sites/default/files/2024-10/SFPUF2022_Data.zip",
    2021: "https://data.cms.gov/sites/default/files/2023-10/SFPUF2021_Data.zip",
    2020: "https://data.cms.gov/sites/default/files/2022-10/SFPUF2020_Data.zip",
    2019: "https://data.cms.gov/sites/default/files/2022-01/SFPUF2019_Data.zip",
    2018: "https://data.cms.gov/sites/default/files/2021-08/SFPUF2018_Data.zip",
    2017: "https://data.cms.gov/sites/default/files/2021-01/SFPUF2017_Data.zip",
    2016: "https://data.cms.gov/sites/default/files/2020-09/SFPUF2016_Data.zip",
    2015: "https://data.cms.gov/sites/default/files/2020-09/SFPUF2015_Data.zip",
}

COST_URLS = {
    2022: "https://data.cms.gov/sites/default/files/2025-01/CSPUF2022_Data.zip",
    2021: "https://data.cms.gov/sites/default/files/2024-01/CSPUF2021_Data.zip",
    2020: "https://data.cms.gov/sites/default/files/2023-01/CSPUF2020_Data.zip",
    2019: "https://data.cms.gov/sites/default/files/2022-01/CSPUF2019_Data.zip",
    2018: "https://data.cms.gov/sites/default/files/2021-08/CSPUF2018_Data.zip",
}


def download_file(url: str, output_path: str) -> bool:
    """Download a file via streaming HTTPS. Returns True on success."""
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"    Already exists ({size_mb:.1f} MB) — skipping. Delete to re-download.")
        return True

    try:
        resp = requests.get(url, stream=True, timeout=600)
        resp.raise_for_status()

        total = 0
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                total += len(chunk)
                print(f"    {total / (1024*1024):.0f} MB downloaded...", end="\r")

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"    Downloaded {size_mb:.1f} MB -> {output_path}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"    ERROR: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        if hasattr(e, "response") and e.response is not None and e.response.status_code == 404:
            print("    URL may have changed. Visit the CMS MCBS page to find the updated link:")
            print("    https://data.cms.gov/medicare-current-beneficiary-survey-mcbs")
            print("    Then re-run with: --url <new_url>")
        return False


def extract_csv_from_zip(zip_path: str, output_dir: str, prefix: str, year: int) -> list[str]:
    """Extract CSV files from a ZIP archive. Returns list of extracted file paths."""
    extracted = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_files = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_files:
                print(f"    [WARN] No CSV files found in {zip_path}")
                print(f"    Contents: {zf.namelist()[:10]}")
                return extracted

            for csv_name in csv_files:
                # Rename to our convention: survey_2022.csv or cost_2021.csv
                ext_name = f"{prefix}_{year}.csv"
                ext_path = os.path.join(output_dir, ext_name)
                if os.path.exists(ext_path):
                    print(f"    {ext_name} already extracted — skipping")
                    extracted.append(ext_path)
                    continue

                with zf.open(csv_name) as src, open(ext_path, "wb") as dst:
                    dst.write(src.read())
                size_mb = os.path.getsize(ext_path) / (1024 * 1024)
                print(f"    Extracted: {ext_name} ({size_mb:.1f} MB)")
                extracted.append(ext_path)
                break  # Take first CSV only (data file, not codebook)

    except zipfile.BadZipFile:
        print(f"    [ERROR] Corrupt ZIP: {zip_path}")
    return extracted


def download_year(year: int, file_type: str, output_dir: str) -> list[str]:
    """Download and extract MCBS data for one year. Returns extracted CSV paths."""
    os.makedirs(output_dir, exist_ok=True)
    extracted = []

    if file_type in ("survey", "both"):
        url = SURVEY_URLS.get(year)
        if url:
            print(f"  Survey File {year}:")
            zip_path = os.path.join(output_dir, f"SFPUF{year}_Data.zip")
            if download_file(url, zip_path):
                extracted.extend(extract_csv_from_zip(zip_path, output_dir, "survey", year))
        else:
            print(f"  Survey File {year}: No URL available (years: {sorted(SURVEY_URLS.keys())})")

    if file_type in ("cost", "both"):
        url = COST_URLS.get(year)
        if url:
            print(f"  Cost Supplement {year}:")
            zip_path = os.path.join(output_dir, f"CSPUF{year}_Data.zip")
            if download_file(url, zip_path):
                extracted.extend(extract_csv_from_zip(zip_path, output_dir, "cost", year))
        else:
            print(f"  Cost Supplement {year}: No URL available (years: {sorted(COST_URLS.keys())})")

    return extracted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download MCBS Public Use File data from CMS (Survey File + Cost Supplement)."
    )
    parser.add_argument("--year", type=int,
                        help="Specific year (e.g. 2022). If omitted, downloads all available years.")
    parser.add_argument("--type", choices=["survey", "cost", "both"], default="both",
                        help="Which MCBS file(s) to download (default: both)")
    parser.add_argument("--output-dir", default=os.path.join("data", "mcbs"),
                        help="Output directory (default: data/mcbs)")
    parser.add_argument("--url",
                        help="Override: download a specific URL (use with --year and --type)")
    args = parser.parse_args()

    if args.url:
        # Manual URL override
        if not args.year:
            parser.error("--url requires --year")
        prefix = "survey" if args.type == "survey" else "cost"
        os.makedirs(args.output_dir, exist_ok=True)
        zip_path = os.path.join(args.output_dir, f"{prefix}_{args.year}_manual.zip")
        print(f"Downloading custom URL for {args.type} {args.year}...")
        if download_file(args.url, zip_path):
            extract_csv_from_zip(zip_path, args.output_dir, prefix, args.year)
    else:
        # Determine years to download
        if args.year:
            years = [args.year]
        else:
            all_years = set(SURVEY_URLS.keys()) | set(COST_URLS.keys())
            years = sorted(all_years)

        print(f"MCBS PUF download: {len(years)} year(s), type={args.type}")
        print(f"Output: {args.output_dir}\n")

        total_extracted = []
        for year in years:
            total_extracted.extend(download_year(year, args.type, args.output_dir))

        print(f"\nDone. Extracted {len(total_extracted)} CSV file(s).")
