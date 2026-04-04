"""
pull_provider_data.py — Download CMS "by Provider" summary data

Downloads the Medicare Physician & Other Practitioners — by Provider dataset
for 2013-2023. This dataset has one row per NPI per year and includes
Bene_Avg_Risk_Scre (HCC risk score) which is NOT in the "by Provider and
Service" dataset used by pull_medicare_data.py.

The risk scores are joined into the gold layer by 03_gold_features_local.py
on (Rndrng_NPI, year).

Usage:
    python pull_provider_data.py
    python pull_provider_data.py --year 2023
    python pull_provider_data.py --output-dir data
"""

import os
import argparse
import requests

# Direct CSV download URLs from CMS data catalog
# Source: https://catalog.data.gov/dataset/medicare-physician-other-practitioners-by-provider-b297e
DOWNLOAD_URLS = {
    2023: "https://data.cms.gov/sites/default/files/2025-04/22edfd1e-d17a-4478-ad6b-92cac2a5a3c4/MUP_PHY_R25_P05_V20_D23_Prov.csv",
    2022: "https://data.cms.gov/sites/default/files/2025-11/adcd20c5-4534-43cd-8dfa-881ebe7bacfd/MUP_PHY_R25_P07_V20_D22_Prov.csv",
    2021: "https://data.cms.gov/sites/default/files/2025-11/fc6ea9aa-12f0-4c2f-9909-6c8e06c961cf/MUP_PHY_R25_P07_V20_D21_Prov.csv",
    2020: "https://data.cms.gov/sites/default/files/2025-11/056e8c6b-7e39-4945-b9a4-52d0a1cbbb9a/MUP_PHY_R25_P07_V20_D20_Prov.csv",
    2019: "https://data.cms.gov/sites/default/files/2025-11/ac110c46-3429-4f3c-9348-56f0a5312cb8/MUP_PHY_R25_P07_V20_D19_Prov.csv",
    2018: "https://data.cms.gov/sites/default/files/2025-11/57ea60f2-ef4b-46f2-8778-c7a50fab1737/MUP_PHY_R25_P07_V20_D18_Prov.csv",
    2017: "https://data.cms.gov/sites/default/files/2025-11/b7e195ce-710d-4584-8293-787ffc38b017/MUP_PHY_R25_P07_V20_D17_Prov.csv",
    2016: "https://data.cms.gov/sites/default/files/2025-11/e4cd28e1-2d05-4f33-b232-f32b5bebc153/MUP_PHY_R25_P05_V20_D16_Prov.csv",
    2015: "https://data.cms.gov/sites/default/files/2025-11/efb7e2f9-b903-4f54-b57f-742f436c316f/MUP_PHY_R25_P05_V20_D15_Prov.csv",
    2014: "https://data.cms.gov/sites/default/files/2025-11/da4b3b2b-1ee2-4e25-b2fe-012b880afd37/MUP_PHY_R25_P05_V20_D14_Prov.csv",
    2013: "https://data.cms.gov/sites/default/files/2025-11/4fe919a0-94d4-4681-b076-4106f3766eef/MUP_PHY_R25_P05_V20_D13_Prov.csv",
}


def download_year(year: int, url: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"provider_summary_{year}.csv")

    if os.path.exists(output_file):
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  {year}: Already exists ({size_mb:.0f} MB) — skipping. Delete to re-download.")
        return

    print(f"  {year}: Downloading from CMS...")
    try:
        resp = requests.get(url, stream=True, timeout=600)
        resp.raise_for_status()

        total = 0
        with open(output_file, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                total += len(chunk)
                print(f"    {total / (1024*1024):.0f} MB downloaded...", end="\r")

        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  {year}: Done — {size_mb:.0f} MB saved to {output_file}")

    except requests.exceptions.RequestException as e:
        print(f"  {year}: ERROR — {e}")
        if os.path.exists(output_file):
            os.remove(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download CMS 'by Provider' summary data (includes Bene_Avg_Risk_Scre)."
    )
    parser.add_argument("--year", type=int,
                        help="Specific year to pull (e.g. 2023). If omitted, pulls all years.")
    parser.add_argument("--output-dir", default="data",
                        help="Output directory for CSVs (default: data/)")
    args = parser.parse_args()

    years = {args.year: DOWNLOAD_URLS[args.year]} if args.year else DOWNLOAD_URLS

    print(f"Downloading CMS 'by Provider' data for {len(years)} year(s)...")
    for year, url in sorted(years.items()):
        download_year(year, url, args.output_dir)
    print("\nDone.")
