"""
cms_revoked.py -- Phase 4 ground-truth source: CMS "Revoked Medicare Providers
and Suppliers".

Downloads the CMS list of providers/suppliers whose Medicare enrollment was
REVOKED under 42 CFR 424.535, and materializes it as a parquet keyed by NPI.
This dataset was first published as a public download in March 2026 (quarterly
refresh, accrualPeriodicity R/P3M) -- it did not exist when the LEIE loader was
built. Unlike the LEIE, every row carries an NPI, so it joins directly to the
project's NPI-keyed flags.

A revocation under 424.535 is an ADMINISTRATIVE enrollment action, NOT a
criminal conviction. It is used here only as one external "the government acted
against this provider" label for lift/lead-time validation, and surfaced to a
human analyst as context -- never as an accusation of fraud.

Source (preferred, stable across quarterly file renames):
    CMS data-API:  https://data.cms.gov/data-api/v1/dataset/{DATASET_UUID}/data
Source (fallback, file path changes each quarter):
    Direct CSV:    https://data.cms.gov/sites/default/files/.../Revocation_Extract_YYYY.MM.DD.vN.csv

Columns
-------
    ENRLMT_ID, NPI, FIRST_NAME, MDL_NAME, LAST_NAME, ORG_NAME,
    MULTIPLE_NPI_FLAG, STATE_CD, PROVIDER_TYPE_DESC, REVOCATION_RSN,
    REVOCATION_EFCTV_DT, REENROLLMENT_BAR_EXPRTN_DT

REVOCATION_RSN holds one or more "424.535(A)(n) <label>" clauses, ';'-joined.
Authority (A)(8) = pattern of abusive billing and (A)(3) = felonies are the
most billing-fraud-relevant; (A)(2) = provider conduct (often an OIG exclusion).

Outputs
-------
    local_pipeline/anomaly/cms_revoked.parquet            -- normalized table
    local_pipeline/anomaly/cms_revoked_metadata.json      -- provenance
    local_pipeline/anomaly/groundtruth_snapshots/cms_revoked_<UTCDATE>.parquet
        -- dated snapshot (the CMS file is a point-in-time, active-only
           snapshot; revoked providers drop off once the re-enrollment bar
           expires, so we MUST persist successive pulls to reconstruct an
           as-of-date label history).

Usage
-----
    python anomaly/groundtruth/cms_revoked.py
    python anomaly/groundtruth/cms_revoked.py --source path/to/Revocation_Extract.csv
    python anomaly/groundtruth/cms_revoked.py --insecure
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import ssl
import sys
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen

import pandas as pd


_PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT_DIR = os.path.join(_PROJECT_ROOT, "local_pipeline", "anomaly")
DEFAULT_OUT     = os.path.join(DEFAULT_OUT_DIR, "cms_revoked.parquet")
DEFAULT_META    = os.path.join(DEFAULT_OUT_DIR, "cms_revoked_metadata.json")
SNAPSHOT_DIR    = os.path.join(DEFAULT_OUT_DIR, "groundtruth_snapshots")

# Stable dataset UUID for "Revoked Medicare Providers and Suppliers".
# The data-API endpoint survives the quarterly CSV file-path rename, so it is
# the preferred autonomous source.
DATASET_UUID = "a6496a7d-4e19-479a-a9ad-d4c0a49e07c3"
DATA_API_URL = f"https://data.cms.gov/data-api/v1/dataset/{DATASET_UUID}/data"

EXPECTED_COLS = [
    "ENRLMT_ID", "NPI", "FIRST_NAME", "MDL_NAME", "LAST_NAME", "ORG_NAME",
    "MULTIPLE_NPI_FLAG", "STATE_CD", "PROVIDER_TYPE_DESC", "REVOCATION_RSN",
    "REVOCATION_EFCTV_DT", "REENROLLMENT_BAR_EXPRTN_DT",
]

# Authority clauses look like "424.535(A)(8) ..." — capture BOTH parentheticals
# and join them into a single code ("A8", "A3", "A2", "A13", ...).
_AUTH_RE = re.compile(r"424\.535\(([A-Za-z0-9]+)\)\(([A-Za-z0-9]+)\)", re.IGNORECASE)


def _make_ctx(insecure: bool):
    if not insecure:
        return None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def fetch_data_api(insecure: bool = False, page: int = 5000, timeout: int = 120) -> pd.DataFrame:
    """Page through the CMS data-API and return all rows as a DataFrame."""
    ctx = _make_ctx(insecure)
    rows: list[dict] = []
    offset = 0
    while True:
        url = f"{DATA_API_URL}?size={page}&offset={offset}"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (AllowanceMap CMS revoked loader)"})
        with urlopen(req, timeout=timeout, context=ctx) as resp:
            batch = json.loads(resp.read())
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        if len(batch) < page:
            break
        offset += page
    if not rows:
        raise RuntimeError("CMS data-API returned no rows")
    return pd.DataFrame(rows)


def fetch_csv(url: str, insecure: bool = False, timeout: int = 120) -> pd.DataFrame:
    ctx = _make_ctx(insecure)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (AllowanceMap CMS revoked loader)"})
    with urlopen(req, timeout=timeout, context=ctx) as resp:
        raw = resp.read()
    return pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CMS revoked data missing expected columns {missing}. Got: {list(df.columns)}"
        )
    df = df[EXPECTED_COLS].copy()
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    # Date parsing: data-API returns ISO (YYYY-MM-DD); CSV returns MM/DD/YYYY.
    df["REVOCATION_EFCTV_DT"] = _parse_dates(df["REVOCATION_EFCTV_DT"])
    df["REENROLLMENT_BAR_EXPRTN_DT"] = _parse_dates(df["REENROLLMENT_BAR_EXPRTN_DT"])

    # Parse the 424.535 authority code(s) out of the free-text reason.
    df["AUTHORITY_CODES"] = df["REVOCATION_RSN"].apply(
        lambda s: ";".join(sorted({f"{a.upper()}{b}" for a, b in _AUTH_RE.findall(s or "")}))
    )
    return df


def _parse_dates(s: pd.Series) -> pd.Series:
    # Try ISO first (data-API), fall back to US m/d/Y (CSV); coerce blanks to NaT.
    s = s.replace({"": None})
    out = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    fallback = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    out = out.fillna(fallback)
    # Last resort: let pandas infer anything still unparsed.
    still = out.isna() & s.notna()
    if still.any():
        out.loc[still] = pd.to_datetime(s[still], errors="coerce")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data-api",
                    help="'data-api' (default, stable) or an HTTP(S) CSV URL or local CSV path")
    ap.add_argument("--output", default=DEFAULT_OUT)
    ap.add_argument("--metadata", default=DEFAULT_META)
    ap.add_argument("--insecure", action="store_true",
                    help="Disable TLS verification (use only if the cert chain fails)")
    ap.add_argument("--no-snapshot", action="store_true",
                    help="Skip writing a dated snapshot copy")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    t0 = time.time()

    if args.source == "data-api":
        print(f"Fetching CMS Revoked via data-API (insecure={args.insecure}) ...")
        raw_df = fetch_data_api(insecure=args.insecure)
        fetched_from = DATA_API_URL
    elif args.source.startswith(("http://", "https://")):
        print(f"Fetching {args.source} (insecure={args.insecure}) ...")
        raw_df = fetch_csv(args.source, insecure=args.insecure)
        fetched_from = args.source
    else:
        print(f"Reading local file {args.source} ...")
        raw_df = pd.read_csv(args.source, dtype=str, keep_default_na=False)
        fetched_from = f"file://{os.path.abspath(args.source)}"

    df = normalize(raw_df)
    total = len(df)
    valid_npi = df["NPI"].str.len().ge(10) & df["NPI"].ne("0") & ~df["NPI"].str.fullmatch(r"0+")
    n_npi = int(valid_npi.sum())

    df.to_parquet(args.output, index=False, compression="snappy")

    snapshot_path = None
    if not args.no_snapshot:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        snapshot_path = os.path.join(SNAPSHOT_DIR, f"cms_revoked_{stamp}.parquet")
        df.to_parquet(snapshot_path, index=False, compression="snappy")

    meta = {
        "source":             fetched_from,
        "fetched_at":         datetime.now(timezone.utc).isoformat(),
        "row_count_total":    int(total),
        "row_count_with_npi": n_npi,
        "snapshot":           snapshot_path,
        "columns":            EXPECTED_COLS + ["AUTHORITY_CODES"],
    }
    with open(args.metadata, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Wrote {total:,} CMS revoked rows -> {args.output}  ({time.time()-t0:.1f}s)")
    print(f"  Rows with valid NPI: {n_npi:,} ({n_npi/total*100:.1f}%)")
    if snapshot_path:
        print(f"  Snapshot -> {snapshot_path}")
    print(f"  Metadata -> {args.metadata}")

    auth = (df["AUTHORITY_CODES"].str.split(";").explode().str.strip())
    auth = auth[auth.ne("")]
    print("\nTop 424.535 authority codes:")
    print(auth.value_counts().head(12).to_string())


if __name__ == "__main__":
    main()
