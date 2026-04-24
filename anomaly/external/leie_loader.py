"""
leie_loader.py -- Phase 9 follow-up (LEIE_EXCLUDED rule unlock)

Downloads the OIG List of Excluded Individuals/Entities (LEIE) "UPDATED" CSV
from oig.hhs.gov and materializes it as a parquet keyed by NPI. The LEIE is
the authoritative list of individuals and entities excluded from federal
health care programs under 42 USC 1320a-7.

An NPI match on LEIE is a conviction-grade signal -- any Medicare payment
to an actively excluded NPI is program-ineligible by law. This is not a
statistical anomaly, so the LEIE_EXCLUDED rule is a hard override that
should push a brief to CRITICAL.

Columns in LEIE CSV
-------------------
    LASTNAME, FIRSTNAME, MIDNAME, BUSNAME, GENERAL, SPECIALTY, UPIN, NPI,
    DOB, ADDRESS, CITY, STATE, ZIP, EXCLTYPE, EXCLDATE, REINDATE,
    WAIVERDATE, WVRSTATE

EXCLTYPE codes
--------------
    1128a1: Conviction of program-related crimes
    1128a2: Conviction relating to patient abuse/neglect
    1128a3: Felony conviction relating to health care fraud
    1128a4: Felony conviction relating to controlled substances
    1128b*:  Various permissive exclusions (license revocation, default on
             HEAL loan, kickback, etc.)

Inputs
------
    Remote URL:  https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv
    (Monthly full refresh; incremental supplements published separately.)

Outputs
-------
    local_pipeline/anomaly/leie_exclusions.parquet
        Full LEIE, parquet-encoded, filtered to rows with a valid NPI.
    local_pipeline/anomaly/leie_metadata.json
        Source URL, download timestamp, row count, NPI-match row count.

Usage
-----
    python anomaly/external/leie_loader.py
    python anomaly/external/leie_loader.py --source path/to/UPDATED.csv
    python anomaly/external/leie_loader.py --insecure   # skip TLS verify
"""

from __future__ import annotations

import argparse
import io
import json
import os
import ssl
import sys
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen

import pandas as pd


_PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT_DIR = os.path.join(_PROJECT_ROOT, "local_pipeline", "anomaly")
DEFAULT_OUT     = os.path.join(DEFAULT_OUT_DIR, "leie_exclusions.parquet")
DEFAULT_META    = os.path.join(DEFAULT_OUT_DIR, "leie_metadata.json")

LEIE_URL = "https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv"

EXPECTED_COLS = [
    "LASTNAME", "FIRSTNAME", "MIDNAME", "BUSNAME", "GENERAL", "SPECIALTY",
    "UPIN", "NPI", "DOB", "ADDRESS", "CITY", "STATE", "ZIP",
    "EXCLTYPE", "EXCLDATE", "REINDATE", "WAIVERDATE", "WVRSTATE",
]


def fetch_csv(url: str, insecure: bool = False, timeout: int = 120) -> bytes:
    """Fetch the LEIE CSV. Returns raw bytes.

    OIG occasionally rotates TLS certs; --insecure skips verification for
    one-off refreshes where the chain fails locally.
    """
    ctx = None
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (AllowanceMap LEIE loader)"})
    with urlopen(req, timeout=timeout, context=ctx) as resp:
        return resp.read()


def parse_csv(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    # Normalize column names (LEIE header casing has been consistent but guard anyway)
    df.columns = [c.strip().upper() for c in df.columns]
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"LEIE CSV missing expected columns {missing}. Got: {list(df.columns)}"
        )
    # Keep only expected columns in known order
    df = df[EXPECTED_COLS].copy()
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    # NPIs of "0" or empty string mean "no NPI reported" -- still keep the row
    # for completeness but they won't match anything in our pipeline.
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",    default=LEIE_URL,
                    help="HTTP(S) URL or local file path to LEIE CSV")
    ap.add_argument("--output",    default=DEFAULT_OUT)
    ap.add_argument("--metadata",  default=DEFAULT_META)
    ap.add_argument("--insecure",  action="store_true",
                    help="Disable TLS verification (use only if cert chain fails)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    t0 = time.time()

    if args.source.startswith(("http://", "https://")):
        print(f"Fetching {args.source}  (insecure={args.insecure}) ...")
        raw = fetch_csv(args.source, insecure=args.insecure)
        fetched_from = args.source
    else:
        print(f"Reading local file {args.source} ...")
        with open(args.source, "rb") as fh:
            raw = fh.read()
        fetched_from = f"file://{os.path.abspath(args.source)}"

    print(f"  {len(raw):,} bytes ({time.time()-t0:.1f}s). Parsing...")
    df = parse_csv(raw)
    total = len(df)

    npi_matched = df[df["NPI"].str.len().ge(10) & df["NPI"].ne("0")]

    df.to_parquet(args.output, index=False, compression="snappy")

    meta = {
        "source":                  fetched_from,
        "fetched_at":              datetime.now(timezone.utc).isoformat(),
        "row_count_total":         int(total),
        "row_count_with_npi":      int(len(npi_matched)),
        "columns":                 EXPECTED_COLS,
    }
    with open(args.metadata, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Wrote {total:,} LEIE rows -> {args.output}")
    print(f"  NPI-indexed rows:  {len(npi_matched):,} "
          f"({len(npi_matched)/total*100:.1f}% of LEIE)")
    print(f"Metadata -> {args.metadata}")

    # Spot check: top exclusion types
    print("\nTop 8 exclusion types:")
    print(df["EXCLTYPE"].value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
