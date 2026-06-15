"""
build_labels.py -- Phase 4: unify the government ground-truth sources into a
single NPI-keyed label table for backtesting fraud-lead flags.

Combines:
  - OIG LEIE exclusions   (local_pipeline/anomaly/leie_exclusions.parquet)
  - CMS Revoked providers (local_pipeline/anomaly/cms_revoked.parquet)

into local_pipeline/anomaly/labels.parquet with one row per (npi, source, event):

    npi            str   10-digit NPI (zero-padded preserved)
    source         str   'LEIE' | 'CMS_REVOKED'
    event_date     datetime64  exclusion / revocation effective date
    event_year     int   year of event_date
    authority      str   EXCLTYPE (LEIE) or ';'-joined 424.535 codes (CMS)
    tier           str   'leie_conviction' | 'leie_permissive'
                         | 'cms_billing_fraud' | 'cms_enrollment_integrity'
    fraud_relevant bool  True for the billing-fraud-grade subset (headline label)
    reinstated     bool  LEIE only: a REINDATE is present (exclusion later lifted)
    name           str   best-effort display name
    state          str

Label semantics (documented, because they bound every downstream metric):
  * These are POSITIVE-ONLY labels. Absence from this table is "unlabeled",
    NOT a confirmed true-negative.
  * fraud_relevant follows the field-standard fraud-indicative subset:
      LEIE  -> 1128a1/a2/a3 (mandatory convictions) + 1128b4 (fraud/kickback)
               + 1128b7 (fraud/quality) [Herland 2018 / Hancock 2023]
      CMS   -> 424.535(a)(3) felonies + (a)(8) pattern of abusive billing
  * An LEIE/CMS hit means a sanction OCCURRED; it does not prove fraud in the
    specific year a provider was flagged. EXCLDATE/EXCLTYPE are carried verbatim
    so a human analyst interprets, not the agent.

Usage
-----
    python anomaly/groundtruth/build_labels.py
    python anomaly/groundtruth/build_labels.py --snapshot   # also copy sources into a dated snapshot dir
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone

import pandas as pd


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR       = os.path.join(_PROJECT_ROOT, "local_pipeline", "anomaly")
LEIE_PARQUET  = os.path.join(OUT_DIR, "leie_exclusions.parquet")
CMS_PARQUET   = os.path.join(OUT_DIR, "cms_revoked.parquet")
LABELS_OUT    = os.path.join(OUT_DIR, "labels.parquet")
LABELS_META   = os.path.join(OUT_DIR, "labels_metadata.json")
SNAPSHOT_DIR  = os.path.join(OUT_DIR, "groundtruth_snapshots")

# Field-standard fraud-indicative LEIE exclusion types (Herland 2018 / Hancock 2023).
LEIE_FRAUD_TYPES = {"1128A1", "1128A2", "1128A3", "1128B4", "1128B7"}
# Conviction-grade: mandatory exclusions from a CRIMINAL CONVICTION (1128A1/A2/A3).
# The strict headline label -- excludes permissive/administrative actions. (1128A4
# controlled-substance is omitted per Herland/Hancock fraud-code convention.)
LEIE_CONVICTION_TYPES = {"1128A1", "1128A2", "1128A3"}
# Billing-fraud-grade CMS revocation authorities.
CMS_FRAUD_AUTH   = {"A3", "A8"}


def _valid_npi(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    return s.str.len().ge(10) & s.ne("0") & ~s.str.fullmatch(r"0+")


def load_leie() -> pd.DataFrame:
    if not os.path.exists(LEIE_PARQUET):
        print(f"  [skip] LEIE not found at {LEIE_PARQUET} (run anomaly/external/leie_loader.py)")
        return pd.DataFrame()
    df = pd.read_parquet(LEIE_PARQUET)
    df = df[_valid_npi(df["NPI"])].copy()
    excltype = df["EXCLTYPE"].astype(str).str.strip().str.upper()
    event_date = pd.to_datetime(df["EXCLDATE"].astype(str).str.strip(), format="%Y%m%d", errors="coerce")
    reindate = df["REINDATE"].astype(str).str.strip()
    name = (df["LASTNAME"].fillna("").str.strip() + ", " + df["FIRSTNAME"].fillna("").str.strip()).str.strip(", ")
    name = name.where(name.ne(""), df["BUSNAME"].fillna("").str.strip())
    out = pd.DataFrame({
        "npi":            df["NPI"].astype(str).str.strip(),
        "source":         "LEIE",
        "event_date":     event_date,
        "authority":      excltype,
        "fraud_relevant": excltype.isin(LEIE_FRAUD_TYPES),
        "conviction":     excltype.isin(LEIE_CONVICTION_TYPES),
        "tier":           excltype.str.startswith("1128A").map({True: "leie_conviction", False: "leie_permissive"}),
        "reinstated":     reindate.ne("") & reindate.ne("00000000") & reindate.notna(),
        "name":           name,
        "state":          df["STATE"].astype(str).str.strip(),
    })
    return out.dropna(subset=["event_date"])


def load_cms() -> pd.DataFrame:
    if not os.path.exists(CMS_PARQUET):
        print(f"  [skip] CMS revoked not found at {CMS_PARQUET} (run anomaly/groundtruth/cms_revoked.py)")
        return pd.DataFrame()
    df = pd.read_parquet(CMS_PARQUET)
    df = df[_valid_npi(df["NPI"])].copy()
    auth = df["AUTHORITY_CODES"].astype(str).str.upper()
    auth_sets = auth.str.split(";").apply(lambda xs: {x.strip() for x in xs if x.strip()})
    fraud = auth_sets.apply(lambda s: bool(s & CMS_FRAUD_AUTH))
    name = (df["LAST_NAME"].fillna("").str.strip() + ", " + df["FIRST_NAME"].fillna("").str.strip()).str.strip(", ")
    name = name.where(name.ne(""), df["ORG_NAME"].fillna("").str.strip())
    out = pd.DataFrame({
        "npi":            df["NPI"].astype(str).str.strip(),
        "source":         "CMS_REVOKED",
        "event_date":     pd.to_datetime(df["REVOCATION_EFCTV_DT"], errors="coerce"),
        "authority":      auth,
        "fraud_relevant": fraud,
        "conviction":     False,  # CMS revocation is an administrative action, not a criminal conviction
        "tier":           fraud.map({True: "cms_billing_fraud", False: "cms_enrollment_integrity"}),
        "reinstated":     False,
        "name":           name,
        "state":          df["STATE_CD"].astype(str).str.strip(),
    })
    return out.dropna(subset=["event_date"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=LABELS_OUT)
    ap.add_argument("--metadata", default=LABELS_META)
    ap.add_argument("--snapshot", action="store_true",
                    help="Also copy the source parquets into a dated snapshot dir")
    args = ap.parse_args()

    print("Building unified ground-truth labels ...")
    parts = [load_leie(), load_cms()]
    parts = [p for p in parts if not p.empty]
    if not parts:
        print("ERROR: no ground-truth sources available. Run the loaders first.")
        raise SystemExit(1)

    labels = pd.concat(parts, ignore_index=True)
    labels["event_year"] = labels["event_date"].dt.year.astype("int32")
    labels = labels[[
        "npi", "source", "event_date", "event_year", "authority",
        "tier", "fraud_relevant", "conviction", "reinstated", "name", "state",
    ]].sort_values(["npi", "event_date"]).reset_index(drop=True)

    labels.to_parquet(args.output, index=False, compression="snappy")

    if args.snapshot:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        for src in (LEIE_PARQUET, CMS_PARQUET):
            if os.path.exists(src):
                base = os.path.splitext(os.path.basename(src))[0]
                shutil.copy2(src, os.path.join(SNAPSHOT_DIR, f"{base}_{stamp}.parquet"))

    meta = {
        "built_at":          datetime.now(timezone.utc).isoformat(),
        "n_label_rows":      int(len(labels)),
        "n_unique_npi":      int(labels["npi"].nunique()),
        "n_fraud_relevant":  int(labels["fraud_relevant"].sum()),
        "n_conviction":      int(labels["conviction"].sum()),
        "by_source":         labels["source"].value_counts().to_dict(),
        "by_tier":           labels["tier"].value_counts().to_dict(),
        "event_year_range":  [int(labels["event_year"].min()), int(labels["event_year"].max())],
        "leie_fraud_types":  sorted(LEIE_FRAUD_TYPES),
        "cms_fraud_auth":    sorted(CMS_FRAUD_AUTH),
    }
    with open(args.metadata, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Wrote {len(labels):,} label rows -> {args.output}")
    print(f"  unique NPIs: {meta['n_unique_npi']:,}  |  fraud-relevant rows: {meta['n_fraud_relevant']:,}")
    print(f"  by source: {meta['by_source']}")
    print(f"  by tier:   {meta['by_tier']}")
    print(f"  event-year range: {meta['event_year_range']}")
    print(f"Metadata -> {args.metadata}")


if __name__ == "__main__":
    main()
