"""
em_distribution.py -- Phase 9 follow-up (UPCODING rule unlock)

One-pass scan of silver parquets to produce per-(NPI, year) Evaluation &
Management (E&M) code distributions, plus specialty-year benchmarks of
"high-tier share". Feeds the UPCODING rule in check_rules.py, which was
previously NOT EVALUABLE because npi_profiles.parquet does not preserve
per-code counts.

E&M families considered
-----------------------
    Established office visits: 99211 (low) -> 99215 (high)  [primary target]
    New office visits:         99202 -> 99205 (99201 retired pre-2013)

"High-tier" is defined as the top-2 levels of each family:
    est_high  = services on 99214 + 99215
    new_high  = services on 99204 + 99205

The core metric is em_est_high_pct = est_high / est_total.
A provider whose high-tier share sits far above their specialty's P95, on
non-trivial volume, is the classic OIG upcoding signature (MLN SE1418).

Outputs
-------
    local_pipeline/anomaly/em_distributions.parquet
        Keyed by (Rndrng_NPI, year). Columns:
            em_est_total, em_est_99211..99215, em_est_high_pct,
            em_new_total, em_new_99202..99205, em_new_high_pct,
            specialty, state (convenience for joins)
    local_pipeline/anomaly/em_specialty_benchmarks.parquet
        Keyed by (specialty, year). Percentiles of em_est_high_pct and
        em_new_high_pct across providers with est_total >= MIN_VOLUME.

Usage
-----
    python anomaly/rules/em_distribution.py
    python anomaly/rules/em_distribution.py --states CA,TX
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


_PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SILVER  = os.path.join(_PROJECT_ROOT, "local_pipeline", "silver")
DEFAULT_OUT_DIR = os.path.join(_PROJECT_ROOT, "local_pipeline", "anomaly")
DEFAULT_DIST    = os.path.join(DEFAULT_OUT_DIR, "em_distributions.parquet")
DEFAULT_BENCH   = os.path.join(DEFAULT_OUT_DIR, "em_specialty_benchmarks.parquet")

# Established office visits (primary upcoding target per MLN SE1418)
EST_CODES  = ["99211", "99212", "99213", "99214", "99215"]
EST_HIGH   = ["99214", "99215"]

# New office visits
NEW_CODES  = ["99202", "99203", "99204", "99205"]
NEW_HIGH   = ["99204", "99205"]

EM_CODES   = set(EST_CODES) | set(NEW_CODES)

# Minimum annual services in the family to be eligible for the benchmark /
# rule; below this the ratio is too noisy to interpret.
MIN_EST_VOLUME = 50


SILVER_COLS = ["Rndrng_NPI", "Rndrng_Prvdr_Type", "Rndrng_Prvdr_State_Abrvtn",
               "HCPCS_Cd", "Tot_Srvcs", "year"]


def _iter_state(silver_dir: str, states: set[str] | None) -> Iterable[tuple[str, pd.DataFrame]]:
    files = sorted(glob.glob(os.path.join(silver_dir, "*.parquet")))
    if states:
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in states]
    if not files:
        raise SystemExit(f"No silver parquets in {silver_dir}")
    for f in files:
        state = os.path.splitext(os.path.basename(f))[0]
        schema = pq.read_schema(f).names
        use_cols = [c for c in SILVER_COLS if c in schema]
        df = pd.read_parquet(f, columns=use_cols)
        df["HCPCS_Cd"]   = df["HCPCS_Cd"].astype(str).str.strip()
        # Filter to E&M codes of interest BEFORE any further work
        df = df[df["HCPCS_Cd"].isin(EM_CODES)]
        if df.empty:
            continue
        df["Rndrng_NPI"] = df["Rndrng_NPI"].astype(str).str.strip()
        df["year"]       = pd.to_numeric(df["year"], errors="coerce").astype("Int16")
        df = df.dropna(subset=["Rndrng_NPI", "year", "Tot_Srvcs"])
        yield state, df


def build_distributions(silver_dir: str, states: set[str] | None) -> pd.DataFrame:
    """Aggregate per-(NPI, year) service counts by HCPCS, then reshape to wide."""
    parts: list[pd.DataFrame] = []
    t0 = time.time()
    for state, df in _iter_state(silver_dir, states):
        # Collapse (NPI, year, HCPCS) -> sum(Tot_Srvcs); keep specialty/state (first)
        agg = (
            df.groupby(["Rndrng_NPI", "year", "HCPCS_Cd"], observed=True, sort=False)
              .agg(Tot_Srvcs=("Tot_Srvcs", "sum"),
                   specialty=("Rndrng_Prvdr_Type", "first"),
                   state=("Rndrng_Prvdr_State_Abrvtn", "first"))
              .reset_index()
        )
        parts.append(agg)
        print(f"  [{state}] E&M rows={len(df):>7,}  agg={len(agg):>7,}")

    print(f"Scan complete in {time.time()-t0:.1f}s. Concatenating...")
    long = pd.concat(parts, ignore_index=True)
    # Collapse across states (rare: same NPI billing in multiple states)
    grouped = (
        long.groupby(["Rndrng_NPI", "year", "HCPCS_Cd"], observed=True, sort=False)
            .agg(Tot_Srvcs=("Tot_Srvcs", "sum"),
                 specialty=("specialty", "first"),
                 state=("state", "first"))
            .reset_index()
    )

    # Pivot HCPCS to columns
    srvcs = grouped.pivot_table(
        index=["Rndrng_NPI", "year"], columns="HCPCS_Cd",
        values="Tot_Srvcs", aggfunc="sum", fill_value=0.0,
    )
    # Ensure every expected code column exists
    for c in sorted(EM_CODES):
        if c not in srvcs.columns:
            srvcs[c] = 0.0

    # Bring specialty/state alongside (take first seen per NPI-year)
    meta = (
        grouped.sort_values(["Rndrng_NPI", "year"])
               .groupby(["Rndrng_NPI", "year"], observed=True, sort=False)
               .agg(specialty=("specialty", "first"),
                    state=("state", "first"))
    )
    out = srvcs.join(meta).reset_index()

    # Derived columns
    out["em_est_total"] = out[EST_CODES].sum(axis=1)
    out["em_est_high"]  = out[EST_HIGH].sum(axis=1)
    out["em_est_high_pct"] = np.where(
        out["em_est_total"] > 0,
        out["em_est_high"] / out["em_est_total"],
        np.nan,
    )

    out["em_new_total"] = out[NEW_CODES].sum(axis=1)
    out["em_new_high"]  = out[NEW_HIGH].sum(axis=1)
    out["em_new_high_pct"] = np.where(
        out["em_new_total"] > 0,
        out["em_new_high"] / out["em_new_total"],
        np.nan,
    )

    # Rename raw code columns with em_ prefix for clarity
    rename = {c: f"em_{c}" for c in EST_CODES + NEW_CODES}
    out = out.rename(columns=rename)

    # Dtypes
    for c in [*[f"em_{x}" for x in EST_CODES], *[f"em_{x}" for x in NEW_CODES],
              "em_est_total", "em_new_total", "em_est_high", "em_new_high"]:
        out[c] = out[c].astype("float32")
    for c in ["em_est_high_pct", "em_new_high_pct"]:
        out[c] = out[c].astype("float32")
    out["year"] = out["year"].astype("int16")
    return out


def build_specialty_benchmarks(dist: pd.DataFrame) -> pd.DataFrame:
    """Per (specialty, year) percentile distribution of high-tier share.

    Returns a wide-format frame with one row per (specialty, year) and
    columns {est|new}_high_{p50,p75,p90,p95,p99,mean,n}.
    """
    qualifying = dist[dist["em_est_total"] >= MIN_EST_VOLUME].copy()
    groups = qualifying.groupby(["specialty", "year"], observed=True)

    rows: list[dict] = []
    for (spec, yr), g in groups:
        rec: dict = {"specialty": spec, "year": int(yr)}
        for col, prefix in [("em_est_high_pct", "est_high"),
                            ("em_new_high_pct", "new_high")]:
            vals = g[col].dropna()
            if vals.empty:
                continue
            for q, name in [(0.50, "p50"), (0.75, "p75"),
                            (0.90, "p90"), (0.95, "p95"), (0.99, "p99")]:
                rec[f"{prefix}_{name}"] = float(vals.quantile(q))
            rec[f"{prefix}_mean"] = float(vals.mean())
            rec[f"{prefix}_n"]    = int(vals.shape[0])
        rows.append(rec)

    bench = pd.DataFrame(rows)
    if "year" in bench.columns:
        bench["year"] = bench["year"].astype("int16")
    return bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver-dir",        default=DEFAULT_SILVER)
    ap.add_argument("--output-dist",       default=DEFAULT_DIST)
    ap.add_argument("--output-benchmarks", default=DEFAULT_BENCH)
    ap.add_argument("--states",            default="",
                    help="Comma-separated state codes; default: all")
    args = ap.parse_args()

    states = {s.strip().upper() for s in args.states.split(",") if s.strip()} or None
    os.makedirs(os.path.dirname(args.output_dist), exist_ok=True)

    print(f"silver_dir: {args.silver_dir}")
    dist = build_distributions(args.silver_dir, states)
    dist.to_parquet(args.output_dist, index=False, compression="snappy")
    print(f"\nWrote {len(dist):,} (NPI, year) E&M rows -> {args.output_dist}")
    print(f"  with em_est_total>=1:        {int((dist['em_est_total']>=1).sum()):,}")
    print(f"  qualifying (>= {MIN_EST_VOLUME} est svcs): {int((dist['em_est_total']>=MIN_EST_VOLUME).sum()):,}")

    bench = build_specialty_benchmarks(dist)
    bench.to_parquet(args.output_benchmarks, index=False, compression="snappy")
    print(f"Wrote {len(bench):,} (specialty, year) benchmark rows -> {args.output_benchmarks}")

    # Spot check: Internal Medicine, latest year
    im = bench[bench["specialty"].str.contains("Internal Medicine", na=False, case=False)]
    if not im.empty:
        latest = im.sort_values("year").iloc[-1]
        def _f(v, spec=".3f"):
            return format(float(v), spec) if v is not None and not pd.isna(v) else "n/a"
        print(f"\nSpot check (Internal Medicine, year {int(latest['year'])}):")
        print(f"  est_high_p50={_f(latest.get('est_high_p50'))}  "
              f"p75={_f(latest.get('est_high_p75'))}  "
              f"p95={_f(latest.get('est_high_p95'))}  "
              f"n={int(latest.get('est_high_n') or 0):,}")


if __name__ == "__main__":
    main()
