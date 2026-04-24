"""
specialty_scopes.py -- Phase 9 follow-up for OUT_OF_SPECIALTY rule

One-time scan of silver parquets to produce a per-specialty HCPCS whitelist.
A code is considered "in scope" for a specialty if it satisfies either:

  (a) Coverage: billed by >= COVERAGE_MIN_PCT of the specialty's active
      providers in any year (default 1.0%). This captures common legitimate
      codes regardless of total volume.

  (b) Volume: the specialty's cumulative service share from the top codes,
      sorted by total volume, reaches >= VOLUME_CUMULATIVE_PCT (default
      99.0%). This rescues niche but legitimate codes that a minority of
      specialists perform in large quantities (e.g., sub-specialty
      procedures).

A code that fails both is flagged out-of-scope. The spec's OUT_OF_SPECIALTY
rule (42 CFR 424.22) fires when a provider's out-of-scope code share crosses
a threshold (default 20% of their services).

Inputs
------
    local_pipeline/silver/{STATE}.parquet  (columns: Rndrng_NPI,
        Rndrng_Prvdr_Type, HCPCS_Cd, Tot_Srvcs)

Outputs
-------
    local_pipeline/anomaly/specialty_scopes.parquet
        Long-format rows keyed by (specialty, HCPCS_Cd) with coverage +
        volume statistics plus in_scope flag.
    local_pipeline/anomaly/specialty_scopes_summary.json
        Per-specialty summary (n_providers, n_codes, in_scope_codes,
        top_out_of_scope_codes). Human-readable.

Usage
-----
    python anomaly/rules/specialty_scopes.py
    python anomaly/rules/specialty_scopes.py --coverage-min-pct 2.0
    python anomaly/rules/specialty_scopes.py --states CA,TX,FL
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


_PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SILVER  = os.path.join(_PROJECT_ROOT, "local_pipeline", "silver")
DEFAULT_OUT_DIR = os.path.join(_PROJECT_ROOT, "local_pipeline", "anomaly")
DEFAULT_PARQUET = os.path.join(DEFAULT_OUT_DIR, "specialty_scopes.parquet")
DEFAULT_SUMMARY = os.path.join(DEFAULT_OUT_DIR, "specialty_scopes_summary.json")

SILVER_COLS = ["Rndrng_NPI", "Rndrng_Prvdr_Type", "HCPCS_Cd", "Tot_Srvcs"]


def _iter_state_frames(silver_dir: str, states: set[str] | None) -> Iterable[tuple[str, pd.DataFrame]]:
    files = sorted(glob.glob(os.path.join(silver_dir, "*.parquet")))
    if states:
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in states]
    if not files:
        raise SystemExit(f"No silver parquets in {silver_dir}")
    for f in files:
        state = os.path.splitext(os.path.basename(f))[0]
        schema_cols = pq.read_schema(f).names
        use_cols = [c for c in SILVER_COLS if c in schema_cols]
        df = pd.read_parquet(f, columns=use_cols)
        df["Rndrng_NPI"]        = df["Rndrng_NPI"].astype(str).str.strip()
        df["HCPCS_Cd"]          = df["HCPCS_Cd"].astype(str).str.strip()
        df["Rndrng_Prvdr_Type"] = df["Rndrng_Prvdr_Type"].astype(str).str.strip()
        df = df.dropna(subset=["Rndrng_NPI", "HCPCS_Cd", "Rndrng_Prvdr_Type", "Tot_Srvcs"])
        yield state, df


def build_scopes(
    silver_dir: str,
    states: set[str] | None,
    coverage_min_pct: float,
    volume_cumulative_pct: float,
) -> tuple[pd.DataFrame, dict]:
    # Stage 1: accumulate (specialty, HCPCS, NPI) set as (n_providers, total_services).
    # We use dicts keyed by (specialty, HCPCS) -> {npis: set, services: float}
    # Tracking NPI sets directly is too memory-hungry (10M+ NPIs). We instead do
    # a two-pass: first collapse per-state to (specialty, HCPCS, NPI)->services,
    # then concat and groupby.

    per_state_parts = []
    per_state_provider_counts: dict[str, set[str]] = {}  # specialty -> NPI set

    t0 = time.time()
    for state, df in _iter_state_frames(silver_dir, states):
        # Collapse row-level (NPI,HCPCS,year,POS) to (NPI, specialty, HCPCS)->services.
        # A provider may appear with multiple specialty labels across years; we
        # keep the most frequent one per NPI within the state.
        npi_spec = (
            df.groupby(["Rndrng_NPI", "Rndrng_Prvdr_Type"], observed=True)
              .size()
              .reset_index(name="_n")
              .sort_values(["Rndrng_NPI", "_n"], ascending=[True, False])
              .drop_duplicates("Rndrng_NPI", keep="first")
              [["Rndrng_NPI", "Rndrng_Prvdr_Type"]]
        )
        df = df.drop(columns=["Rndrng_Prvdr_Type"]).merge(npi_spec, on="Rndrng_NPI", how="left")

        # (specialty, HCPCS, NPI) services
        agg = (
            df.groupby(["Rndrng_Prvdr_Type", "HCPCS_Cd", "Rndrng_NPI"], observed=True, sort=False)
              ["Tot_Srvcs"].sum()
              .reset_index()
        )
        per_state_parts.append(agg)

        # Track per-specialty NPI set for coverage denominator
        for spec, grp in npi_spec.groupby("Rndrng_Prvdr_Type", observed=True):
            per_state_provider_counts.setdefault(spec, set()).update(grp["Rndrng_NPI"].tolist())

        print(f"  [{state}] rows={len(df):>9,}  npi_spec={len(npi_spec):>7,}  "
              f"(spec,hcpcs,npi)={len(agg):>8,}")

    print(f"Scan complete in {time.time()-t0:.1f}s.  Concatenating...")
    long = pd.concat(per_state_parts, ignore_index=True)
    del per_state_parts

    # Collapse across states: same (specialty, HCPCS, NPI) may appear if a
    # provider billed in multiple states in the dataset.
    long = (
        long.groupby(["Rndrng_Prvdr_Type", "HCPCS_Cd", "Rndrng_NPI"], observed=True, sort=False)
            ["Tot_Srvcs"].sum()
            .reset_index()
    )

    # Specialty -> total provider count (for coverage denominator).
    # Use the already-built set (deduped across states).
    spec_provider_counts = {s: len(ns) for s, ns in per_state_provider_counts.items()}

    # (specialty, HCPCS) aggregates: total services + unique NPIs
    scope = (
        long.groupby(["Rndrng_Prvdr_Type", "HCPCS_Cd"], observed=True, sort=False)
            .agg(n_providers=("Rndrng_NPI", "nunique"),
                 total_services=("Tot_Srvcs", "sum"))
            .reset_index()
            .rename(columns={"Rndrng_Prvdr_Type": "specialty"})
    )
    scope["specialty_total_providers"] = scope["specialty"].map(spec_provider_counts).fillna(0).astype("int64")
    scope["pct_providers"] = np.where(
        scope["specialty_total_providers"] > 0,
        scope["n_providers"] / scope["specialty_total_providers"] * 100.0,
        0.0,
    )

    # Volume share within specialty
    scope["specialty_total_services"] = (
        scope.groupby("specialty", observed=True)["total_services"].transform("sum")
    )
    scope["pct_of_specialty_services"] = np.where(
        scope["specialty_total_services"] > 0,
        scope["total_services"] / scope["specialty_total_services"] * 100.0,
        0.0,
    )

    # Volume cumulative share (sorted descending within specialty)
    scope = scope.sort_values(["specialty", "total_services"], ascending=[True, False])
    scope["cumulative_services_share"] = (
        scope.groupby("specialty", observed=True)["pct_of_specialty_services"].cumsum()
    )

    # In-scope flags
    scope["in_scope_by_coverage"] = scope["pct_providers"] >= coverage_min_pct
    # Volume rule: include all codes up to the cumulative threshold, inclusive of
    # the code that first crosses it. To do that we flag rows where the *prior*
    # cumulative (excluding self) was still under the threshold.
    prior_cum = scope.groupby("specialty", observed=True)["cumulative_services_share"].shift(1).fillna(0.0)
    scope["in_scope_by_volume"] = prior_cum < volume_cumulative_pct
    scope["in_scope"] = scope["in_scope_by_coverage"] | scope["in_scope_by_volume"]

    # Final column dtypes
    scope["n_providers"]              = scope["n_providers"].astype("int32")
    scope["specialty_total_providers"] = scope["specialty_total_providers"].astype("int32")
    scope["total_services"]           = scope["total_services"].astype("float64")
    scope["pct_providers"]            = scope["pct_providers"].astype("float32")
    scope["pct_of_specialty_services"] = scope["pct_of_specialty_services"].astype("float32")
    scope["cumulative_services_share"] = scope["cumulative_services_share"].astype("float32")

    # Summary per specialty
    summary = {
        "thresholds": {
            "coverage_min_pct":      coverage_min_pct,
            "volume_cumulative_pct": volume_cumulative_pct,
        },
        "specialties": [],
    }
    for spec, grp in scope.groupby("specialty", observed=True):
        in_scope_codes   = grp[grp["in_scope"]]
        out_of_scope     = grp[~grp["in_scope"]].sort_values("total_services", ascending=False)
        top_out_of_scope = out_of_scope.head(5)[["HCPCS_Cd", "n_providers", "total_services", "pct_of_specialty_services"]]
        summary["specialties"].append({
            "specialty":             str(spec),
            "n_providers":           int(spec_provider_counts.get(spec, 0)),
            "n_codes_total":         int(len(grp)),
            "n_codes_in_scope":      int(len(in_scope_codes)),
            "n_codes_out_of_scope":  int(len(out_of_scope)),
            "in_scope_services_pct": float(in_scope_codes["pct_of_specialty_services"].sum()),
            "top_out_of_scope":      top_out_of_scope.to_dict(orient="records"),
        })
    summary["specialties"].sort(key=lambda x: -x["n_providers"])

    return scope, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver-dir",            default=DEFAULT_SILVER)
    ap.add_argument("--output-parquet",        default=DEFAULT_PARQUET)
    ap.add_argument("--output-summary",        default=DEFAULT_SUMMARY)
    ap.add_argument("--states",                default="",
                    help="Comma-separated state abbreviations; default: all")
    ap.add_argument("--coverage-min-pct",      type=float, default=1.0,
                    help="Code is in-scope by coverage if billed by >= X%% of specialty's providers")
    ap.add_argument("--volume-cumulative-pct", type=float, default=99.0,
                    help="Include codes until cumulative volume share reaches X%% of specialty's services")
    args = ap.parse_args()

    states = {s.strip().upper() for s in args.states.split(",") if s.strip()} if args.states else None
    os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)

    print(f"silver_dir:             {args.silver_dir}")
    print(f"coverage_min_pct:       {args.coverage_min_pct}")
    print(f"volume_cumulative_pct:  {args.volume_cumulative_pct}")

    scope, summary = build_scopes(
        args.silver_dir, states,
        args.coverage_min_pct, args.volume_cumulative_pct,
    )

    scope.to_parquet(args.output_parquet, index=False, compression="snappy")
    with open(args.output_summary, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # Console summary
    n_spec     = scope["specialty"].nunique()
    n_codes    = len(scope)
    n_in       = int(scope["in_scope"].sum())
    print(f"\nWrote {n_codes:,} (specialty, HCPCS) rows across {n_spec:,} specialties -> {args.output_parquet}")
    print(f"  in_scope:    {n_in:,} ({n_in/n_codes*100:.1f}%)")
    print(f"  out_of_scope:{n_codes - n_in:,} ({(n_codes-n_in)/n_codes*100:.1f}%)")
    print(f"Summary written to {args.output_summary}")

    # Spot-check: cardiology
    card = scope[scope["specialty"].str.lower().str.contains("cardiology", na=False)]
    if not card.empty:
        sp = card["specialty"].iloc[0]
        sub = card[card["specialty"] == sp]
        print(f"\nSpot check ({sp}):")
        print(f"  providers: {int(sub['specialty_total_providers'].iloc[0]):,}")
        print(f"  codes:     {len(sub):,}  in_scope: {int(sub['in_scope'].sum()):,}")
        # Is A9585 (gadobutrol) in scope for cardiology?
        a9585 = sub[sub["HCPCS_Cd"] == "A9585"]
        if not a9585.empty:
            r = a9585.iloc[0]
            print(f"  A9585 (gadobutrol): n_providers={int(r['n_providers']):,}  "
                  f"pct_providers={r['pct_providers']:.3f}%  "
                  f"pct_services={r['pct_of_specialty_services']:.4f}%  "
                  f"in_scope={bool(r['in_scope'])}")


if __name__ == "__main__":
    main()
