"""
build_lead_queue.py -- Phase 4 (C): produce the operational analyst lead queue.

This is the analyst-facing output of the fraud-lead workflow: a ranked,
evidence-linked list of providers for a HUMAN reviewer to triage. It is framed
as REVIEW PRIORITIES / investigative leads, never as accusations of fraud.

Difference from run_validation.py (important):
  - run_validation BACKTESTS the ranking using *future* sanctions (a label that
    only exists in hindsight) to measure lift/lead-time.
  - build_lead_queue is OPERATIONAL: it ranks providers by the validated
    breadth-of-signal score with NO future-label leakage, and annotates each
    provider's PRESENT-DAY public government status (currently on the OIG LEIE
    or the CMS Revoked list) purely as corroborating context.

Ranking = number of distinct (non-fee-schedule) anomaly metrics a provider
trips across years, tiebroken by total severity mass -- the ranking the Phase 4
backtest validated (~4-8x lift at the top vs ~0x for severity-sum). Each lead is
anchored to the provider's single most-anomalous year.

Outputs (local_pipeline/anomaly/validation/):
  lead_queue.parquet   -- full ranked queue with evidence + govt status
  lead_queue.csv       -- same, analyst-readable
  lead_queue_top.md    -- top-N preview (NPIs masked), analyst-framed

Usage:
  python -m anomaly.validation.build_lead_queue
  python -m anomaly.validation.build_lead_queue --top 200 --mask   # drop full NPI for sharing
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANOM_DIR = os.path.join(_PROJECT_ROOT, "local_pipeline", "anomaly")
FLAGS    = os.path.join(ANOM_DIR, "flags.parquet")
LEIE     = os.path.join(ANOM_DIR, "leie_exclusions.parquet")
CMS      = os.path.join(ANOM_DIR, "cms_revoked.parquet")
OUT_DIR  = os.path.join(ANOM_DIR, "validation")

# Negatively-predictive PFS-derived metrics, excluded from the score (see backtest).
FEE_SCHEDULE_METRICS = {"charge_to_allowed_ratio", "avg_allowed"}


def mask_npi(npi: str) -> str:
    s = str(npi)
    return s[:4] + "****" + s[-2:] if len(s) >= 10 else "**********"


def _valid_npi(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    return s.str.len().ge(10) & s.ne("0") & ~s.str.fullmatch(r"0+")


def build_queue(flags: pd.DataFrame, include_fee: bool = False) -> pd.DataFrame:
    scored = flags if include_fee else flags[~flags["flag_metric"].isin(FEE_SCHEDULE_METRICS)]
    agg = (scored.groupby("Rndrng_NPI")
           .agg(n_metrics=("flag_metric", "nunique"),
                n_methods=("flag_type", "nunique"),
                severity_sum=("severity", "sum"),
                severity_max=("severity", "max"),
                n_flags=("severity", "size"),
                n_years=("year", "nunique"),
                specialty=("specialty", "first"),
                state=("state", "first"),
                metrics_tripped=("flag_metric", lambda x: ", ".join(sorted(set(x)))),
                methods=("flag_type", lambda x: ", ".join(sorted(set(x)))))
           .reset_index())
    agg["rank_score"] = agg["n_metrics"] * 1000.0 + agg["severity_sum"]
    agg = agg.sort_values(["rank_score", "n_flags"], ascending=[False, False]).reset_index(drop=True)
    agg["review_priority"] = agg.index + 1

    # Anchor each lead to its single most-anomalous year.
    top = set(agg["Rndrng_NPI"])
    peak = (scored[scored["Rndrng_NPI"].isin(top)]
            .groupby(["Rndrng_NPI", "year"])["severity"].sum().reset_index()
            .sort_values("severity", ascending=False).drop_duplicates("Rndrng_NPI"))
    agg = agg.merge(peak[["Rndrng_NPI", "year"]].rename(columns={"year": "peak_year"}),
                    on="Rndrng_NPI", how="left")
    return agg


def annotate_govt_status(queue: pd.DataFrame) -> pd.DataFrame:
    queue = queue.copy()
    queue["npi"] = queue["Rndrng_NPI"].astype("int64").astype(str)

    # Present-day OIG LEIE status (current active-exclusion snapshot).
    if os.path.exists(LEIE):
        leie = pd.read_parquet(LEIE)
        leie = leie[_valid_npi(leie["NPI"])].copy()
        leie["npi"] = leie["NPI"].astype(str).str.strip()
        leie = leie.sort_values("EXCLDATE").drop_duplicates("npi", keep="last")
        lm = leie.set_index("npi")
        queue["on_leie"] = queue["npi"].isin(lm.index)
        queue["leie_excltype"] = queue["npi"].map(lm["EXCLTYPE"]).fillna("")
        queue["leie_excldate"] = queue["npi"].map(lm["EXCLDATE"]).fillna("")
    else:
        queue["on_leie"] = False; queue["leie_excltype"] = ""; queue["leie_excldate"] = ""

    # Present-day CMS Revoked status.
    if os.path.exists(CMS):
        cms = pd.read_parquet(CMS)
        cms = cms[_valid_npi(cms["NPI"])].copy()
        cms["npi"] = cms["NPI"].astype(str).str.strip()
        cms = cms.sort_values("REVOCATION_EFCTV_DT").drop_duplicates("npi", keep="last")
        cm = cms.set_index("npi")
        queue["on_cms_revoked"] = queue["npi"].isin(cm.index)
        queue["cms_authority"] = queue["npi"].map(cm["AUTHORITY_CODES"]).fillna("")
        queue["cms_revocation_date"] = queue["npi"].map(cm["REVOCATION_EFCTV_DT"]).astype(str).replace("NaT", "")
    else:
        queue["on_cms_revoked"] = False; queue["cms_authority"] = ""; queue["cms_revocation_date"] = ""

    queue["govt_corroborated"] = queue["on_leie"] | queue["on_cms_revoked"]
    queue["masked_npi"] = queue["npi"].map(mask_npi)
    return queue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=25, help="rows in the markdown preview")
    ap.add_argument("--mask", action="store_true", help="drop full NPI column (share-safe output)")
    ap.add_argument("--include-fee-schedule", action="store_true")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Building operational lead queue ...")
    flags = pd.read_parquet(FLAGS, columns=[
        "Rndrng_NPI", "year", "specialty", "state", "flag_type", "flag_metric", "severity"])
    q = build_queue(flags, include_fee=args.include_fee_schedule)
    q = annotate_govt_status(q)

    cols = ["review_priority", "masked_npi", "npi", "specialty", "state", "peak_year",
            "n_metrics", "n_methods", "n_years", "n_flags", "severity_sum", "severity_max",
            "metrics_tripped", "methods",
            "govt_corroborated", "on_leie", "leie_excltype", "leie_excldate",
            "on_cms_revoked", "cms_authority", "cms_revocation_date"]
    q = q[cols]
    if args.mask:
        q = q.drop(columns=["npi"])

    q.to_parquet(os.path.join(args.out_dir, "lead_queue.parquet"), index=False)
    q.to_csv(os.path.join(args.out_dir, "lead_queue.csv"), index=False)

    # Markdown preview, analyst-framed, NPIs masked regardless of --mask.
    head = q.head(args.top)
    lines = [
        "# Provider Review Queue — investigative leads (NOT accusations)",
        f"\n_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}._\n",
        "Each row is a **statistical outlier relative to specialty/state peers that "
        "warrants analyst review** — it is NOT a finding of wrongdoing; the large "
        "majority of outliers have legitimate explanations. `govt corroboration` shows "
        "whether the provider *currently* appears on a public government list, stated as "
        "bare public fact.\n",
        "| priority | NPI | specialty | state | peak yr | distinct signals | yrs | govt corroboration |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in head.iterrows():
        corro = "—"
        if r["govt_corroborated"]:
            bits = []
            if r["on_leie"]:
                bits.append(f"LEIE {r['leie_excltype']} {r['leie_excldate']}".strip())
            if r["on_cms_revoked"]:
                bits.append(f"CMS {r['cms_authority']} {r['cms_revocation_date']}".strip())
            corro = "; ".join(bits)
        lines.append(
            f"| {int(r['review_priority'])} | {r['masked_npi']} | {r['specialty']} | {r['state']} | "
            f"{int(r['peak_year'])} | {int(r['n_metrics'])} ({r['methods']}) | {int(r['n_years'])} | {corro} |")
    n_corro = int(q["govt_corroborated"].sum())
    lines.append(
        f"\n_{len(q):,} providers queued; {n_corro:,} currently appear on a public "
        f"government list (LEIE/CMS). Ranking is by breadth of distinct anomaly signals "
        f"(fee-schedule metrics excluded), validated at ~4-8× lift at the top against "
        f"later sanctions. Leads are for authorized analyst review only._")
    with open(os.path.join(args.out_dir, "lead_queue_top.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"  queued providers: {len(q):,}  |  currently govt-corroborated: {n_corro:,}")
    print(f"  -> {os.path.join(args.out_dir, 'lead_queue.csv')}")
    print(f"  -> {os.path.join(args.out_dir, 'lead_queue_top.md')}")
    print("\nTop 5 leads:")
    for _, r in q.head(5).iterrows():
        print(f"  #{int(r['review_priority'])} {r['masked_npi']} {r['specialty']} {r['state']} "
              f"peak={int(r['peak_year'])} signals={int(r['n_metrics'])} "
              f"govt={'YES' if r['govt_corroborated'] else 'no'}")


if __name__ == "__main__":
    main()
