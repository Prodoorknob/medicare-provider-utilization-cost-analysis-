"""
scorecard.py -- Phase 4: agent detection scorecard (lead-wise efficiency).

Reports how good the AGENT'S OWN leads are -- ranked by the validated breadth
score (fee-schedule metrics excluded), the same ranking agent.py uses -- against
the government ground truth, with two clearly-separated notions of "accuracy":

  * PREDICTIVE  -- the agent flagged the provider in some year AND a fraud-
                   relevant LEIE/CMS sanction landed in a LATER year. This is the
                   lead-generation success; lead-time = months from the first
                   flag to that sanction.
  * CORROBORATED -- the provider currently appears on a public government list
                   (LEIE/CMS), regardless of timing. A broader "is this a real
                   bad actor" overlap.

Both are sparse (sanctions are rare), so the honest headline is LIFT (precision
normalized by base rate) and LEAD-TIME, with absolute counts always shown.
These are investigative leads for analyst review, never accusations.

Reads:
  local_pipeline/anomaly/flags.parquet
  local_pipeline/anomaly/labels.parquet
  local_pipeline/anomaly/validation/lead_queue.parquet   (the agent ranking)
Writes:
  local_pipeline/anomaly/validation/agent_scorecard.md

Usage:
  python -m anomaly.validation.scorecard
  python -m anomaly.validation.scorecard --top 25
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANOM_DIR = os.path.join(_PROJECT_ROOT, "local_pipeline", "anomaly")
FLAGS    = os.path.join(ANOM_DIR, "flags.parquet")
LABELS   = os.path.join(ANOM_DIR, "labels.parquet")
VAL_DIR  = os.path.join(ANOM_DIR, "validation")
LEAD_Q   = os.path.join(VAL_DIR, "lead_queue.parquet")
OUT      = os.path.join(VAL_DIR, "agent_scorecard.md")

FEE_SCHEDULE_METRICS = {"charge_to_allowed_ratio", "avg_allowed"}
KS = [10, 25, 50, 100, 250, 500, 1000]


def _lead_stats(days: np.ndarray) -> dict:
    d = np.asarray([x for x in days if x is not None and not np.isnan(x)], dtype=float)
    d = d[d > 0]
    if d.size == 0:
        return {"n": 0}
    return {"n": int(d.size),
            "median": float(np.median(d)) / 30.44,
            "p25": float(np.percentile(d, 25)) / 30.44,
            "p75": float(np.percentile(d, 75)) / 30.44,
            "min": float(d.min()) / 30.44,
            "max": float(d.max()) / 30.44}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=25, help="rows in the per-lead detail table")
    args = ap.parse_args()

    lq = pd.read_parquet(LEAD_Q).sort_values("review_priority").reset_index(drop=True)
    lq["npi"] = lq["npi"].astype(str)

    # first flag year per provider on the AGENT's signals (fee-schedule excluded).
    flags = pd.read_parquet(FLAGS, columns=["Rndrng_NPI", "year", "flag_metric"])
    flags["npi"] = flags["Rndrng_NPI"].astype("int64").astype(str)
    flags = flags[~flags["flag_metric"].isin(FEE_SCHEDULE_METRICS)]
    first_year = flags.groupby("npi")["year"].min().rename("first_flag_year")

    # earliest fraud-relevant government sanction per provider.
    labels = pd.read_parquet(LABELS)
    labels["npi"] = labels["npi"].astype(str)
    fr = labels[labels["fraud_relevant"]].sort_values("event_date") \
        .drop_duplicates("npi", keep="first").set_index("npi")

    lq = lq.merge(first_year, on="npi", how="left")
    lq["event_date"] = lq["npi"].map(fr["event_date"])
    lq["event_year"] = lq["npi"].map(fr["event_year"])
    lq["sanction_source"] = lq["npi"].map(fr["source"])

    # PREDICTIVE hit: a fraud-relevant sanction in a year AFTER the first flag.
    lq["predictive_hit"] = lq["event_year"].notna() & (lq["event_year"] > lq["first_flag_year"])
    cutoff = pd.to_datetime(dict(year=lq["first_flag_year"].fillna(0).astype(int),
                                 month=12, day=31), errors="coerce")
    lq["lead_days"] = np.where(lq["predictive_hit"], (lq["event_date"] - cutoff).dt.days, np.nan)

    n = len(lq)
    base = float(lq["predictive_hit"].mean())
    corr_base = float(lq["govt_corroborated"].mean())
    pos_arr = lq["predictive_hit"].to_numpy()
    corr_arr = lq["govt_corroborated"].to_numpy()

    # ---- per-k metrics ----
    rows = []
    for k in KS:
        ke = min(k, n)
        tp = int(pos_arr[:ke].sum())
        prec = tp / ke
        lift = prec / base if base else float("nan")
        ctp = int(corr_arr[:ke].sum())
        lt = _lead_stats(lq["lead_days"].to_numpy()[:ke])
        rows.append({"k": ke, "tp": tp, "prec": prec, "lift": lift,
                     "reviews_per_find": (ke / tp) if tp else float("inf"),
                     "corr": ctp, "corr_rate": ctp / ke,
                     "lead_med": lt.get("median")})

    overall_lt = _lead_stats(lq["lead_days"].to_numpy())
    om = overall_lt
    if om.get("n"):
        lead_all_line = (f"median **{om['median']:.1f} mo**, IQR {om['p25']:.0f}–{om['p75']:.0f} mo, "
                         f"range {om['min']:.1f}–{om['max']:.1f} mo.")
        lead_med_int = f"{om['median']:.0f}"
    else:
        lead_all_line = "no predictive hits in cohort."
        lead_med_int = "0"

    # ---- per-lead detail (top N) ----
    head = lq.head(args.top)
    detail = ["| # | NPI | specialty | state | peak yr | signals | govt status (present-day) | predictive | lead (mo) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in head.iterrows():
        status = "—"
        if r["govt_corroborated"]:
            bits = []
            if r["on_leie"]:
                bits.append(f"LEIE {r['leie_excltype']} {str(r['leie_excldate'])}".strip())
            if r["on_cms_revoked"]:
                bits.append(f"CMS {r['cms_authority']} {str(r['cms_revocation_date'])[:10]}".strip())
            status = "; ".join(bits)
        pred = "—"
        if r["predictive_hit"]:
            pred = f"{r['sanction_source']}"
        lead = f"{r['lead_days']/30.44:.0f}" if (isinstance(r["lead_days"], (int, float)) and not np.isnan(r["lead_days"])) else "—"
        detail.append(
            f"| {int(r['review_priority'])} | {r['masked_npi']} | {r['specialty']} | {r['state']} | "
            f"{int(r['peak_year'])} | {int(r['n_metrics'])} | {status} | {pred} | {lead} |")

    # ---- markdown ----
    def kt():
        out = ["| top-k | predictive TP | precision | **lift** | reviews/find | corroborated (rate) | lead-time med |",
               "|---|---|---|---|---|---|---|"]
        for r in rows:
            lead = f"{r['lead_med']:.0f} mo" if r["lead_med"] else "—"
            rpf = f"~{r['reviews_per_find']:.0f}" if np.isfinite(r["reviews_per_find"]) else "—"
            out.append(f"| {r['k']} | {r['tp']} | {r['prec']*100:.2f}% | **{r['lift']:.1f}×** | "
                       f"{rpf} | {r['corr']} ({r['corr_rate']*100:.1f}%) | {lead} |")
        return "\n".join(out)

    md = f"""# Agent Detection Scorecard

_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}._

> ⚠️ **RETROSPECTIVE / IN-SAMPLE — superseded for honest numbers by
> `anomaly/validation/point_in_time.py`.** This ranking aggregates each
> provider's *entire* 2013–2023 flag panel (look-ahead) and the score was
> selected against these same labels (no holdout), so the small-k lift here is
> optimistic and rests on 1–2 hits. The leakage-corrected, held-out,
> volume-stratified evaluation (point_in_time_report.md) is authoritative.
> Cohort here = the agent's fee-schedule-excluded ranking ({n:,} providers);
> run_validation REPORT.md ranks on all metrics (~391,411) for the ablation contrast.

Ranking = the agent's validated **breadth-of-signal** score (fee-schedule
metrics excluded), one lead per provider. Cohort = **{n:,}** flagged providers.
Two accuracy notions, both sparse by nature (read **lift**, not raw precision):

- **Predictive** — flagged in a year *before* a fraud-relevant LEIE/CMS sanction.
  Base rate **{base*100:.3f}%** ({int(lq['predictive_hit'].sum())} of {n:,}).
- **Corroborated** — currently on a public LEIE/CMS list (any timing).
  Base rate **{corr_base*100:.3f}%** ({int(lq['govt_corroborated'].sum())} of {n:,}).

## Detection metrics by review depth (k)
- **lift** = precision@k ÷ base rate (how many × better than screening random flags).
- **reviews/find** = leads an analyst reviews to surface one eventual sanction.
{kt()}

## Lead-time (predictive hits only)
- **All hits** (n={overall_lt.get('n')}): {lead_all_line}
- Interpretation: when a flagged provider *is* later sanctioned, the agent surfaced
  them a median of ~{lead_med_int} months ahead of the government action.

## Top {args.top} leads (the current briefs) — lead-wise
{chr(10).join(detail)}

## How to read this
- Precision is low in absolute terms because sanctioned providers are <0.3% of
  any cohort — that is the base rate, not a detector failure. **Lift** is the
  honest "accuracy vs random," and **lead-time** is the product value.
- "Predictive —" with a present-day govt status means the sanction predates the
  first flag (corroborating the provider, but not an early-warning win).
- Leads are statistical outliers for **analyst review**, never accusations.
"""
    os.makedirs(VAL_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(md)

    print(f"[scorecard] cohort={n:,}  predictive base={base*100:.3f}%  corroborated base={corr_base*100:.3f}%")
    for r in rows:
        if r["k"] in (10, 25, 100, 1000):
            print(f"  top-{r['k']:>4}: TP={r['tp']:>3} prec={r['prec']*100:5.2f}% lift={r['lift']:4.1f}x "
                  f"corrob={r['corr']:>3} lead_med={r['lead_med'] or 0:.0f}mo")
    print(f"  overall lead-time median={overall_lt.get('median', 0):.1f}mo (n={overall_lt.get('n')})")
    print(f"[scorecard] -> {OUT}")


if __name__ == "__main__":
    main()
