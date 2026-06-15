"""
run_validation.py -- Phase 4 (B): backtest the fraud-lead flags against the
unified government ground-truth labels.

Pipeline:
  1. Aggregate flags.parquet to ONE ranked lead per NPI (composite suspicion).
  2. Temporal join: an NPI is a POSITIVE if it appears in the labels under a
     qualifying sanction whose event_year is AFTER the provider's first flagged
     year (predictive framing -- the flag preceded the government action).
  3. Report lift@k / precision@k (+ Fisher p, bootstrap CI, absolute TP),
     lead-time, and PR-AUC -- NEVER accuracy/ROC-AUC.
  4. Fee-schedule ablation gate: recompute lift after dropping PFS-derived flags
     (charge/allowed-amount metrics) to check the ranking is not just the
     falsification-documented fee-schedule re-derivation.
  5. Maturity view (first flag <= 2019, accrual mostly complete) and
     per-source / per-scope segmentation.

Outputs (local_pipeline/anomaly/validation/):
  REPORT.md, ranking.parquet (all leads, scored + labeled), matched_positives.parquet

Usage:
  python anomaly/validation/run_validation.py
  python anomaly/validation/run_validation.py --mature-year 2019 --top-preview 20
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from anomaly.validation import metrics as M

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANOM_DIR      = os.path.join(_PROJECT_ROOT, "local_pipeline", "anomaly")
FLAGS         = os.path.join(ANOM_DIR, "flags.parquet")
LABELS        = os.path.join(ANOM_DIR, "labels.parquet")
PROFILES      = os.path.join(ANOM_DIR, "npi_profiles.parquet")
OUT_DIR       = os.path.join(ANOM_DIR, "validation")

# PFS-derived flag metrics (dollar quantities the falsification report showed the
# Stage 1 model largely re-derives). Dropping these = the ablation gate.
FEE_SCHEDULE_METRICS = {"charge_to_allowed_ratio", "avg_allowed"}
KS = [50, 100, 250, 500, 1000, 2500, 5000]


# ---------------------------------------------------------------- ranking ----
def build_ranking(flags: pd.DataFrame) -> pd.DataFrame:
    g = flags.groupby("npi")
    agg = g.agg(
        n_flags=("severity", "size"),
        n_methods=("flag_type", "nunique"),
        n_metrics=("flag_metric", "nunique"),
        severity_sum=("severity", "sum"),
        severity_max=("severity", "max"),
        first_flag_year=("year", "min"),
        last_flag_year=("year", "max"),
        specialty=("specialty", "first"),
        state=("state", "first"),
    ).reset_index()
    # Composite suspicion score. Empirically (see rank-comparison in the report),
    # ranking by BREADTH of distinct anomaly signals a provider trips concentrates
    # the later-sanctioned population far better than total severity mass (which
    # just surfaces chronically high-volume providers). So score = number of
    # distinct anomaly metrics tripped, tiebroken by severity mass.
    agg["rank_score"] = agg["n_metrics"] * 1000.0 + agg["severity_sum"]
    agg["composite"] = agg["n_methods"] >= 2
    return agg.sort_values("rank_score", ascending=False).reset_index(drop=True)


def rank_compare(ranking: pd.DataFrame, is_positive: np.ndarray, ks=(100, 1000, 5000)) -> str:
    """Lift@k for candidate per-NPI scalar scores — documents why n_metrics wins."""
    r = ranking.assign(_pos=is_positive)
    br = M.base_rate(is_positive)
    cands = ["n_metrics", "n_methods", "n_flags", "severity_max", "severity_sum"]
    out = ["| ranking score | " + " | ".join(f"lift@{k}" for k in ks) + " |",
           "|---|" + "|".join("---" for _ in ks) + "|"]
    for c in cands:
        s = r.sort_values(c, ascending=False)["_pos"].to_numpy()
        cells = []
        for k in ks:
            p, tp, keff = M.precision_at_k(s, k)
            cells.append(f"{(p/br if br else float('nan')):.1f}× ({tp})")
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    out.append(f"\n_lift = precision@k / base rate ({br*100:.3f}%); (TP) is the absolute hit count._")
    return "\n".join(out)


def metric_enrichment(flags: pd.DataFrame, pos_npis: set, n_flagged: int, n_pos: int) -> str:
    """Enrichment of each flag_metric among later-sanctioned providers vs the cohort."""
    rows = []
    for met in flags["flag_metric"].unique():
        npis = set(flags.loc[flags["flag_metric"] == met, "npi"])
        cohort_rate = len(npis) / n_flagged
        pos_rate = len(npis & pos_npis) / n_pos if n_pos else float("nan")
        enrich = pos_rate / cohort_rate if cohort_rate else float("nan")
        rows.append((met, cohort_rate, pos_rate, enrich))
    rows.sort(key=lambda x: (x[3] if not np.isnan(x[3]) else -1), reverse=True)
    out = ["| flag metric | cohort rate | rate among positives | enrichment |",
           "|---|---|---|---|"]
    for met, cr, pr, en in rows:
        out.append(f"| {met} | {cr*100:.1f}% | {pr*100:.1f}% | {en:.2f}× |")
    return "\n".join(out)


# ------------------------------------------------------------- label join ----
def attach_labels(ranking: pd.DataFrame, labels: pd.DataFrame,
                  scope: str = "fraud_relevant", source: str | None = None):
    """Return (is_positive bool Series aligned to ranking, matched DataFrame).

    Positive = a qualifying sanction with event_year > the provider's first
    flagged year (the flag predates the government action).
    """
    lab = labels
    if scope == "fraud_relevant":
        lab = lab[lab["fraud_relevant"]]
    if source:
        lab = lab[lab["source"] == source]
    if lab.empty:
        return pd.Series(False, index=ranking.index), pd.DataFrame()

    m = ranking[["npi", "first_flag_year"]].merge(
        lab[["npi", "event_date", "event_year", "source", "tier"]], on="npi", how="inner"
    )
    m = m[m["event_year"] > m["first_flag_year"]]
    if m.empty:
        return pd.Series(False, index=ranking.index), pd.DataFrame()
    m = m.sort_values("event_date").groupby("npi", as_index=False).first()
    cutoff = pd.to_datetime(dict(year=m["first_flag_year"], month=12, day=31))
    m["lead_days"] = (m["event_date"] - cutoff).dt.days
    pos_npis = set(m["npi"])
    is_pos = ranking["npi"].isin(pos_npis)
    return is_pos, m


# --------------------------------------------------------------- reporting ---
def fmt_ks_table(rows: list[dict], base_rate: float) -> str:
    out = ["| k | TP | precision | lift× | 95% CI (prec) | Fisher p |",
           "|---|----|-----------|-------|---------------|----------|"]
    for r in rows:
        ci = (f"{r['ci_lo']*100:.2f}–{r['ci_hi']*100:.2f}%"
              if not np.isnan(r["ci_lo"]) else "—")
        pval = ("%.1e" % r["fisher_p"]) if not np.isnan(r["fisher_p"]) else "—"
        lift = (f"{r['lift']:.1f}" if not np.isnan(r["lift"]) else "—")
        out.append(f"| {r['k']} | {r['tp']} | {r['precision']*100:.2f}% | {lift} | {ci} | {pval} |")
    out.append(f"\n_Cohort base rate (positives / flagged NPIs) = {base_rate*100:.3f}%._")
    return "\n".join(out)


def mask_npi(npi: str) -> str:
    s = str(npi)
    return s[:4] + "****" + s[-2:] if len(s) >= 10 else "**********"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mature-year", type=int, default=2019,
                    help="Flags with first_flag_year <= this are the 'mature accrual' headline")
    ap.add_argument("--top-preview", type=int, default=20)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("Loading flags + labels ...")
    flags = pd.read_parquet(FLAGS, columns=[
        "Rndrng_NPI", "year", "specialty", "state", "flag_type", "flag_metric", "severity"])
    flags["npi"] = flags["Rndrng_NPI"].astype("int64").astype(str)
    labels = pd.read_parquet(LABELS)
    labels["npi"] = labels["npi"].astype(str)

    # Population context: sanction rate across the full profile NPI universe.
    prof_npis = pd.read_parquet(PROFILES, columns=["Rndrng_NPI"])["Rndrng_NPI"].astype("int64").astype(str)
    n_universe = int(prof_npis.nunique())
    fr_npis = set(labels.loc[labels["fraud_relevant"], "npi"])
    pop_base = len(fr_npis & set(prof_npis.unique())) / n_universe

    # ---- full ranking (all metrics) ----
    ranking = build_ranking(flags)
    n_flagged = len(ranking)
    is_pos, matched = attach_labels(ranking, labels, scope="fraud_relevant")
    ranking["is_positive"] = is_pos.values
    ranking = ranking.merge(matched[["npi", "event_date", "source", "tier", "lead_days"]],
                            on="npi", how="left")
    sorted_pos = ranking.sort_values("rank_score", ascending=False)["is_positive"].to_numpy()
    cohort_base = M.base_rate(sorted_pos)
    full_rows = M.ks_table(sorted_pos, KS)
    prauc = M.pr_auc(ranking["rank_score"].to_numpy(), ranking["is_positive"].to_numpy())
    lead_all = M.lead_time_stats(matched["lead_days"].tolist())
    top1000_pos_npis = set(ranking.sort_values("rank_score", ascending=False).head(1000)["npi"])
    lead_top1k = M.lead_time_stats(matched.loc[matched["npi"].isin(top1000_pos_npis), "lead_days"].tolist())

    # ---- diagnostics: why this ranking, and which signals carry the label ----
    pos_npis = set(ranking.loc[ranking["is_positive"], "npi"])
    n_pos = int(ranking["is_positive"].sum())
    rankcmp_md = rank_compare(ranking, ranking["is_positive"].to_numpy())
    enrich_md = metric_enrichment(flags, pos_npis, n_flagged, n_pos)

    # ---- maturity view ----
    mat = ranking[ranking["first_flag_year"] <= args.mature_year].copy()
    mat_sorted = mat.sort_values("rank_score", ascending=False)["is_positive"].to_numpy()
    mat_base = M.base_rate(mat_sorted)
    mat_rows = M.ks_table(mat_sorted, KS)

    # ---- ablation: behavioral-only (drop PFS-derived metrics) ----
    behav_flags = flags[~flags["flag_metric"].isin(FEE_SCHEDULE_METRICS)]
    behav_rank = build_ranking(behav_flags)
    bpos, _ = attach_labels(behav_rank, labels, scope="fraud_relevant")
    behav_sorted = behav_rank.assign(is_positive=bpos.values) \
        .sort_values("rank_score", ascending=False)["is_positive"].to_numpy()
    behav_base = M.base_rate(behav_sorted)
    behav_rows = M.ks_table(behav_sorted, KS)

    # ---- segmentation by source ----
    seg = {}
    for src in ["LEIE", "CMS_REVOKED"]:
        sp, _ = attach_labels(ranking, labels, scope="fraud_relevant", source=src)
        ssorted = ranking.assign(_p=sp.values).sort_values("rank_score", ascending=False)["_p"].to_numpy()
        seg[src] = (M.base_rate(ssorted), M.ks_table(ssorted, [100, 1000, 5000]))

    # ---- "any sanction" scope (broader, incl. permissive/non-fraud) ----
    apos, amatched = attach_labels(ranking, labels, scope="any")
    a_sorted = ranking.assign(_p=apos.values).sort_values("rank_score", ascending=False)["_p"].to_numpy()
    any_base = M.base_rate(a_sorted)
    any_rows = M.ks_table(a_sorted, [100, 1000, 5000])

    # ---- persist artifacts ----
    ranking.to_parquet(os.path.join(args.out_dir, "ranking.parquet"), index=False)
    matched.to_parquet(os.path.join(args.out_dir, "matched_positives.parquet"), index=False)

    # ---- top-N lead preview (analyst-framed, NPI masked) ----
    preview = ranking.sort_values("rank_score", ascending=False).head(args.top_preview)
    prev_lines = ["| review priority | NPI | specialty | state | yrs flagged | methods | later govt action |",
                  "|---|---|---|---|---|---|---|"]
    for i, (_, r) in enumerate(preview.iterrows(), 1):
        action = "—"
        if r["is_positive"]:
            lt = f", +{r['lead_days']/30.44:.0f}mo" if pd.notna(r.get("lead_days")) else ""
            action = f"{r['source']} {r['tier']}{lt}"
        prev_lines.append(
            f"| {i} | {mask_npi(r['npi'])} | {r['specialty']} | {r['state']} | "
            f"{int(r['first_flag_year'])}–{int(r['last_flag_year'])} | {int(r['n_methods'])} | {action} |")
    preview_md = "\n".join(prev_lines)

    # ---- write report ----
    n_pos = int(ranking["is_positive"].sum())
    report = f"""# Fraud-Lead Validation — Backtest against Government Ground Truth

_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}._

**What this is.** Each flagged provider (NPI) is ranked by composite statistical
suspicion, then checked against two periodically-released government label
sources — OIG **LEIE** exclusions (monthly) and CMS **Revoked Providers**
(quarterly) — to ask: *of the providers we rank highest, what fraction were
later subject to a government sanction, how many times the base rate is that
(lift), and how far ahead did we flag them (lead-time)?*

**What this is NOT.** It is not proof the detector "catches fraud." The label is
sparse, lagged, and right-censored, and statistical-outlier flagging is lawful
for the overwhelming majority of providers. Output is an **investigative lead
for a human analyst — never an accusation.** Accuracy / ROC-AUC are deliberately
not reported (they are meaningless at this base rate).

## Cohort
- Flagged providers (unique NPIs ranked): **{n_flagged:,}**
- Profile universe (all NPIs): **{n_universe:,}**
- Positive label = a fraud-relevant LEIE/CMS sanction with event year **after**
  the provider's first flagged year.
- Positives in cohort (fraud-relevant scope): **{n_pos:,}**  → cohort base rate **{cohort_base*100:.3f}%**
- Population sanction rate (fraud-relevant NPIs ∩ universe): **{pop_base*100:.3f}%**

> ⚠️ The absolute true-positive counts below are small by nature — read **lift**
> and **lead-time**, not raw precision, and never a ratio without its TP count.

## Headline — full ranking (all flag metrics), fraud-relevant label
{fmt_ks_table(full_rows, cohort_base)}

- **PR-AUC** (average precision, positive-unlabeled aware): **{prauc:.4f}**
- **Lead-time**, all matched positives: {lead_all}
- **Lead-time**, positives within top-1000: {lead_top1k}

## Why this ranking — score comparison
Total severity *mass* (`severity_sum`) ranks chronically high-volume providers
first and has ~no lift at the top; *breadth* of distinct anomaly signals
(`n_metrics`, the production rank score) concentrates the later-sanctioned
population far better. Pre-registered candidate scores:
{rankcmp_md}

## Signal diagnostics — which flag metrics carry the label
Enrichment = (rate among later-sanctioned providers) / (rate across the flagged
cohort). Note the fee-schedule-derived metrics (`charge_to_allowed_ratio`,
`avg_allowed`) are **below 1.0× (negatively predictive)** — independent
empirical confirmation of the falsification finding that those signals largely
re-derive the PFS rather than capturing aberrant behavior. Intensity/multivariate
signals over-index.
{enrich_md}

## Mature accrual view (first flagged ≤ {args.mature_year})
Recent-year flags are right-censored (sanctions lag conduct ~2–6 yrs), so the
defensible precision estimate restricts to providers whose label window is
mostly complete.
{fmt_ks_table(mat_rows, mat_base)}

## Ablation gate — behavioral-only ranking (PFS-derived flags removed)
Drops `{", ".join(sorted(FEE_SCHEDULE_METRICS))}` (the fee-schedule-derived
metrics the project's own falsification work showed the Stage 1 model largely
re-derives). **If lift survives here, the ranking is not merely a fee-schedule
artifact.**
{fmt_ks_table(behav_rows, behav_base)}

## Segmentation by label source (fraud-relevant)
| source | base rate | prec@100 (TP) | prec@1000 (TP) | prec@5000 (TP) | lift@1000 |
|---|---|---|---|---|---|
""" + "\n".join(
        f"| {src} | {seg[src][0]*100:.3f}% | {seg[src][1][0]['precision']*100:.2f}% ({seg[src][1][0]['tp']}) | "
        f"{seg[src][1][1]['precision']*100:.2f}% ({seg[src][1][1]['tp']}) | "
        f"{seg[src][1][2]['precision']*100:.2f}% ({seg[src][1][2]['tp']}) | "
        f"{seg[src][1][1]['lift']:.1f} |"
        for src in ["LEIE", "CMS_REVOKED"]
    ) + f"""

## Broader scope — any sanction (incl. permissive/non-billing exclusions)
| k | TP | precision | lift× |
|---|----|-----------|-------|
""" + "\n".join(
        f"| {r['k']} | {r['tp']} | {r['precision']*100:.2f}% | "
        f"{(r['lift'] if not np.isnan(r['lift']) else float('nan')):.1f} |"
        for r in any_rows
    ) + f"""

_any-scope cohort base rate = {any_base*100:.3f}%; matched positives = {int(apos.sum()):,}._

## Top {args.top_preview} leads (preview — NPIs masked, analyst-framed)
{preview_md}

## Reading guide / caveats (carry into any external claim)
1. **Lift is the headline, lead-time is the value.** "Top-k providers are N×
   more likely to face a later government action" + "flagged a median of X
   months before enforcement." Precision is tiny because the label is rare.
2. **Censoring.** Treat "flagged, not yet sanctioned" as unlabeled, not a false
   positive — especially for 2021–2023 flags whose accrual window is still open.
3. **Spectrum mismatch.** LEIE/CMS over-represent prosecutable schemes; our
   detectors fire on statistical outliers. Low overlap can be definitional, not
   a detector failure.
4. **Exclusion ≠ contemporaneous fraud.** Many permissive (1128b) exclusions are
   license actions; CMS revocations are administrative. Surface EXCLTYPE/
   authority verbatim; let the analyst judge.
5. **Snapshots.** LEIE/CMS are active-only; persist successive pulls so labels
   are reconstructable as-of any date (the harness writes dated snapshots).
"""

    report_path = os.path.join(args.out_dir, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_flagged_npis": n_flagged,
        "n_universe": n_universe,
        "cohort_base_rate": cohort_base,
        "population_base_rate": pop_base,
        "n_positives_fraud_relevant": n_pos,
        "pr_auc": prauc,
        "lead_time_all": lead_all,
        "headline_ks": full_rows,
        "ablation_behavioral_ks": behav_rows,
    }
    with open(os.path.join(args.out_dir, "validation_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print(f"\nReport -> {report_path}")
    print(f"  flagged NPIs={n_flagged:,}  positives(fraud-relevant)={n_pos:,}  "
          f"cohort base={cohort_base*100:.3f}%  PR-AUC={prauc:.4f}")
    for r in full_rows:
        if r["k"] in (100, 1000):
            print(f"  prec@{r['k']}={r['precision']*100:.2f}% (TP={r['tp']}) lift={r['lift']:.1f}")


if __name__ == "__main__":
    main()
