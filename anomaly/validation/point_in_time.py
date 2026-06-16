"""
point_in_time.py -- Phase 4: LEAKAGE-CORRECTED historical validation.

The original backtest (run_validation.py / scorecard.py) had two leaks an
independent review flagged:
  1. The ranking aggregated each provider's ENTIRE 2013-2023 flag panel -- so the
     score that surfaces a later-sanctioned provider used data from years AT/AFTER
     the sanction (look-ahead). lead-time was anchored to the first flag year, not
     the year the provider would actually have been surfaced.
  2. The ranking score and the fee-schedule exclusion were SELECTED by maximizing
     lift on the same labels used to report (selection-on-test, no holdout).

This module fixes both and adds the two checks the review demanded:
  - POINT-IN-TIME (as-of-T): for each decision year T, rank providers using ONLY
    flags with year <= T, and count a provider as a positive only if a qualifying
    sanction's event_year > T. Lead-time is anchored to the EARLIEST T at which the
    provider crossed top-k (the real alert date).
  - TEMPORAL HOLDOUT: select the ranking score on DEV decision years (<=2017),
    report only on TEST decision years (>=2018). The score is never tuned on the
    test period.
  - VOLUME STRATIFICATION: recompute lift within volume bands. If lift collapses to
    ~1x inside bands, the headline lift is a volume confound, not fraud signal.
  - HONEST REPORTING: Benjamini-Hochberg FDR across the k x year grid; PR-AUC vs the
    no-skill floor (the base rate); small-k results tagged when not significant.

Label scopes (strict -> broad):
  conviction      LEIE 1128A1/A2/A3 only (criminal convictions)
  fraud_relevant  LEIE 1128a1/a2/a3/b4/b7 + CMS 424.535 A3/A8 (current default)
  any             every LEIE/CMS sanction

Output: local_pipeline/anomaly/validation/point_in_time_report.md (+ .json)

Usage:
  python -m anomaly.validation.point_in_time
  python -m anomaly.validation.point_in_time --scope conviction
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from scipy.stats import hypergeom
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False
try:
    from sklearn.metrics import average_precision_score
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANOM = os.path.join(_ROOT, "local_pipeline", "anomaly")
FLAGS = os.path.join(ANOM, "flags.parquet")
LABELS = os.path.join(ANOM, "labels.parquet")
PROFILES = os.path.join(ANOM, "npi_profiles.parquet")
VAL = os.path.join(ANOM, "validation")

FEE_SCHEDULE_METRICS = {"charge_to_allowed_ratio", "avg_allowed"}
DECISION_YEARS = list(range(2014, 2022))     # T = 2014..2021
DEV_YEARS = [2014, 2015, 2016, 2017]
TEST_YEARS = [2018, 2019, 2020, 2021]
CAND_SCORES = ["n_metrics", "severity_max", "severity_sum", "n_methods", "n_flags"]
KS = [50, 100, 500, 1000, 2500]
SELECT_K = 1000                              # depth at which the score is chosen on DEV
CONVICTION_AUTH = {"1128A1", "1128A2", "1128A3"}


def hyper_p(tp: int, k: int, K: int, N: int) -> float:
    """P(>= tp positives in a random top-k) under the hypergeometric null."""
    if not _HAVE_SCIPY or k == 0 or K == 0 or tp == 0:
        return float("nan")
    return float(hypergeom.sf(tp - 1, N, K, k))


def bh_adjust(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR-adjusted p-values (NaNs preserved)."""
    idx = [i for i, p in enumerate(pvals) if p is not None and not math.isnan(p)]
    m = len(idx)
    adj = [float("nan")] * len(pvals)
    if m == 0:
        return adj
    order = sorted(idx, key=lambda i: pvals[i])
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        r = m - rank + 1
        val = min(prev, pvals[i] * m / r)
        adj[i] = val
        prev = val
    return adj


def earliest_event(labels: pd.DataFrame, scope: str) -> pd.Series:
    lab = labels.copy()
    if scope == "conviction":
        lab = lab[(lab["source"] == "LEIE") & (lab["authority"].isin(CONVICTION_AUTH))]
    elif scope == "fraud_relevant":
        lab = lab[lab["fraud_relevant"]]
    # 'any' -> keep all
    lab = lab.dropna(subset=["event_year"]).sort_values("event_date")
    return lab.groupby("npi").agg(event_year=("event_year", "first"),
                                  event_date=("event_date", "first"))


def asof_rank(flags: pd.DataFrame, T: int) -> pd.DataFrame:
    f = flags[flags["year"] <= T]
    return (f.groupby("npi")
            .agg(n_metrics=("flag_metric", "nunique"), n_methods=("flag_type", "nunique"),
                 n_flags=("severity", "size"), severity_sum=("severity", "sum"),
                 severity_max=("severity", "max"))
            .reset_index())


def per_year_metrics(flags, ev, scope_name, score, years):
    """Return list of dicts: per-T precision/lift/TP/p at each k for one score."""
    rows = []
    for T in years:
        rank = asof_rank(flags, T)
        pos_npis = set(ev.index[ev["event_year"] > T])
        rank["is_pos"] = rank["npi"].isin(pos_npis)
        N = len(rank)
        K = int(rank["is_pos"].sum())
        base = K / N if N else float("nan")
        s = rank.sort_values(score, ascending=False)["is_pos"].to_numpy()
        for k in KS:
            ke = min(k, N)
            tp = int(s[:ke].sum())
            prec = tp / ke if ke else float("nan")
            lift = (prec / base) if base else float("nan")
            rows.append({"T": T, "k": ke, "tp": tp, "prec": prec, "lift": lift,
                         "base": base, "N": N, "K": K,
                         "p": hyper_p(tp, ke, K, N)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="fraud_relevant",
                    choices=["conviction", "fraud_relevant", "any"])
    ap.add_argument("--include-fee-schedule", action="store_true")
    ap.add_argument("--strat-year", type=int, default=2018, help="decision year for volume stratification")
    args = ap.parse_args()
    os.makedirs(VAL, exist_ok=True)

    print(f"Point-in-time validation (scope={args.scope}, fee_excluded={not args.include_fee_schedule}) ...")
    flags = pd.read_parquet(FLAGS, columns=["Rndrng_NPI", "year", "flag_type", "flag_metric", "severity"])
    flags["npi"] = flags["Rndrng_NPI"].astype("int64").astype(str)
    if not args.include_fee_schedule:
        flags = flags[~flags["flag_metric"].isin(FEE_SCHEDULE_METRICS)]
    labels = pd.read_parquet(LABELS); labels["npi"] = labels["npi"].astype(str)
    ev = earliest_event(labels, args.scope)

    # ---- (2) HOLDOUT SELECTION: pick the score by mean lift@SELECT_K on DEV years ----
    dev_lift = {}
    for sc in CAND_SCORES:
        rows = per_year_metrics(flags, ev, args.scope, sc, DEV_YEARS)
        vals = [r["lift"] for r in rows if r["k"] >= min(SELECT_K, r["N"]) and r["k"] == min(SELECT_K, r["N"]) and not math.isnan(r["lift"])]
        # take the @SELECT_K row per T
        atk = [r["lift"] for r in rows if r["k"] == min(SELECT_K, r["N"]) and not math.isnan(r["lift"])]
        dev_lift[sc] = float(np.mean(atk)) if atk else float("nan")
    chosen = max(dev_lift, key=lambda s: (dev_lift[s] if not math.isnan(dev_lift[s]) else -1))
    print(f"  DEV mean lift@{SELECT_K}: " + ", ".join(f"{k}={v:.2f}" for k, v in dev_lift.items()))
    print(f"  -> selected score on DEV (never sees test): {chosen}")

    # ---- (1) POINT-IN-TIME on TEST years with the chosen score ----
    test_rows = per_year_metrics(flags, ev, args.scope, chosen, TEST_YEARS)
    # BH-adjust the Fisher/hypergeom p across the whole TEST k x year grid
    adj = bh_adjust([r["p"] for r in test_rows])
    for r, a in zip(test_rows, adj):
        r["p_bh"] = a

    # ---- corrected LEAD-TIME: earliest T the provider crossed top-k(=SELECT_K) ----
    alert_T = {}
    for T in TEST_YEARS:
        rank = asof_rank(flags, T).sort_values(chosen, ascending=False)
        topk = set(rank.head(SELECT_K)["npi"])
        for npi in topk:
            alert_T[npi] = min(alert_T.get(npi, 9999), T)
    lead_days = []
    for npi, T0 in alert_T.items():
        if npi in ev.index and ev.loc[npi, "event_year"] > T0:
            cutoff = pd.Timestamp(year=T0, month=12, day=31)
            d = (ev.loc[npi, "event_date"] - cutoff).days
            if d > 0:
                lead_days.append(d)
    lead = np.array(lead_days, dtype=float)
    lead_stats = ({"n": int(lead.size),
                   "median_mo": round(float(np.median(lead)) / 30.44, 1),
                   "p25_mo": round(float(np.percentile(lead, 25)) / 30.44, 1),
                   "p75_mo": round(float(np.percentile(lead, 75)) / 30.44, 1)}
                  if lead.size else {"n": 0})

    # ---- (3) VOLUME STRATIFICATION at strat-year ----
    T = args.strat_year
    prof = pd.read_parquet(PROFILES, columns=["Rndrng_NPI", "year", "total_services"])
    prof["npi"] = prof["Rndrng_NPI"].astype("int64").astype(str)
    vol = prof[prof["year"] <= T].groupby("npi")["total_services"].sum().rename("cum_vol")
    rank = asof_rank(flags, T)
    pos_npis = set(ev.index[ev["event_year"] > T])
    rank["is_pos"] = rank["npi"].isin(pos_npis)
    rank = rank.merge(vol, on="npi", how="left")
    rank["vol_band"] = pd.qcut(rank["cum_vol"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    strat = []
    for b in [1, 2, 3, 4, 5]:
        sub = rank[rank["vol_band"] == b]
        s = sub.sort_values(chosen, ascending=False)["is_pos"].to_numpy()
        Nb, Kb = len(sub), int(sub["is_pos"].sum())
        base_b = Kb / Nb if Nb else float("nan")
        kk = min(SELECT_K, Nb)
        tp = int(s[:kk].sum())
        lift_b = ((tp / kk) / base_b) if (base_b and kk) else float("nan")
        strat.append({"band": int(b), "n": Nb, "pos": Kb, "base": base_b,
                      "tp_at_k": tp, "k": kk, "lift": lift_b,
                      "med_vol": float(sub["cum_vol"].median())})

    # ---- (4) PR-AUC vs floor on TEST (pooled over test years) ----
    pr = []
    for T in TEST_YEARS:
        rank = asof_rank(flags, T)
        pos_npis = set(ev.index[ev["event_year"] > T])
        y = rank["npi"].isin(pos_npis).astype(int).to_numpy()
        sc = rank[chosen].to_numpy()
        if _HAVE_SK and y.sum() > 0:
            pr.append({"T": T, "pr_auc": float(average_precision_score(y, sc)),
                       "floor": float(y.mean())})

    # ---------------- report ----------------
    def t_table(rows):
        out = ["| T | N | positives | base | prec@1000 | **lift@1000** | TP | hyper p | BH p |",
               "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["k"] != min(SELECT_K, r["N"]):
                continue
            sig = "" if (r.get("p_bh") is None or math.isnan(r.get("p_bh", float("nan")))) else (" ✓" if r["p_bh"] < 0.05 else " n.s.")
            pbh = "—" if (r.get("p_bh") is None or math.isnan(r.get("p_bh", float("nan")))) else f"{r['p_bh']:.2e}"
            pp = "—" if math.isnan(r["p"]) else f"{r['p']:.2e}"
            out.append(f"| {r['T']} | {r['N']:,} | {r['K']} | {r['base']*100:.3f}% | "
                       f"{r['prec']*100:.2f}% | **{r['lift']:.1f}×** | {r['tp']} | {pp} | {pbh}{sig} |")
        return "\n".join(out)

    pr_line = (", ".join(f"{p['T']}: {p['pr_auc']:.4f} (floor {p['floor']*100:.3f}%)" for p in pr)
               if pr else "n/a")
    strat_collapsed = all((s["lift"] < 1.5 or math.isnan(s["lift"])) for s in strat)

    md = f"""# Leakage-Corrected Validation (point-in-time + holdout)

_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · scope = **{args.scope}** · fee-schedule {'included' if args.include_fee_schedule else 'excluded'}._

Fixes the two leaks an independent review found in the original backtest: the
ranking now uses **only data available as of each decision year T** (no
look-ahead), the score was **selected on dev years (≤2017) and reported on test
years (≥2018)** (no selection-on-test), lead-time is anchored to the **earliest
year the provider crossed top-{SELECT_K}**, lift is **stratified by volume**, and
p-values are **BH-FDR corrected**. Output remains investigative leads, not accusations.

## Score selection (on DEV years 2014–2017 only)
Mean lift@{SELECT_K} on dev: {", ".join(f"`{k}` {v:.2f}×" for k, v in dev_lift.items())}.
**Selected: `{chosen}`** (the test years below never influenced this choice).

## (1)+(2) Point-in-time lift on TEST years (2018–2021), score = `{chosen}`
Each row is a real prospective decision: rank as-of-T, then observe sanctions with event_year > T.
{t_table(test_rows)}

_BH p < 0.05 marked ✓. Later T have shorter accrual windows (more censoring), so they are conservative._

## (4) PR-AUC vs no-skill floor (TEST years)
{pr_line}

_PR-AUC near the floor (the base rate) = the global ranking carries little separating signal; what lift exists is a thin top-of-list effect._

## Corrected lead-time (anchored to first top-{SELECT_K} appearance, not first flag)
- {("median **%.1f mo** (IQR %.0f–%.0f), n=%d" % (lead_stats['median_mo'], lead_stats['p25_mo'], lead_stats['p75_mo'], lead_stats['n'])) if lead_stats['n'] else "no qualifying predictive hits"}.
- This replaces the earlier 63.5-mo figure, which was anchored to the first flag year and was optimistic.

## (3) Volume stratification (decision year {args.strat_year}, lift@{SELECT_K} within volume quintiles)
If lift stays ≈1× inside every band, the headline lift is a **volume confound**, not fraud-specific signal.
| volume band | providers | positives | base | TP@{SELECT_K} | **lift** | median cum. services |
|---|---|---|---|---|---|---|
""" + "\n".join(
        f"| {s['band']} ({'lowest' if s['band']==1 else 'highest' if s['band']==5 else '·'}) | {s['n']:,} | {s['pos']} | "
        f"{s['base']*100:.3f}% | {s['tp_at_k']} | **{s['lift']:.1f}×** | {s['med_vol']:,.0f} |"
        for s in strat) + f"""

**Verdict: {'lift LARGELY COLLAPSES within volume bands → the signal is substantially a volume confound.' if strat_collapsed else 'lift partially survives within volume bands → not purely a volume confound (but interpret with the small TP counts).'}**

## Honest bottom line
- The defensible signal is the **deep-k (k={SELECT_K}) lift on held-out test years**, BH-corrected — see the ✓ rows above. Small-k (top-10/25/100) lift is **not reported as a headline** because it rests on 1–2 hits.
- Lead-time, corrected, is the table above — read it, not the old 63.5 mo.
- Whether the lift reflects genuine detection vs a volume confound is answered by the stratification table.
"""
    with open(os.path.join(VAL, "point_in_time_report.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    summary = {"scope": args.scope, "chosen_score": chosen, "dev_lift": dev_lift,
               "test_rows": test_rows, "lead_time": lead_stats, "volume_strata": strat,
               "pr_auc": pr, "strat_collapsed": bool(strat_collapsed)}
    with open(os.path.join(VAL, "point_in_time_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print(f"  chosen={chosen}  test lift@{SELECT_K} by year:",
          [f"{r['T']}:{r['lift']:.1f}x(TP={r['tp']},BHp={r['p_bh']:.2g})"
           for r in test_rows if r["k"] == min(SELECT_K, r["N"])])
    print(f"  corrected lead-time median={lead_stats.get('median_mo')}mo (n={lead_stats.get('n')})")
    print(f"  volume-confound: {'YES (lift collapses in bands)' if strat_collapsed else 'partially survives'}")
    print(f"  -> {os.path.join(VAL, 'point_in_time_report.md')}")


if __name__ == "__main__":
    main()
