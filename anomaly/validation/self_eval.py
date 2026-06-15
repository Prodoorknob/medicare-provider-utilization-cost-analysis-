"""
self_eval.py -- Phase 4 (E): track the fraud-lead workflow's own performance
over time.

Each run of the validation backtest (run_validation.py) writes a point-in-time
validation_summary.json. As the government ground truth accrues (LEIE refreshes
monthly, CMS Revoked quarterly), the SAME detector's leads should be re-scored
against the updated labels -- this module records each re-scoring as a row in a
history log and renders a self-eval dashboard so drift in lift / lead-time /
positive count is visible.

This is the "is it still working?" instrument for an autonomous deployment:
the agent grades its own past leads against new sanctions and reports the trend.

Reads:
  local_pipeline/anomaly/validation/validation_summary.json   (run_validation)
  local_pipeline/anomaly/labels_metadata.json                 (build_labels)
  local_pipeline/anomaly/leie_metadata.json, cms_revoked_metadata.json (loaders)
Writes/appends:
  local_pipeline/anomaly/validation/validation_history.jsonl  (one row per run)
  local_pipeline/anomaly/validation/self_eval_dashboard.md     (trend dashboard)

Usage:
  python -m anomaly.validation.self_eval
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANOM_DIR = os.path.join(_PROJECT_ROOT, "local_pipeline", "anomaly")
VAL_DIR  = os.path.join(ANOM_DIR, "validation")
SUMMARY  = os.path.join(VAL_DIR, "validation_summary.json")
LABELS_META = os.path.join(ANOM_DIR, "labels_metadata.json")
LEIE_META   = os.path.join(ANOM_DIR, "leie_metadata.json")
CMS_META    = os.path.join(ANOM_DIR, "cms_revoked_metadata.json")
HISTORY  = os.path.join(VAL_DIR, "validation_history.jsonl")
DASH     = os.path.join(VAL_DIR, "self_eval_dashboard.md")


def _load(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _at_k(rows: list, k: int) -> dict:
    for r in rows or []:
        if r.get("k") == k:
            return r
    return {}


def build_record() -> dict:
    s = _load(SUMMARY)
    if not s:
        raise SystemExit(f"No validation_summary.json at {SUMMARY} -- run run_validation first.")
    lab = _load(LABELS_META)
    leie = _load(LEIE_META)
    cms = _load(CMS_META)
    ks = s.get("headline_ks", [])
    k100, k1000 = _at_k(ks, 100), _at_k(ks, 1000)
    lead = s.get("lead_time_all", {}) or {}
    by_source = (lab.get("by_source") or {})
    return {
        "run_at":               datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labels_built_at":      lab.get("built_at"),
        "leie_fetched_at":      leie.get("fetched_at"),
        "cms_fetched_at":       cms.get("fetched_at"),
        "n_flagged_npis":       s.get("n_flagged_npis"),
        "n_universe":           s.get("n_universe"),
        "n_positives":          s.get("n_positives_fraud_relevant"),
        "cohort_base_rate":     s.get("cohort_base_rate"),
        "pr_auc":               s.get("pr_auc"),
        "prec_at_100":          k100.get("precision"),
        "lift_at_100":          k100.get("lift"),
        "tp_at_100":            k100.get("tp"),
        "prec_at_1000":         k1000.get("precision"),
        "lift_at_1000":         k1000.get("lift"),
        "tp_at_1000":           k1000.get("tp"),
        "lead_time_median_mo":  lead.get("median_mo"),
        "lead_time_n":          lead.get("n"),
        "label_unique_npi":     lab.get("n_unique_npi"),
        "label_fraud_relevant": lab.get("n_fraud_relevant"),
        "leie_rows":            by_source.get("LEIE"),
        "cms_rows":             by_source.get("CMS_REVOKED"),
    }


def append_history(rec: dict) -> list[dict]:
    os.makedirs(VAL_DIR, exist_ok=True)
    with open(HISTORY, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    rows = []
    with open(HISTORY, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _fmt(v, pct=False, mult=False):
    if v is None:
        return "—"
    if isinstance(v, float):
        if pct:
            return f"{v*100:.3f}%"
        if mult:
            return f"{v:.1f}×"
        return f"{v:.4f}"
    return str(v)


def _mo(v):
    return "—" if v is None else f"{float(v):.1f}"


def render_dashboard(rows: list[dict]) -> str:
    latest = rows[-1]
    prev = rows[-2] if len(rows) > 1 else None

    def delta(field, mult=False):
        if not prev or latest.get(field) is None or prev.get(field) is None:
            return ""
        d = latest[field] - prev[field]
        sign = "+" if d >= 0 else ""
        return f" ({sign}{d:.1f}×)" if mult else f" ({sign}{d:.3f})"

    lines = [
        "# Fraud-Lead Self-Evaluation Dashboard",
        f"\n_Updated {latest['run_at']} · {len(rows)} run(s) recorded._\n",
        "The detector's leads are re-scored against the latest government ground "
        "truth (OIG LEIE + CMS Revoked) each run. Headline metric is **lift** "
        "(base-rate-normalized) and **lead-time** (months the flag preceded the "
        "sanction). These are leads for analyst review, not accusations.\n",
        "## Latest run",
        f"- Ground truth: **{_fmt(latest.get('label_unique_npi'))}** sanctioned NPIs "
        f"(LEIE {_fmt(latest.get('leie_rows'))} + CMS {_fmt(latest.get('cms_rows'))}); "
        f"**{_fmt(latest.get('label_fraud_relevant'))}** fraud-relevant.",
        f"- Flagged cohort: **{_fmt(latest.get('n_flagged_npis'))}** providers; "
        f"**{_fmt(latest.get('n_positives'))}** later sanctioned "
        f"(base rate {_fmt(latest.get('cohort_base_rate'), pct=True)}).",
        f"- **lift@100 = {_fmt(latest.get('lift_at_100'), mult=True)}**"
        f"{delta('lift_at_100', mult=True)} (TP={_fmt(latest.get('tp_at_100'))}) · "
        f"**lift@1000 = {_fmt(latest.get('lift_at_1000'), mult=True)}**"
        f"{delta('lift_at_1000', mult=True)} (TP={_fmt(latest.get('tp_at_1000'))})",
        f"- **lead-time median = {_mo(latest.get('lead_time_median_mo'))} mo** "
        f"(n={_fmt(latest.get('lead_time_n'))}) · PR-AUC {_fmt(latest.get('pr_auc'))}",
        "\n## History",
        "| run (UTC) | sanctioned NPIs | cohort positives | base rate | lift@100 (TP) | lift@1000 (TP) | lead-time med | PR-AUC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('run_at','—')[:16]} | {_fmt(r.get('label_unique_npi'))} | "
            f"{_fmt(r.get('n_positives'))} | {_fmt(r.get('cohort_base_rate'), pct=True)} | "
            f"{_fmt(r.get('lift_at_100'), mult=True)} ({_fmt(r.get('tp_at_100'))}) | "
            f"{_fmt(r.get('lift_at_1000'), mult=True)} ({_fmt(r.get('tp_at_1000'))}) | "
            f"{_mo(r.get('lead_time_median_mo'))} mo | {_fmt(r.get('pr_auc'))} |")
    lines.append(
        "\n_Watch for: lift@k drifting toward 1.0× (detector no better than random), "
        "a falling lead-time (sanctions catching up to flags), or a stalled "
        "sanctioned-NPI count (ground-truth refresh failing)._")
    return "\n".join(lines)


def main():
    rec = build_record()
    rows = append_history(rec)
    md = render_dashboard(rows)
    with open(DASH, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"[self-eval] recorded run {rec['run_at']} -> {HISTORY}")
    print(f"[self-eval] lift@100={_fmt(rec.get('lift_at_100'), mult=True)} "
          f"lift@1000={_fmt(rec.get('lift_at_1000'), mult=True)} "
          f"lead-time={_mo(rec.get('lead_time_median_mo'))}mo "
          f"positives={_fmt(rec.get('n_positives'))} of {_fmt(rec.get('n_flagged_npis'))}")
    print(f"[self-eval] dashboard -> {DASH}  ({len(rows)} run(s))")


if __name__ == "__main__":
    main()
