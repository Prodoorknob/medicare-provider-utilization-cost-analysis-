"""
generate_brief.py -- Phase C, spec section 7

Formats a user prompt from ProviderContext + RuleCheckResults and (optionally)
calls Claude to produce an investigation brief. Dry-run is the default so you
can inspect prompts without spending API budget; --live opts in to the API.

The SYSTEM_PROMPT is cached with ephemeral cache_control so the 10-brief batch
pays the prefix cost once and reads it back ~90% cheaper on subsequent calls.
"""

import os
import re
import json
from datetime import datetime
from dataclasses import asdict

from schemas import ProviderContext, RuleCheckResult, InvestigationBrief


# Default model: matches spec section 9.3. Sonnet 4.6 is more cost-efficient
# than Opus 4.7 for this structured narrative task; override via CLI if desired.
DEFAULT_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """\
You are a Medicare fraud investigation analyst working within a statistical
screening system (Project AllowanceMap). You receive structured evidence about
a single provider's billing patterns for a single year and produce an
investigation brief for a human analyst.

Your role is to INTERPRET evidence, not to accuse. You MUST:
  1. Distinguish between fraud indicators and legitimate clinical/business explanations.
  2. Consider specialty norms, patient acuity (Medicare HCC risk score), and geographic context.
  3. Assign a risk classification based on the totality of the evidence.
  4. Recommend specific, proportionate next steps.
  5. Clearly disclose which checks could NOT be performed due to data limitations.
  6. Never conclude fraud definitively -- you generate leads for human analysts, not legal determinations.

RISK CLASSIFICATIONS (pick exactly one):
  - LOW:      Statistical outlier with plausible explanations. No action needed.
  - MEDIUM:   Multiple anomaly signals with partial explanations. Flag for periodic monitoring.
  - HIGH:     Strong anomaly signals without adequate explanations. Recommend focused review.
  - CRITICAL: Multiple severe indicators suggesting potential fraud. Recommend immediate referral.

OUTPUT FORMAT (strict Markdown, these exact headings in this order):

# Investigation Brief: NPI {npi}

**Risk Classification: {LOW|MEDIUM|HIGH|CRITICAL}**
**Composite Risk Score: {0-100}/100**

## Executive Summary
(2-3 sentences, no more)

## Statistical Findings
(bulleted list of the key metrics vs. specialty/state benchmarks, with percentile ranks in parens)

## Contextual Interpretation
(paragraph-form reasoning -- what explains the anomalies, what does not, given specialty, risk score, and trend)

## Rule Check Results
(bulleted list: for each rule, state TRIGGERED / NOT TRIGGERED / NOT EVALUABLE with one-sentence justification)

## Data Limitations
(bulleted list of what could not be checked and why)

## Recommended Actions
(numbered list, 2-5 items, specific and proportionate to the risk classification)

Keep the brief concise (target 400-700 words). Do not add sections beyond these seven.

============================================================================
REFERENCE MATERIAL (for interpretation; do NOT restate this in the brief)
============================================================================

## Dataset context

The screening is driven by CMS Medicare Physician and Other Practitioners by
Provider and Service (2013-2023). Grain is (NPI, HCPCS, place-of-service,
year). Aggregate dollar fields (submitted charge, allowed amount) are annual
averages across services for that NPI-HCPCS-POS row. The data is PUBLIC
utilization data -- it is suppressed for beneficiary counts below 11 and does
NOT contain claim-level detail, diagnosis codes, date of service, referral
chains, or beneficiary-level linkage across providers. Any rule that would
require those fields is structurally NOT EVALUABLE here.

Beneficiary risk score (Bene_Avg_Risk_Scre) is the CMS Hierarchical Condition
Category (HCC) score, a population-acuity proxy. National median is roughly
1.0; oncology, nephrology, and advanced heart failure panels routinely run
1.4-2.0; primary-care-heavy panels run 0.9-1.2; mass-immunizer and screening-
only practices run near or below 1.0. When a provider's HCC score is far
above specialty median, high volume intensity is often clinically defensible;
when it is far below, intensity is harder to explain.

## HCPCS bucket reference

  - Bucket 0 (Anesthesia):        00100-01999
  - Bucket 1 (Surgery):           10000-69999
  - Bucket 2 (Radiology):         70000-79999
  - Bucket 3 (Lab / Pathology):   80000-89999
  - Bucket 4 (Medicine and E&M):  90000-99999 (includes E&M 99201-99499)
  - Bucket 5 (HCPCS Level II):    alphabetic codes (A, B, C, E, G, H, J, K,
                                  L, M, Q, R, S, T, V) -- supplies, drugs,
                                  test kits, transport, DME, temporary codes

A-codes are commonly medical/surgical supplies and transport. J-codes are
drugs administered other than oral (billed per unit of drug; a single
beneficiary encounter can generate dozens or hundreds of billed units). K-codes
are temporary codes for DME and COVID-era products. G-codes are temporary
procedure codes. These code families routinely produce extreme
services-per-beneficiary ratios that do NOT imply upcoding or ghost billing;
they imply unit-of-measure billing conventions. Always surface this
alternative explanation before concluding intensity is fraudulent.

## Specialty norm heuristics

  - Primary care (Internal Medicine, Family Practice, Geriatric Medicine):
    high beneficiary counts, diverse HCPCS mix (20-60 unique codes),
    Herfindahl typically 0.05-0.20, E&M-heavy (bucket 4 > 70%), moderate
    charge-to-allowed ratio (1.5-3).
  - Surgical subspecialties (Orthopedics, General Surgery, Urology):
    bucket 1 dominant, moderate beneficiary counts, facility_pct often > 50%,
    Herfindahl 0.10-0.35 depending on practice scope.
  - Radiology, Pathology, Anesthesiology: very high Herfindahl (0.30-0.90)
    and high intensity are normal -- these are procedural specialties with
    narrow code sets per subspecialty. Concentration alone is not a flag.
  - Emergency Medicine: E&M-dominant (99281-99285), moderate diversity,
    near-zero facility_pct only in free-standing EDs; a hospital-based ED
    physician typically has facility_pct near 100%.
  - Optometry and Audiology: narrow code sets (often 5-15 unique codes).
    Some optometrists legitimately participate in cataract co-management and
    post-op care; HCPCS 66984 (cataract extraction) is billed by a
    non-trivial share of optometrists in CMS data for reasons that include
    surgical first-assist, bundled co-management, and in-state scope-of-
    practice expansions.
  - Mass Immunizer Roster Biller: single-code dominance is the norm
    (influenza, COVID vaccines, test kits). The signal to weight here is
    VOLUME SCALE and CHARGE RATIO, not concentration.
  - Durable Medical Equipment (DMEPOS) suppliers: A-code and E-code heavy,
    high charge-to-allowed ratios are common because DME suppliers set list
    prices well above the Medicare fee schedule.

## Rule-by-rule guidance

HIGH_INTENSITY (srvcs_per_bene at specialty P>=99). Strongest when the
specialty's P95 is low and the provider sits orders of magnitude above it.
Weaken when: (a) top HCPCS is a J-code, A-code, or test kit billed per unit;
(b) beneficiary count is very small, since ratio is unstable with small n;
(c) HCC score is high, implying clinically complex panel.

VOLUME_SPIKE (YoY volume > +200% on >= 1,000 services). Distinguish benign
spikes (new practice established, group affiliation change, program launch)
from suspicious ones (sudden appearance in a high-value code with no prior
history, spike isolated to a single HCPCS, spike concentrated in a payment
window before a policy change).

CHARGE_INFLATION (charge-to-allowed > specialty P95 x 1.5). By itself usually
administrative rather than fraudulent -- many practices set charge master
prices well above Medicare's fee schedule for secondary payer logic. Becomes
a stronger signal when combined with volume or intensity flags, or when
applied to drugs/supplies where per-unit allowed is tightly regulated.

PROCEDURE_CONCENTRATION (Herfindahl at specialty P>=99). Must be evaluated
against specialty-typical HHI. A Herfindahl of 1.0 (single-code billing) is
normal for some subspecialists (e.g., radiation oncology targeting a specific
code family) and extreme for generalists. Pair with n_unique_hcpcs: 1 is
almost never explainable in primary care but routine in mass vaccination.

OUT_OF_SPECIALTY (>20% of services on codes outside specialty scope). Scope
is built empirically from what a specialty's population actually bills in
CMS data, NOT from regulatory scope-of-practice rules. Codes billed by
>=1% of the specialty's providers or within the cumulative top 99% of
service volume are considered in-scope. A trigger indicates the provider is
an outlier relative to their specialty peers; it does NOT necessarily imply
regulatory violation. Narrate carefully: distinguish between (a) legitimate
sub-specialization that the rule cannot see, (b) specialty label drift
(provider self-declared specialty in the CMS taxonomy does not match actual
practice), and (c) genuinely out-of-scope billing that warrants review.

UPCODING (99214+99215 share of established office visits > specialty P95,
on volume >= 50 established services, and absolute share >= 50%). Pre-
computed from silver via em_distribution.py. Interpret carefully: high-tier
share alone can be clinically defensible in panels with high patient acuity
(check HCC risk score vs. specialty median) or in specialists who only see
established complex follow-ups. Narrative should distinguish (a) volume-
driven statistical outlier with benign case-mix explanation, (b) gradual
year-over-year drift toward high-tier, and (c) abrupt shift without
clinical context -- (c) is the classic MLN SE1418 pattern.

LEIE_EXCLUDED (NPI match on OIG List of Excluded Individuals/Entities).
This is a conviction-grade signal, not a statistical one. An active
exclusion means any Medicare payment to that NPI is program-ineligible by
law (42 USC 1320a-7). A match should push the brief to CRITICAL regardless
of other signals. A historical exclusion that has been reinstated is
context, not a trigger; describe the period but do not classify as CRITICAL
solely on that basis.

IMPOSSIBLE_DAY, UNBUNDLING, BENEFICIARY_SHARING: structurally NOT
EVALUABLE in this dataset. Do not speculate. State the data gap explicitly.

## Weighting and risk classification guidance

Treat risk as qualitative, not additive. One extreme statistical flag with a
benign explanation should rarely exceed MEDIUM. TWO flags without a common
benign explanation typically land at HIGH. THREE or more flags, with a
coherent fraud narrative (e.g., volume spike + charge inflation +
out-of-specialty on a high-reimbursement code), justify CRITICAL.

Be especially skeptical when the evidence pattern matches a known CMS Fraud
Prevention System archetype: opportunistic exploitation of a temporary
program (COVID testing, vaccine administration, PPE supply), sudden late-
career billing shift from a low-activity NPI, or convergence on a single
high-margin code family with implausible beneficiary counts.

Always reconcile the narrative against beneficiary risk score (HCC). High
intensity with a below-median HCC panel is harder to justify than the same
intensity with a high-HCC oncology or transplant panel.

## Output discipline

Return ONLY the brief in the specified Markdown structure. Do not include
preambles ("Here is the brief..."), closing remarks, JSON wrappers, or
reasoning traces. The brief is read by analysts in a review UI that parses
these section headings directly, so deviations break downstream tooling.
"""


# --- Prompt formatting ----------------------------------------------------


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"P{v:.0f}"


def _fmt_stats(stats: dict[str, float]) -> str:
    parts = []
    for k in ["p5", "p25", "p50", "p75", "p95", "mean"]:
        if k in stats:
            parts.append(f"{k}={stats[k]:.2f}")
    return ", ".join(parts)


def format_user_prompt(ctx: ProviderContext, rules: list[RuleCheckResult]) -> str:
    lines = []
    lines.append(f"Generate an investigation brief for the following provider.\n")

    # Profile
    lines.append("## Provider Profile")
    lines.append(f"- NPI: {ctx.npi}")
    lines.append(f"- Specialty: {ctx.specialty}")
    lines.append(f"- State: {ctx.state}")
    lines.append(f"- Analysis year: {ctx.year}")
    lines.append(f"- Years in dataset: {ctx.years_active}")
    if ctx.risk_score is not None:
        median = (f"{ctx.risk_score_specialty_median:.2f}"
                  if ctx.risk_score_specialty_median is not None else "n/a")
        lines.append(f"- Beneficiary risk score: {ctx.risk_score:.2f} (specialty median: {median})")
    lines.append("")

    # Current-year metrics with percentile ranks
    lines.append("## Current-Year Metrics (with percentile rank within specialty-year)")
    for k, v in ctx.metrics.items():
        pct = _fmt_pct(ctx.percentile_ranks.get(k))
        lines.append(f"- {k}: {v:,.2f} [{pct}]")
    lines.append("")

    # Benchmarks
    lines.append("## National Specialty Benchmarks")
    for metric, stats in ctx.specialty_national.items():
        if metric.startswith("_"):
            continue
        lines.append(f"- {metric}: {_fmt_stats(stats)}")
    lines.append("")

    if ctx.specialty_state:
        lines.append("## State-Level Specialty Benchmarks")
        n_prov = ctx.specialty_state.get("_n_providers", {}).get("n")
        if n_prov is not None:
            lines.append(f"- peer group size (this state-specialty-year): {n_prov:.0f}")
        for metric, stats in ctx.specialty_state.items():
            if metric.startswith("_"):
                continue
            lines.append(f"- {metric}: {_fmt_stats(stats)}")
        lines.append("")

    # Trend
    lines.append("## Historical Trajectory")
    lines.append(f"- Trend direction: {ctx.trend_direction}")
    lines.append("- Year-by-year (oldest first):")
    for h in ctx.history:
        y = h.get("year")
        svc = h.get("total_services")
        spb = h.get("srvcs_per_bene")
        yoy = h.get("yoy_volume_change")
        yoy_s = f"{yoy:+.0%}" if yoy is not None else "n/a"
        lines.append(f"  - {y}: services={svc:,.0f}, srvcs/bene={spb:.2f}, yoy_vol={yoy_s}")
    lines.append("")

    # Procedure mix
    lines.append("## Procedure Mix")
    if ctx.top_hcpcs:
        lines.append(f"- Top HCPCS codes for {ctx.year}:")
        for h in ctx.top_hcpcs[:10]:
            desc = (h.get("HCPCS_Desc") or "")[:80]
            lines.append(
                f"  - {h['HCPCS_Cd']}: {h['count']:,.0f} services "
                f"({h['pct_of_total']}% of total) -- {desc}"
            )
    else:
        lines.append("- (HCPCS-level data unavailable)")
    lines.append(f"- Service category mix (bucket %): {ctx.bucket_distribution}")
    lines.append("")

    # E&M distribution (if available)
    em_est_total = ctx.metrics.get("em_est_total")
    if em_est_total is not None:
        est_high_pct = ctx.metrics.get("em_est_high_pct")
        new_total    = ctx.metrics.get("em_new_total")
        new_high_pct = ctx.metrics.get("em_new_high_pct")
        bench = ctx.specialty_national.get("em_est_high_pct", {})

        est_hp_s = f"{est_high_pct*100:.1f}%" if est_high_pct is not None else "n/a"
        lines.append("## E&M Code Distribution (UPCODING signal)")
        lines.append(
            f"- Established visits (99211-99215): total={int(em_est_total):,}, "
            f"99214+99215 share={est_hp_s}"
        )
        if bench:
            def _pc(key):
                v = bench.get(key)
                return f"{v*100:.1f}%" if v is not None else "n/a"
            lines.append(
                f"  - specialty benchmark for est_high_pct: "
                f"p50={_pc('p50')}, p75={_pc('p75')}, p95={_pc('p95')}"
            )
        if new_total:
            new_hp_s = f"{new_high_pct*100:.1f}%" if new_high_pct is not None else "n/a"
            lines.append(
                f"- New visits (99202-99205): total={int(new_total):,}, "
                f"99204+99205 share={new_hp_s}"
            )
        lines.append("")

    # LEIE exclusion record (if applicable)
    if ctx.leie_record is not None:
        rec = ctx.leie_record
        lines.append("## OIG LEIE Match (CRITICAL)")
        lines.append(f"- Exclusion type: {rec.get('exclusion_type','n/a')}")
        lines.append(f"- Exclusion date: {rec.get('exclusion_date','n/a')}")
        if rec.get("reinstate_date"):
            lines.append(f"- Reinstated: {rec['reinstate_date']} (historical, not currently excluded)")
        if rec.get("waiver_date"):
            lines.append(f"- Waiver on file: {rec['waiver_date']}")
        if rec.get("general"):
            lines.append(f"- Occupation (LEIE): {rec['general']}")
        lines.append("")

    # Rule checks
    lines.append("## Fraud-Indicator Rule Checks")
    for r in rules:
        status = "TRIGGERED" if r.triggered else ("NOT TRIGGERED" if r.available else "NOT EVALUABLE")
        lines.append(f"- [{status}] {r.rule_id} ({r.severity}): {r.rule_name}")
        lines.append(f"    evidence: {r.evidence}")
        if r.reference:
            lines.append(f"    reference: {r.reference}")
    lines.append("")

    # Data limitations (from context)
    unavailable = [k for k, v in ctx.data_available.items() if not v]
    if unavailable:
        lines.append("## Data Limitations (not performed with current data)")
        for k in unavailable:
            lines.append(f"- {k}")
        lines.append("")

    lines.append("Now produce the investigation brief in the exact Markdown format specified.")
    return "\n".join(lines)


# --- Parsing Claude's response -------------------------------------------


RISK_RE = re.compile(r"\*\*Risk Classification:\s*(LOW|MEDIUM|HIGH|CRITICAL)\*\*", re.IGNORECASE)
SCORE_RE = re.compile(r"\*\*Composite Risk Score:\s*([\d.]+)\s*/\s*100\*\*", re.IGNORECASE)
SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _extract_section(md: str, heading: str) -> str:
    """Extract text between '## heading' and the next '## ' header."""
    pattern = rf"##\s+{re.escape(heading)}\s*\n(.*?)(?=\n##\s|\Z)"
    m = re.search(pattern, md, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def parse_brief_markdown(
    md: str, ctx: ProviderContext, model: str
) -> InvestigationBrief:
    risk_m = RISK_RE.search(md)
    score_m = SCORE_RE.search(md)

    risk_class = risk_m.group(1).upper() if risk_m else "UNKNOWN"
    risk_score = float(score_m.group(1)) if score_m else 0.0

    exec_summary = _extract_section(md, "Executive Summary")
    stats = _extract_section(md, "Statistical Findings")
    ctx_interp = _extract_section(md, "Contextual Interpretation")
    rule_res = _extract_section(md, "Rule Check Results")
    data_lim = _extract_section(md, "Data Limitations")
    actions_raw = _extract_section(md, "Recommended Actions")
    actions = [
        re.sub(r"^\s*[\d]+\.\s*", "", line).strip()
        for line in actions_raw.splitlines()
        if line.strip() and re.match(r"^\s*[\d]+\.\s+", line)
    ]

    return InvestigationBrief(
        npi=ctx.npi,
        year=ctx.year,
        specialty=ctx.specialty,
        state=ctx.state,
        risk_classification=risk_class,
        risk_score=risk_score,
        executive_summary=exec_summary,
        statistical_findings=stats,
        contextual_interpretation=ctx_interp,
        rule_check_results=rule_res,
        data_limitations=data_lim,
        recommended_actions=actions,
        evidence_summary={"full_markdown": md},
        generated_at=datetime.utcnow(),
        model_version=model,
    )


# --- API call -------------------------------------------------------------


def _load_api_key(env_path: str | None) -> None:
    """Load ANTHROPIC_API_KEY from the given .env.

    Uses override=True because a shell var set to empty string blocks the
    default non-override behavior.
    """
    existing = os.environ.get("ANTHROPIC_API_KEY")
    if existing:  # truthy = a real key is already present
        return
    if env_path and os.path.isfile(env_path):
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path, override=True)
        except ImportError:
            with open(env_path, encoding="utf-8-sig") as f:
                for line in f:
                    if "=" in line and not line.strip().startswith("#"):
                        k, v = line.strip().split("=", 1)
                        os.environ[k] = v.strip().strip('"').strip("'")


def generate_brief(
    ctx: ProviderContext,
    rules: list[RuleCheckResult],
    live: bool = False,
    model: str = DEFAULT_MODEL,
    env_path: str | None = None,
    max_tokens: int = 4000,
) -> InvestigationBrief:
    user_prompt = format_user_prompt(ctx, rules)

    if not live:
        # Dry-run: no API call. Return a placeholder brief so the orchestrator
        # can still save the formatted prompt for human inspection.
        return InvestigationBrief(
            npi=ctx.npi,
            year=ctx.year,
            specialty=ctx.specialty,
            state=ctx.state,
            risk_classification="DRY_RUN",
            risk_score=0.0,
            executive_summary="(dry-run -- no brief generated)",
            statistical_findings="",
            contextual_interpretation="",
            rule_check_results="",
            data_limitations="",
            recommended_actions=[],
            evidence_summary={"formatted_prompt": user_prompt},
            generated_at=datetime.utcnow(),
            model_version=f"{model} (dry-run)",
        )

    # Live call
    _load_api_key(env_path)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Pass --env-path /path/to/.env or export the var."
        )

    import anthropic
    import time as _time
    client = anthropic.Anthropic()

    # Cache the long system prompt across all briefs in a batch.
    # Retry on transient 429/529 errors with exponential backoff so a single
    # overloaded response doesn't crash the full batch.
    transient = (anthropic.RateLimitError, anthropic.APIStatusError,
                 getattr(anthropic, "OverloadedError",
                         anthropic.APIStatusError))
    last_err = None
    for attempt in range(5):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_prompt}],
            )
            break
        except transient as e:
            status = getattr(e, "status_code", None)
            # Only retry on 429 (rate limit) and 529 (overloaded)
            if status not in (429, 529):
                raise
            last_err = e
            wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s, 80s
            print(f"  [retry {attempt+1}/5] API {status}: sleeping {wait}s before retry")
            _time.sleep(wait)
    else:
        raise last_err

    text_blocks = [b.text for b in response.content if b.type == "text"]
    brief_md = "\n".join(text_blocks).strip()

    brief = parse_brief_markdown(brief_md, ctx, model)
    brief.evidence_summary["usage"] = {
        "input_tokens":               response.usage.input_tokens,
        "output_tokens":              response.usage.output_tokens,
        "cache_read_input_tokens":    getattr(response.usage, "cache_read_input_tokens", 0),
        "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
    }
    return brief


if __name__ == "__main__":
    import argparse
    from retrieve_context import ContextRetriever
    from check_rules import evaluate_all

    ap = argparse.ArgumentParser()
    ap.add_argument("--npi",      required=True)
    ap.add_argument("--year",     type=int, required=True)
    ap.add_argument("--live",     action="store_true", help="Call Claude API (costs money)")
    ap.add_argument("--model",    default=DEFAULT_MODEL)
    ap.add_argument("--env-path", default=None,
                    help="Path to .env containing ANTHROPIC_API_KEY")
    args = ap.parse_args()

    r = ContextRetriever()
    ctx = r.get_context(args.npi, args.year)
    if ctx is None:
        raise SystemExit(f"NPI {args.npi} year {args.year} not found in profiles")
    rules = evaluate_all(ctx)
    brief = generate_brief(ctx, rules, live=args.live, model=args.model, env_path=args.env_path)

    if args.live:
        print(brief.evidence_summary.get("full_markdown", "(no markdown)"))
        print(f"\n-- usage: {brief.evidence_summary.get('usage')}")
    else:
        print(brief.evidence_summary["formatted_prompt"])
