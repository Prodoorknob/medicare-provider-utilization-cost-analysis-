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
    client = anthropic.Anthropic()

    # Cache the long system prompt across all briefs in a batch
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
