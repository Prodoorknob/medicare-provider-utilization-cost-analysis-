"""
check_rules.py -- Phase C, spec section 6

Evaluates a set of known Medicare fraud indicators against a ProviderContext.
Rules that cannot be evaluated with the current public dataset are flagged
available=False so the brief can disclose what was NOT checked.
"""

from schemas import ProviderContext, RuleCheckResult


def _pct_rank(ctx: ProviderContext, metric: str) -> float | None:
    return ctx.percentile_ranks.get(metric)


def _value(ctx: ProviderContext, metric: str) -> float | None:
    return ctx.metrics.get(metric)


def _specialty_p95(ctx: ProviderContext, metric: str) -> float | None:
    stats = ctx.specialty_national.get(metric, {})
    return stats.get("p95")


# --- Individual rule evaluators ------------------------------------------


def _fmt(v, spec: str = ".2f") -> str:
    if v is None:
        return "n/a"
    try:
        return format(v, spec)
    except (ValueError, TypeError):
        return str(v)


def check_high_intensity(ctx: ProviderContext) -> RuleCheckResult:
    pct = _pct_rank(ctx, "srvcs_per_bene")
    val = _value(ctx, "srvcs_per_bene")
    p95 = _specialty_p95(ctx, "srvcs_per_bene")
    triggered = pct is not None and pct >= 99.0
    if triggered:
        evidence = (
            f"srvcs_per_bene={_fmt(val)} at P{_fmt(pct, '.0f')} within specialty "
            f"(national P95={_fmt(p95)})"
        )
    else:
        evidence = f"srvcs_per_bene={_fmt(val)} at P{_fmt(pct, '.0f')} -- within normal specialty range"
    return RuleCheckResult(
        rule_id="HIGH_INTENSITY",
        rule_name="Excessive Services Per Beneficiary",
        triggered=triggered,
        severity="medium",
        evidence=evidence,
        reference="OIG Data Brief OEI-03-17-00470",
        available=True,
    )


def check_volume_spike(ctx: ProviderContext) -> RuleCheckResult:
    history = ctx.history
    if not history:
        return RuleCheckResult("VOLUME_SPIKE", "Sudden Volume Increase", False,
                               "high", "no history available",
                               "CMS FPS Algorithm", available=False)
    current = history[-1]
    yoy_vol  = current.get("yoy_volume_change")
    yoy_bene = current.get("yoy_billing_change")  # bene change isn't in history; use billing as proxy
    total    = current.get("total_services") or 0

    triggered = False
    if yoy_vol is not None and total >= 1000 and yoy_vol > 2.0:
        triggered = True
        evidence = f"yoy_volume={_fmt(yoy_vol, '+.0%')} (current services={_fmt(total, '.0f')})"
        if yoy_bene is not None:
            evidence += f", yoy_billing={_fmt(yoy_bene, '+.0%')}"
    else:
        evidence = f"yoy_volume={_fmt(yoy_vol, '+.0%')} -- within normal variation"
    return RuleCheckResult(
        rule_id="VOLUME_SPIKE",
        rule_name="Sudden Volume Increase",
        triggered=triggered,
        severity="high",
        evidence=evidence,
        reference="CMS FPS Algorithm",
        available=True,
    )


def check_charge_inflation(ctx: ProviderContext) -> RuleCheckResult:
    ratio = _value(ctx, "charge_to_allowed_ratio")
    pct   = _pct_rank(ctx, "charge_to_allowed_ratio")
    p95   = _specialty_p95(ctx, "charge_to_allowed_ratio")
    triggered = False
    if ratio is not None and p95 is not None:
        triggered = ratio > p95 * 1.5
    if ratio is not None and pct is not None and p95 is not None:
        evidence = (
            f"charge_to_allowed_ratio={_fmt(ratio)} at P{_fmt(pct, '.0f')} "
            f"(specialty P95={_fmt(p95)}, trigger at P95*1.5 = {_fmt(p95 * 1.5)})"
        )
    else:
        evidence = "insufficient data"
    return RuleCheckResult(
        rule_id="CHARGE_INFLATION",
        rule_name="Excessive Charge Markup",
        triggered=triggered,
        severity="low",
        evidence=evidence,
        reference="CMS Limiting Charge Policy",
        available=True,
    )


def check_procedure_concentration(ctx: ProviderContext) -> RuleCheckResult:
    hhi = _value(ctx, "herfindahl_index")
    pct = _pct_rank(ctx, "herfindahl_index")
    p95 = _specialty_p95(ctx, "herfindahl_index")
    triggered = pct is not None and pct >= 99.0
    n_hcpcs = _value(ctx, "n_unique_hcpcs")
    if hhi is not None and pct is not None:
        evidence = (
            f"herfindahl={_fmt(hhi, '.3f')} at P{_fmt(pct, '.0f')} "
            f"(specialty P95={_fmt(p95, '.3f')}), n_unique_hcpcs={_fmt(n_hcpcs, '.0f')}"
        )
    else:
        evidence = "insufficient data"
    return RuleCheckResult(
        rule_id="PROCEDURE_CONCENTRATION",
        rule_name="Extreme Procedure Concentration",
        triggered=triggered,
        severity="medium",
        evidence=evidence,
        reference="Derived from CMS billing pattern analysis",
        available=True,
    )


# --- Rules that require data not in current dataset ----------------------


def check_impossible_day(ctx: ProviderContext) -> RuleCheckResult:
    return RuleCheckResult(
        rule_id="IMPOSSIBLE_DAY",
        rule_name="Impossible Day Billing",
        triggered=False,
        severity="critical",
        evidence="Cannot evaluate: CMS public data is annual aggregate, no date-of-service field.",
        reference="OIG Report OEI-04-11-00680",
        available=False,
    )


# Minimum established-visit volume required for the ratio to be stable.
# Keep aligned with em_distribution.MIN_EST_VOLUME.
UPCODING_MIN_EST_VOLUME = 50


def check_upcoding(ctx: ProviderContext) -> RuleCheckResult:
    """High-tier E&M share outlier: 99214/99215 share > specialty P95.

    Unlocked by anomaly/rules/em_distribution.py, which pre-computes per-NPI
    E&M counts and per-specialty benchmark percentiles. If either input is
    missing the rule transparently reports NOT EVALUABLE.
    """
    if not ctx.data_available.get("em_distribution"):
        return RuleCheckResult(
            rule_id="UPCODING",
            rule_name="Systematic Upcoding",
            triggered=False,
            severity="high",
            evidence=(
                "Cannot evaluate: em_distributions table not available. Build "
                "with anomaly/rules/em_distribution.py."
            ),
            reference="CMS MLN Matters SE1418",
            available=False,
        )

    est_total = ctx.metrics.get("em_est_total")
    est_high_pct = ctx.metrics.get("em_est_high_pct")
    bench = ctx.specialty_national.get("em_est_high_pct", {})
    p95 = bench.get("p95")

    # Low-volume short-circuit: the ratio is too noisy to act on. Note that
    # est_total=None means the provider billed zero established office visits
    # this year; est_total<threshold means they billed some but not enough.
    if est_total is None:
        return RuleCheckResult(
            rule_id="UPCODING",
            rule_name="Systematic Upcoding",
            triggered=False,
            severity="high",
            evidence="No established office-visit volume (99211-99215) this year.",
            reference="CMS MLN Matters SE1418",
            available=True,
        )
    if est_total < UPCODING_MIN_EST_VOLUME:
        return RuleCheckResult(
            rule_id="UPCODING",
            rule_name="Systematic Upcoding",
            triggered=False,
            severity="high",
            evidence=(
                f"Insufficient E&M volume: est_total={int(est_total)} "
                f"(min {UPCODING_MIN_EST_VOLUME}). Ratio is unstable at low n."
            ),
            reference="CMS MLN Matters SE1418",
            available=True,
        )

    if est_high_pct is None or p95 is None:
        return RuleCheckResult(
            rule_id="UPCODING",
            rule_name="Systematic Upcoding",
            triggered=False,
            severity="high",
            evidence="Specialty benchmark missing for E&M high-tier share.",
            reference="CMS MLN Matters SE1418",
            available=False,
        )

    # Saturated-specialty short-circuit: if the top 5% of the peer group
    # already bills 99214/99215 exclusively, the high-tier share cannot
    # discriminate. Report transparently rather than returning a false
    # negative. This is the case for ~80% of specialties (IM, Cardiology,
    # Neurology, Psychiatry, etc. where 99215 is the default established
    # visit).
    if p95 >= 0.99:
        return RuleCheckResult(
            rule_id="UPCODING",
            rule_name="Systematic Upcoding",
            triggered=False,
            severity="high",
            evidence=(
                f"Cannot discriminate: specialty P95 for high-tier E&M share "
                f"is {p95*100:.1f}% (i.e., the norm is already saturated at "
                f"99214/99215). Provider high-tier share is "
                f"{est_high_pct*100:.1f}% of {int(est_total):,} established "
                f"visits. Consider YoY drift or absolute volume instead."
            ),
            reference="CMS MLN Matters SE1418",
            available=True,
        )

    # Trigger when high-tier share exceeds specialty P95 in a non-saturated
    # specialty. We do NOT add a separate absolute-share floor because P95 is
    # already specialty-aware: in a specialty where P95=35%, a 45% share is
    # meaningfully high; in a specialty where P95=60%, a 45% share is not.
    triggered = est_high_pct > p95
    evidence = (
        f"99214+99215 share = {est_high_pct*100:.1f}% of "
        f"{int(est_total):,} established visits "
        f"(specialty P95={p95*100:.1f}%, trigger at > P95)"
    )
    return RuleCheckResult(
        rule_id="UPCODING",
        rule_name="Systematic Upcoding",
        triggered=triggered,
        severity="high",
        evidence=evidence,
        reference="CMS MLN Matters SE1418",
        available=True,
    )


def check_unbundling(ctx: ProviderContext) -> RuleCheckResult:
    return RuleCheckResult(
        rule_id="UNBUNDLING",
        rule_name="Service Unbundling",
        triggered=False,
        severity="high",
        evidence="Cannot evaluate: requires per-encounter claim grouping, which is not available in CMS public Provider & Service dataset.",
        reference="CMS NCCI Edits",
        available=False,
    )


OUT_OF_SPECIALTY_THRESHOLD = 0.20  # 20% of services on codes outside specialty scope


def check_out_of_specialty(ctx: ProviderContext) -> RuleCheckResult:
    # Retriever only populates out_of_specialty_pct when the specialty_scopes
    # table is available and the specialty was covered in the scan.
    if not ctx.data_available.get("out_of_specialty") or "out_of_specialty_pct" not in ctx.metrics:
        return RuleCheckResult(
            rule_id="OUT_OF_SPECIALTY",
            rule_name="Out-of-Specialty Billing",
            triggered=False,
            severity="medium",
            evidence=(
                "Cannot evaluate: specialty-HCPCS scope table not available for "
                "this specialty. Build with anomaly/rules/specialty_scopes.py."
            ),
            reference="42 CFR 424.22",
            available=False,
        )

    pct = ctx.metrics["out_of_specialty_pct"]
    triggered = pct > OUT_OF_SPECIALTY_THRESHOLD
    n_codes  = len(ctx.out_of_specialty_codes)
    preview  = ", ".join(ctx.out_of_specialty_codes[:5])
    if preview:
        preview = f" (top out-of-scope: {preview}{'...' if n_codes > 5 else ''})"
    evidence = (
        f"{_fmt(pct*100, '.1f')}% of services on codes outside {ctx.specialty} scope "
        f"(trigger at >{OUT_OF_SPECIALTY_THRESHOLD*100:.0f}%); "
        f"{n_codes} distinct out-of-scope code(s){preview}"
    )
    return RuleCheckResult(
        rule_id="OUT_OF_SPECIALTY",
        rule_name="Out-of-Specialty Billing",
        triggered=triggered,
        severity="medium",
        evidence=evidence,
        reference="42 CFR 424.22",
        available=True,
    )


def check_leie_excluded(ctx: ProviderContext) -> RuleCheckResult:
    """OIG LEIE cross-reference: NPI match on the exclusion list.

    An exclusion is a conviction-grade signal and should top-rank any flagged
    provider. Not in the spec's original 9 rules but a natural addition once
    the LEIE loader (anomaly/external/leie_loader.py) has run.
    """
    if not ctx.data_available.get("leie"):
        return RuleCheckResult(
            rule_id="LEIE_EXCLUDED",
            rule_name="OIG Exclusion List Match",
            triggered=False,
            severity="critical",
            evidence=(
                "Cannot evaluate: LEIE exclusion list not loaded. Fetch with "
                "anomaly/external/leie_loader.py."
            ),
            reference="OIG LEIE 42 USC 1320a-7",
            available=False,
        )

    if ctx.leie_record is None:
        return RuleCheckResult(
            rule_id="LEIE_EXCLUDED",
            rule_name="OIG Exclusion List Match",
            triggered=False,
            severity="critical",
            evidence="NPI not found on OIG LEIE at time of agent run.",
            reference="OIG LEIE 42 USC 1320a-7",
            available=True,
        )

    rec = ctx.leie_record
    # Active vs. reinstated: a present, non-empty REINDATE indicates the
    # individual was reinstated after the exclusion period. An active exclusion
    # is the most severe signal; a reinstatement is still worth flagging as
    # historical context but not a trigger.
    reinstated = bool(rec.get("reinstate_date"))
    excl_date  = rec.get("exclusion_date") or ""
    excl_type  = rec.get("exclusion_type") or ""
    if reinstated:
        evidence = (
            f"Historical exclusion: {excl_type} on {excl_date}, reinstated "
            f"{rec.get('reinstate_date')}. Active exclusion no longer in force."
        )
        triggered = False
    else:
        evidence = (
            f"ACTIVE exclusion on OIG LEIE: type={excl_type}, date={excl_date}. "
            f"Any Medicare payment to this NPI is program-ineligible."
        )
        triggered = True
    return RuleCheckResult(
        rule_id="LEIE_EXCLUDED",
        rule_name="OIG Exclusion List Match",
        triggered=triggered,
        severity="critical",
        evidence=evidence,
        reference="OIG LEIE 42 USC 1320a-7",
        available=True,
    )


def check_beneficiary_sharing(ctx: ProviderContext) -> RuleCheckResult:
    return RuleCheckResult(
        rule_id="BENEFICIARY_SHARING",
        rule_name="Unusual Beneficiary Overlap",
        triggered=False,
        severity="critical",
        evidence="Cannot evaluate: requires beneficiary-level claims linkage across providers, not available in public data.",
        reference="Anti-Kickback Statute 42 USC 1320a-7b",
        available=False,
    )


# --- Orchestrator --------------------------------------------------------


RULE_CHECKS = [
    check_leie_excluded,
    check_high_intensity,
    check_volume_spike,
    check_charge_inflation,
    check_procedure_concentration,
    check_impossible_day,
    check_upcoding,
    check_unbundling,
    check_out_of_specialty,
    check_beneficiary_sharing,
]


def evaluate_all(ctx: ProviderContext) -> list[RuleCheckResult]:
    return [rule(ctx) for rule in RULE_CHECKS]


if __name__ == "__main__":
    import argparse
    from retrieve_context import ContextRetriever

    ap = argparse.ArgumentParser()
    ap.add_argument("--npi",  required=True)
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()

    r = ContextRetriever()
    ctx = r.get_context(args.npi, args.year)
    if ctx is None:
        print("NPI/year not found")
        raise SystemExit(1)

    print(f"NPI {ctx.npi} ({ctx.specialty}, {ctx.state}) year {ctx.year}\n")
    for rr in evaluate_all(ctx):
        icon = "X" if rr.triggered else ("-" if rr.available else "?")
        avail = "" if rr.available else " (NOT EVALUABLE)"
        print(f"  [{icon}] {rr.rule_id}: {rr.rule_name}{avail}")
        print(f"      {rr.evidence}")
