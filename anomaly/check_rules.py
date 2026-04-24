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


def check_upcoding(ctx: ProviderContext) -> RuleCheckResult:
    return RuleCheckResult(
        rule_id="UPCODING",
        rule_name="Systematic Upcoding",
        triggered=False,
        severity="high",
        evidence="Cannot evaluate: requires E&M code level distribution (99211-99215 split), which requires per-code filtering not pre-computed in npi_profiles.",
        reference="CMS MLN Matters SE1418",
        available=False,
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


def check_out_of_specialty(ctx: ProviderContext) -> RuleCheckResult:
    return RuleCheckResult(
        rule_id="OUT_OF_SPECIALTY",
        rule_name="Out-of-Specialty Billing",
        triggered=False,
        severity="medium",
        evidence="Cannot evaluate in this run: specialty-HCPCS scope table not yet built. Planned follow-up.",
        reference="42 CFR 424.22",
        available=False,
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
