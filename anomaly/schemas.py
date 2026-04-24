"""Dataclasses shared across the anomaly agent pipeline (spec sections 4-7)."""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any


@dataclass
class ProviderContext:
    """Spec section 5.1 -- evidence package for a single (NPI, year)."""
    npi: str
    year: int
    specialty: str
    state: str

    # Provider profile
    years_active: int
    risk_score: float | None
    risk_score_specialty_median: float | None

    # Core metrics (from npi_profiles)
    metrics: dict[str, float]

    # Benchmark comparison: dict of {metric: {mean, p5, p25, p50, p75, p95}}
    specialty_national: dict[str, dict[str, float]]
    specialty_state: dict[str, dict[str, float]]

    # Percentile rank within specialty-year: {metric: percentile (0-100)}
    percentile_ranks: dict[str, float]

    # Historical trajectory (oldest first): list of per-year metric dicts
    history: list[dict[str, Any]]
    trend_direction: str   # "increasing" | "decreasing" | "stable" | "spike"

    # Procedure analysis
    top_hcpcs: list[dict[str, Any]]       # [{code, count, pct_of_total}] ranked
    bucket_distribution: dict[str, float] # {bucket_name: pct}
    out_of_specialty_codes: list[str]

    # Data completeness flags -- what we could / could not compute
    data_available: dict[str, bool] = field(default_factory=dict)


@dataclass
class RuleCheckResult:
    """Spec section 6 -- one fraud indicator evaluation."""
    rule_id: str
    rule_name: str
    triggered: bool
    severity: str           # "low" | "medium" | "high" | "critical"
    evidence: str           # concise human-readable explanation
    reference: str = ""
    available: bool = True  # False means rule could not be evaluated with current data


@dataclass
class InvestigationBrief:
    """Spec section 7.2 -- the narrative output of the Claude reasoning step."""
    npi: str
    year: int
    specialty: str
    state: str
    risk_classification: str   # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    risk_score: float          # composite 0-100

    executive_summary: str
    statistical_findings: str
    contextual_interpretation: str
    rule_check_results: str
    data_limitations: str
    recommended_actions: list[str]

    evidence_summary: dict[str, Any]
    generated_at: datetime
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["generated_at"] = self.generated_at.isoformat()
        return d
