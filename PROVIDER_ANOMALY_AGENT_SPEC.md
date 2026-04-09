# Provider Anomaly Investigation Agent — Design Specification

> Project: AllowanceMap
> Author: Raj Vedire
> Date: 2026-04-08
> Status: Design phase

---

## 1. Problem Statement

Medicare fraud, waste, and abuse cost the federal government an estimated $60-100 billion annually. Traditional fraud detection relies on rule-based systems that flag obvious violations but miss sophisticated patterns. Provider-level anomaly detection is inherently a reasoning problem: a cardiologist billing 3x the state average might be committing fraud, running a high-volume legitimate practice, or treating an unusually sick patient population. Distinguishing these cases requires contextual reasoning that combines statistical outlier detection with domain knowledge about specialty norms, geographic variation, and patient acuity.

AllowanceMap already has the data infrastructure to support this: 103M provider-service records across 11 years, 131 specialties, 63 states, and NPI-level risk scores. This spec designs an agent that uses that data to identify, contextualize, and triage provider anomalies.

---

## 2. Agent Architecture

### High-Level Workflow

```
Step 1: Outlier Detection (Statistical)
    Compute NPI-level metrics from silver/gold parquets
    Flag providers exceeding specialty/state thresholds
            |
            v
Step 2: Contextual Retrieval (Data Enrichment)
    For each flagged NPI, pull:
    - Specialty benchmarks (national + state)
    - Provider's own historical trend (2013-2023)
    - Patient acuity context (risk score)
    - Procedure mix analysis
            |
            v
Step 3: Rule-Based Cross-Reference (Known Fraud Indicators)
    Check against CMS fraud patterns:
    - Impossible day billing
    - Out-of-specialty procedures
    - Sudden volume spikes
    - Upcoding patterns
            |
            v
Step 4: Investigation Brief Generation (LLM Reasoning)
    Claude synthesizes evidence into narrative brief:
    - Statistical summary
    - Contextual interpretation
    - Risk classification (low/medium/high/critical)
    - Recommended action
            |
            v
Step 5: Human Review Gate
    Present brief to analyst
    Analyst approves, modifies, or dismisses
    Feedback loop for calibration
```

### Agent Type

Built with the **Claude Agent SDK** (Python). The agent orchestrates a multi-tool pipeline where each step feeds the next. Claude's reasoning is invoked at Step 4 to interpret the evidence, not at Steps 1-3 which are deterministic data operations.

---

## 3. Data Requirements

### 3.1 NPI-Level Feature Extraction

The current gold parquets drop `Rndrng_NPI` during feature engineering. The anomaly agent needs a parallel pipeline that preserves NPI for aggregation.

**New script: `compute_npi_profiles.py`**

Input: `local_pipeline/silver/{STATE}.parquet` (NPI preserved at silver layer)

Output: `local_pipeline/anomaly/npi_profiles.parquet`

**NPI-level features to compute:**

| Feature | Formula | Anomaly Signal |
|---|---|---|
| `total_services` | SUM(Tot_Srvcs) per NPI-year | Extreme billing volume |
| `total_beneficiaries` | SUM(Tot_Benes) per NPI-year | Patient volume |
| `total_billing` | SUM(Tot_Srvcs * Avg_Sbmtd_Chrg) per NPI-year | Dollar volume |
| `total_allowed` | SUM(Tot_Srvcs * Avg_Mdcr_Alowd_Amt) per NPI-year | Medicare exposure |
| `srvcs_per_bene` | total_services / total_beneficiaries | Intensity per patient |
| `avg_charge` | MEAN(Avg_Sbmtd_Chrg) per NPI-year | Pricing level |
| `avg_allowed` | MEAN(Avg_Mdcr_Alowd_Amt) per NPI-year | Reimbursement level |
| `charge_to_allowed_ratio` | avg_charge / avg_allowed | Markup ratio |
| `n_unique_hcpcs` | COUNT(DISTINCT HCPCS_Cd) per NPI-year | Service diversity |
| `herfindahl_index` | SUM(service_share^2) across HCPCS per NPI | Procedure concentration |
| `facility_pct` | % services at facility vs office | Setting mix |
| `bucket_distribution` | % services in each hcpcs_bucket (6 values) | Service category mix |
| `risk_score` | Bene_Avg_Risk_Scre per NPI-year | Patient acuity |
| `yoy_volume_change` | (current_year_services - prior_year) / prior_year | Volume trend |
| `yoy_billing_change` | (current_year_billing - prior_year) / prior_year | Billing trend |
| `specialty` | Rndrng_Prvdr_Type per NPI | Provider type |
| `state` | Rndrng_Prvdr_State_Abrvtn per NPI | Geography |

**Estimated output size:** ~10M NPI-year rows (10M NPIs across 11 years, though most NPIs appear in fewer than 11 years)

### 3.2 Benchmark Tables

**Specialty benchmarks:** `anomaly/specialty_benchmarks.parquet`
- Aggregate NPI profiles by (specialty, year): mean, median, P5, P25, P75, P95 for each metric
- ~131 specialties x 11 years = ~1,441 rows

**State-specialty benchmarks:** `anomaly/state_specialty_benchmarks.parquet`
- Aggregate by (specialty, state, year): mean, median, P5, P95
- ~131 x 63 x 11 = ~90K rows (many will be empty)

**National benchmarks:** `anomaly/national_benchmarks.parquet`
- Aggregate by (year): mean, P5, P95 for each metric
- 11 rows

---

## 4. Step 1: Outlier Detection

### 4.1 Statistical Methods

**Method A: Z-Score Flagging (within specialty-state-year groups)**

For each NPI in a given (specialty, state, year) group, compute z-scores for:
- `total_services`
- `srvcs_per_bene`
- `avg_allowed`
- `charge_to_allowed_ratio`
- `n_unique_hcpcs`
- `herfindahl_index`

Flag if any z-score > 3.0 or < -3.0 (adjustable threshold).

**Method B: Isolation Forest (multivariate)**

Train an Isolation Forest on the NPI profile feature vector within each specialty group. Contamination parameter set to 0.01 (expect 1% anomaly rate). Features: all numeric NPI-level metrics normalized within specialty.

Advantage over z-score: catches multivariate anomalies (e.g., normal volume AND normal charges individually, but unusual combination of high volume + low charge that suggests unbundling).

**Method C: Temporal Anomaly Detection**

For NPIs with 3+ years of history, flag:
- Year-over-year volume increase > 100%
- Year-over-year billing increase > 150%
- Services-per-beneficiary ratio increase > 50% in a single year
- Sudden appearance of new HCPCS bucket (e.g., surgeon starts billing lab codes)

### 4.2 Flagging Output

```python
@dataclass
class AnomalyFlag:
    npi: str
    year: int
    specialty: str
    state: str
    flag_type: str        # "z_score" | "isolation_forest" | "temporal"
    flag_reason: str      # human-readable: "services_per_bene z=4.2 (specialty P99=8.1, this NPI=14.3)"
    severity: float       # 0-1 normalized score
    metrics: dict         # all NPI-level metrics for this provider-year
```

**Expected flag rate:** 1-3% of NPI-years (~100K-300K flags from 10M NPI-years). Most will be explainable after contextual analysis.

---

## 5. Step 2: Contextual Retrieval

For each flagged NPI, the agent retrieves contextual data to determine whether the anomaly is explainable.

### 5.1 Context Package

```python
@dataclass
class ProviderContext:
    # Provider profile
    npi: str
    specialty: str
    state: str
    years_active: int                  # number of years in dataset
    risk_score: float                  # current year Bene_Avg_Risk_Scre
    risk_score_specialty_median: float # comparison point

    # Benchmark comparison
    specialty_national: dict           # {metric: {mean, median, p5, p95}}
    specialty_state: dict              # same structure, state-specific
    percentile_ranks: dict             # where this NPI falls: {metric: percentile}

    # Historical trend
    history: list[dict]                # [{year, total_services, total_billing, srvcs_per_bene, ...}]
    trend_direction: str               # "increasing" | "decreasing" | "stable" | "spike"

    # Procedure analysis
    top_hcpcs: list[dict]              # [{code, description, count, pct_of_total}] top 10
    bucket_distribution: dict          # {bucket_name: pct}
    out_of_specialty_codes: list[str]  # HCPCS codes unusual for this specialty

    # Geographic context
    state_provider_density: float      # providers per capita in this state-specialty
    is_rural: bool                     # derived from AHRF if available
```

### 5.2 Retrieval Functions

```python
async def get_provider_context(npi: str, year: int) -> ProviderContext:
    """Retrieve all contextual data for a flagged provider."""

    # 1. Pull NPI profile from npi_profiles.parquet
    profile = query_npi_profile(npi, year)

    # 2. Pull specialty benchmarks
    benchmarks = query_specialty_benchmarks(profile.specialty, profile.state, year)

    # 3. Compute percentile ranks
    percentiles = compute_percentile_ranks(profile, benchmarks)

    # 4. Pull historical trend (all years for this NPI)
    history = query_npi_history(npi)

    # 5. Analyze procedure mix
    procedure_analysis = analyze_procedure_mix(npi, year, profile.specialty)

    # 6. Geographic context (if AHRF data available)
    geo_context = query_geographic_context(profile.state, profile.specialty)

    return ProviderContext(...)
```

---

## 6. Step 3: Rule-Based Cross-Reference

### 6.1 Known Fraud Indicators

These rules encode known Medicare fraud patterns documented by the OIG (Office of Inspector General) and CMS:

```python
FRAUD_RULES = [
    {
        "id": "IMPOSSIBLE_DAY",
        "name": "Impossible Day Billing",
        "description": "Provider bills more than 24 hours of services in a single day",
        "check": lambda npi_profile: npi_profile.max_daily_services > SPECIALTY_DAY_LIMITS[npi_profile.specialty],
        "severity": "critical",
        "reference": "OIG Report OEI-04-11-00680"
    },
    {
        "id": "UPCODING",
        "name": "Systematic Upcoding",
        "description": "Provider consistently bills highest-level E&M codes (99215/99205) at rates far above specialty average",
        "check": lambda npi_profile: npi_profile.high_level_em_pct > specialty_p95_high_em,
        "severity": "high",
        "reference": "CMS MLN Matters SE1418"
    },
    {
        "id": "UNBUNDLING",
        "name": "Service Unbundling",
        "description": "Provider bills separately for components that should be bundled under a single code",
        "check": lambda npi_profile: npi_profile.avg_codes_per_encounter > specialty_p99,
        "severity": "high",
        "reference": "CMS NCCI Edits"
    },
    {
        "id": "OUT_OF_SPECIALTY",
        "name": "Out-of-Specialty Billing",
        "description": "Provider bills significant volume of codes outside their designated specialty scope",
        "check": lambda npi_profile: npi_profile.out_of_specialty_pct > 0.20,
        "severity": "medium",
        "reference": "42 CFR 424.22"
    },
    {
        "id": "VOLUME_SPIKE",
        "name": "Sudden Volume Increase",
        "description": "Year-over-year service volume increase exceeds 200% without corresponding beneficiary increase",
        "check": lambda npi_profile: npi_profile.yoy_volume_change > 2.0 and npi_profile.yoy_bene_change < 0.5,
        "severity": "high",
        "reference": "CMS FPS Algorithm"
    },
    {
        "id": "HIGH_INTENSITY",
        "name": "Excessive Services Per Beneficiary",
        "description": "Provider delivers significantly more services per patient than specialty peers",
        "check": lambda npi_profile: npi_profile.srvcs_per_bene_zscore > 4.0,
        "severity": "medium",
        "reference": "OIG Data Brief OEI-03-17-00470"
    },
    {
        "id": "CHARGE_INFLATION",
        "name": "Excessive Charge Markup",
        "description": "Provider submits charges far exceeding Medicare fee schedule, suggesting inflated pricing",
        "check": lambda npi_profile: npi_profile.charge_to_allowed_ratio > specialty_p99_ratio * 1.5,
        "severity": "low",
        "reference": "CMS Limiting Charge Policy"
    },
    {
        "id": "BENEFICIARY_SHARING",
        "name": "Unusual Beneficiary Overlap",
        "description": "Provider shares an unusually high percentage of beneficiaries with a small number of other providers (potential kickback ring)",
        "check": "requires_claims_data",
        "severity": "critical",
        "reference": "Anti-Kickback Statute 42 USC 1320a-7b",
        "note": "Not implementable with current public data — requires beneficiary-level claims"
    },
]
```

### 6.2 Scope Limitations

Several important fraud indicators cannot be checked with the current public dataset:

| Indicator | Requires | Available? |
|---|---|---|
| Phantom billing (services never rendered) | Beneficiary confirmation | No |
| Beneficiary sharing rings | Beneficiary-level claims linkage | No |
| Identity theft (stolen NPIs) | NPPES + claims cross-reference | Partial (NPPES is available) |
| Duplicate billing | Claim-level dates of service | No |
| Medical necessity | Diagnosis codes linked to procedures | No |

The agent should clearly state in its investigation briefs which checks could and could not be performed.

---

## 7. Step 4: Investigation Brief Generation

### 7.1 LLM Prompt Design

The Claude agent receives the structured evidence from Steps 1-3 and generates a narrative investigation brief.

**System prompt:**

```
You are a Medicare fraud investigation analyst. You receive statistical
evidence about provider billing patterns and generate investigation briefs.

Your role is to INTERPRET evidence, not to accuse. You must:
1. Distinguish between fraud indicators and legitimate explanations
2. Consider specialty norms, patient acuity, and geographic context
3. Assign a risk classification based on the totality of evidence
4. Recommend specific next steps proportional to the risk level
5. Note which checks could not be performed due to data limitations

Risk classifications:
- LOW: Statistical outlier with plausible explanations. No action needed.
- MEDIUM: Multiple anomaly signals with partial explanations. Flag for
  periodic monitoring.
- HIGH: Strong anomaly signals without adequate explanations. Recommend
  focused review.
- CRITICAL: Multiple severe indicators suggesting potential fraud.
  Recommend immediate investigation referral.

Never conclude fraud definitively. You are generating leads for human
analysts, not making legal determinations.
```

**User prompt template:**

```
Generate an investigation brief for the following provider:

## Provider Profile
NPI: {npi}
Specialty: {specialty}
State: {state}
Years in Dataset: {years_active}
Beneficiary Risk Score: {risk_score} (specialty median: {specialty_risk_median})

## Anomaly Flags
{formatted_flags}

## Benchmark Comparison
{formatted_percentile_table}

## Historical Trend
{formatted_history_table}

## Procedure Analysis
Top procedures: {top_hcpcs}
Out-of-specialty codes: {out_of_specialty_codes}
Service category distribution: {bucket_distribution}

## Rule Check Results
{formatted_rule_results}

## Data Limitations
The following checks could not be performed with available data:
- Beneficiary-level claims linkage
- Date-of-service analysis
- Diagnosis code appropriateness
- Duplicate billing detection

Generate the investigation brief.
```

### 7.2 Brief Output Schema

```python
@dataclass
class InvestigationBrief:
    npi: str
    year: int
    risk_classification: str           # LOW | MEDIUM | HIGH | CRITICAL
    risk_score: float                  # 0-100 composite score

    executive_summary: str             # 2-3 sentence overview
    statistical_findings: str          # detailed statistical analysis
    contextual_interpretation: str     # what explains vs. doesn't explain the anomalies
    rule_check_results: str            # which known fraud patterns were triggered
    data_limitations: str              # what we couldn't check
    recommended_actions: list[str]     # specific next steps

    evidence_summary: dict             # structured data backing the narrative
    generated_at: datetime
    model_version: str                 # Claude model used
```

### 7.3 Example Output

```markdown
# Investigation Brief: NPI 1234567890

**Risk Classification: HIGH**
**Composite Risk Score: 72/100**

## Executive Summary
Dr. [redacted], a cardiologist in Florida, exhibits billing patterns
significantly above specialty and state norms across multiple dimensions.
Services per beneficiary (14.3) is 4.2 standard deviations above the
Florida cardiology median (6.8), and year-over-year billing increased
217% from 2021 to 2022 without a proportional increase in beneficiary
count (+12%). The provider's beneficiary risk score (1.85) is above
average but does not fully account for the volume discrepancy.

## Statistical Findings
- Services per beneficiary: 14.3 (specialty-state P99: 11.2) — FLAGGED
- Total billing: $2.4M (specialty-state P95: $1.8M) — ELEVATED
- Charge-to-allowed ratio: 3.8x (specialty median: 2.9x) — ELEVATED
- Procedure concentration (Herfindahl): 0.08 (diverse mix, not suspicious)
- Unique HCPCS codes: 47 (specialty median: 38) — NORMAL

## Contextual Interpretation
The high services-per-beneficiary ratio is the primary concern. While
the provider's beneficiary risk score (1.85) suggests a sicker-than-average
patient panel, this accounts for approximately 30% of the volume excess.
The 217% billing increase in 2022 coincides with no major practice
changes visible in the data. The procedure mix is diverse and
consistent with cardiology scope, which argues against simple upcoding.

## Rule Check Results
- VOLUME_SPIKE: TRIGGERED — 217% YoY increase with only 12% beneficiary growth
- HIGH_INTENSITY: TRIGGERED — srvcs_per_bene z-score = 4.2
- UPCODING: NOT TRIGGERED — E&M level distribution within norms
- OUT_OF_SPECIALTY: NOT TRIGGERED — 98% of codes are cardiology-appropriate
- CHARGE_INFLATION: NOT TRIGGERED — markup ratio within P95

## Data Limitations
This analysis cannot determine: whether services were medically necessary,
whether any services were phantom-billed, whether billing dates are
plausible, or whether beneficiary overlap patterns are unusual.

## Recommended Actions
1. Request detailed claims data for 2022 to examine date-of-service patterns
2. Compare beneficiary panel against cardiac diagnosis codes for medical necessity
3. Monitor 2023 billing for continuation of elevated patterns
4. Flag for inclusion in next routine audit cycle
```

---

## 8. Step 5: Human Review Gate

### 8.1 Review Interface

The agent presents investigation briefs through one of:

**Option A: CLI Interactive Mode**
```
=== Investigation Brief: NPI 1234567890 ===
Risk: HIGH (72/100)
Specialty: Cardiology | State: FL | Flags: VOLUME_SPIKE, HIGH_INTENSITY

[Summary displayed]

Actions:
  [A] Approve and add to investigation queue
  [M] Modify risk classification
  [D] Dismiss (false positive)
  [S] Skip (review later)
  [V] View full brief

Choice:
```

**Option B: Web Dashboard (AllowanceMap Extension)**
- New page: `/investigations`
- Table of flagged providers with risk scores, sortable/filterable
- Click to expand investigation brief
- Approve/dismiss/escalate buttons
- Feedback captured for model calibration

**Option C: Batch Report**
- Generate a PDF/HTML report of top N highest-risk providers
- Include all investigation briefs with evidence
- Suitable for periodic review (weekly/monthly)

### 8.2 Feedback Loop

Human decisions (approve/dismiss) are logged and used to calibrate:
- Threshold adjustment: if analysts consistently dismiss z > 3.0 flags for a specialty, raise the threshold for that specialty
- Rule weighting: if VOLUME_SPIKE is confirmed 80% of the time but CHARGE_INFLATION only 5%, weight accordingly
- False positive tracking: specialty-specific false positive rates inform future filtering

```python
@dataclass
class ReviewDecision:
    npi: str
    year: int
    risk_classification_original: str
    analyst_decision: str              # "approve" | "dismiss" | "modify"
    analyst_classification: str        # analyst's risk assessment (may differ)
    analyst_notes: str                 # free text
    reviewed_at: datetime
    analyst_id: str
```

---

## 9. Technical Implementation

### 9.1 File Structure

```
anomaly/
├── compute_npi_profiles.py           # Step 0: NPI-level feature extraction from silver
├── compute_benchmarks.py             # Step 0: Specialty/state/national benchmarks
├── detect_outliers.py                # Step 1: Z-score, Isolation Forest, temporal
├── retrieve_context.py               # Step 2: Context package assembly
├── check_rules.py                    # Step 3: Known fraud indicator rules
├── generate_brief.py                 # Step 4: Claude API brief generation
├── review_gateway.py                 # Step 5: CLI interactive review
├── agent.py                          # Orchestrator: runs Steps 1-5 end-to-end
├── config.py                         # Thresholds, API keys, model selection
├── schemas.py                        # Dataclass definitions
└── rules/
    ├── fraud_indicators.py           # Rule definitions
    └── specialty_scopes.py           # Expected HCPCS codes per specialty
```

### 9.2 Dependencies

```
# Core
pandas >= 2.0
pyarrow >= 12.0
scikit-learn >= 1.3        # Isolation Forest
numpy >= 1.24

# Agent
anthropic >= 0.40          # Claude API client (or claude_agent_sdk)

# Optional
rich >= 13.0               # CLI formatting for review interface
```

### 9.3 Agent SDK Integration

```python
import anthropic
from claude_agent_sdk import Agent, Tool

# Define tools the agent can call
tools = [
    Tool(
        name="query_npi_profile",
        description="Retrieve NPI-level billing metrics for a specific provider and year",
        input_schema={...},
        handler=query_npi_profile,
    ),
    Tool(
        name="query_benchmarks",
        description="Retrieve specialty and state benchmarks for comparison",
        input_schema={...},
        handler=query_specialty_benchmarks,
    ),
    Tool(
        name="check_fraud_rules",
        description="Run known fraud indicator checks against provider profile",
        input_schema={...},
        handler=check_fraud_rules,
    ),
    Tool(
        name="query_history",
        description="Retrieve historical billing trend for an NPI across all years",
        input_schema={...},
        handler=query_npi_history,
    ),
]

agent = Agent(
    model="claude-sonnet-4-6",         # Fast enough for brief generation
    system=INVESTIGATION_SYSTEM_PROMPT,
    tools=tools,
    max_turns=10,
)
```

### 9.4 Performance Estimates

| Step | Data Size | Time Estimate | Compute |
|---|---|---|---|
| NPI profile extraction | 103M rows (silver) | 30-60 min | CPU, 16GB RAM |
| Benchmark computation | 10M NPI-year rows | 5 min | CPU |
| Outlier detection | 10M NPI-year rows | 10-20 min | CPU (Isolation Forest) |
| Context retrieval | ~100K flagged NPIs | 5-10 min | Parquet lookups |
| Rule checking | ~100K flagged NPIs | 2-5 min | CPU |
| Brief generation | Top 100-1000 NPIs | 5-30 min | Claude API |
| Human review | Top 50-100 briefs | Manual | Analyst time |

**Total automated pipeline:** ~1-2 hours for full run on 11 years of data
**Claude API cost:** ~$0.50-5.00 for 100-1000 briefs (Sonnet, ~1K input + 500 output tokens per brief)

---

## 10. Supabase Integration (Optional)

For the web dashboard version, add tables to the existing AllowanceMap Supabase project:

```sql
CREATE TABLE anomaly_flags (
    id              SERIAL PRIMARY KEY,
    npi             TEXT NOT NULL,
    year            SMALLINT NOT NULL,
    specialty       TEXT NOT NULL,
    state           TEXT NOT NULL,
    flag_type       TEXT NOT NULL,
    flag_reason     TEXT NOT NULL,
    severity        REAL NOT NULL,
    metrics         JSONB,
    UNIQUE (npi, year, flag_type)
);

CREATE TABLE investigation_briefs (
    id                  SERIAL PRIMARY KEY,
    npi                 TEXT NOT NULL,
    year                SMALLINT NOT NULL,
    risk_classification TEXT NOT NULL,
    risk_score          REAL NOT NULL,
    executive_summary   TEXT NOT NULL,
    full_brief          TEXT NOT NULL,
    evidence            JSONB,
    generated_at        TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (npi, year)
);

CREATE TABLE review_decisions (
    id                      SERIAL PRIMARY KEY,
    brief_id                INTEGER REFERENCES investigation_briefs(id),
    analyst_decision        TEXT NOT NULL,
    analyst_classification  TEXT,
    analyst_notes           TEXT,
    reviewed_at             TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 11. Ethical Considerations

### 11.1 False Positives

Statistical outlier detection will produce false positives. Providers flagged by this system should never be assumed fraudulent. The agent's output is an investigation lead, not an accusation.

Specific high-false-positive scenarios:
- **Rural solo practitioners:** Low peer group size makes z-scores unreliable (a group of 3 cardiologists in Wyoming has no meaningful P95)
- **Subspecialists:** A Mohs surgeon will look like an outlier compared to general dermatologists
- **New practices:** First-year providers have unusual volume patterns (ramp-up phase)
- **Telehealth expansion (2020+):** COVID-era telehealth waivers created legitimate volume spikes

### 11.2 Bias Considerations

- Geographic bias: providers in states with fewer peers have less reliable benchmarks
- Specialty bias: rare specialties (e.g., "Undersea And Hyperbaric Medicine") have too few providers for meaningful statistical comparison
- Temporal bias: COVID years (2020-2021) should be excluded or treated separately in trend analysis

### 11.3 Data Privacy

- CMS public data does not include patient-identifiable information
- NPI is a public identifier but links to real provider names via NPPES
- Investigation briefs should be treated as confidential and not published
- The web dashboard (if built) should require authentication

### 11.4 Disclosure

The agent should include in every brief:
1. This is a statistical screening tool, not a fraud determination
2. Which data sources were used and which were not available
3. The false positive rate for the detection method used
4. A recommendation to consult with compliance and legal before any action

---

## 12. Phased Implementation

### Phase A: Data Foundation (1-2 sessions)
1. Write `compute_npi_profiles.py` — extract NPI-level metrics from silver parquets
2. Write `compute_benchmarks.py` — specialty/state/national benchmarks
3. Validate: spot-check known high-volume specialties (ophthalmology, cardiology)

### Phase B: Detection Engine (1-2 sessions)
4. Implement z-score flagging within specialty-state groups
5. Implement Isolation Forest multivariate detection
6. Implement temporal anomaly rules
7. Validate: check flag rates by specialty (expect 1-3%)

### Phase C: Investigation Agent (1-2 sessions)
8. Implement context retrieval functions
9. Implement fraud indicator rule checks
10. Write Claude API brief generation with system prompt
11. Validate: generate 10 sample briefs, review for quality and accuracy

### Phase D: Review Interface (1 session)
12. Build CLI interactive review mode
13. Implement feedback logging
14. Optional: add `/investigations` page to AllowanceMap web app

### Phase E: Calibration (ongoing)
15. Run full pipeline on 2023 data
16. Review top 50 briefs manually
17. Adjust thresholds based on false positive analysis
18. Document specialty-specific threshold overrides

---

## 13. Success Criteria

| Metric | Target |
|---|---|
| Flag rate | 1-3% of NPI-years |
| Brief generation time | < 30 seconds per provider |
| False positive rate (analyst review) | < 50% of HIGH/CRITICAL flags |
| Coverage of known fraud patterns | 5+ of 7 rule categories implemented |
| Brief quality (analyst rating) | > 4/5 on usefulness scale |
| End-to-end pipeline time | < 2 hours for full dataset |

---

## 14. Limitations and Future Work

### Current Data Limitations
- No claim-level date-of-service data (cannot detect impossible day billing precisely)
- No diagnosis codes (cannot assess medical necessity)
- No beneficiary-level linkage (cannot detect referral rings)
- Public data has CMS cell-size suppression (providers with < 11 beneficiaries excluded)

### Future Enhancements
- Integration with NPPES for provider identity verification
- Integration with OIG LEIE (List of Excluded Individuals/Entities) for cross-reference
- Network analysis: build provider referral graphs from shared beneficiary patterns (requires claims data)
- Longitudinal tracking: monitor flagged providers across years for pattern persistence
- Specialty-specific models: train separate anomaly detectors per specialty for better calibration
- Real-time monitoring: if connected to streaming claims data, run detection on rolling windows
