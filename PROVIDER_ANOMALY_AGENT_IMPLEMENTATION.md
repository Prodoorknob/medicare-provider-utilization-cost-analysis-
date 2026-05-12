# Provider Anomaly Investigation Agent — Implementation Doc

> Companion to [`PROVIDER_ANOMALY_AGENT_SPEC.md`](PROVIDER_ANOMALY_AGENT_SPEC.md).
> Spec described the design; this doc describes what shipped and how to operate it.
> Status: Phases A–E complete on `main`. **8 of 11 rules evaluable** (REVENUE_DEVIATION added 2026-05; uses production LightGBM Stage 1 R²=0.943 as a per-row baseline). Last full validation run 2026-04 (top-100 2023 flags, $3.82 spend).

---

## 1. What this is

A statistical-screen → contextual-evidence-package → rule-cross-reference → Claude-narrative pipeline that takes 103M CMS Provider & Service rows and produces investigation briefs on individual NPIs.

The agent's job is **not** to decide whether fraud occurred. Its job is to surface the small subset of provider-years where the joint statistical signature, peer-benchmark context, historical trajectory, and known fraud-rule triggers warrant a human analyst's attention — and to write that argument up coherently. Claude is invoked **once** per provider-year, at the very last stage, to interpret evidence; everything upstream is deterministic data processing.

---

## 2. Pipeline at a glance

```
  silver/{STATE}.parquet  (NPI × HCPCS × POS × year — preserved at silver layer)
            │
            │  compute_npi_profiles.py          (~3 min)
            ▼
  npi_profiles.parquet                          (1.76M NPIs × 11 yrs = 11.5M rows, 22 metrics)
            │
            ├─► compute_benchmarks.py           (~30s)
            │       specialty_benchmarks.parquet            (specialty × year)
            │       state_specialty_benchmarks.parquet      (specialty × state × year)
            │       national_benchmarks.parquet             (year)
            │
            ├─► rules/specialty_scopes.py       (one-time scan; HCPCS whitelist per specialty)
            ├─► rules/em_distribution.py        (one-time scan; per-NPI E&M counts + spec P-stats)
            ├─► rules/revenue_residuals.py      (silver -> LightGBM Stage 1 scoring -> per-NPI revenue_ratio)
            └─► external/leie_loader.py         (OIG LEIE exclusion list, ~83K rows)
            │
            │  detect_outliers.py               (~3 min)
            ▼
  flags.parquet                                 (long-format: NPI × year × method × metric)
            │
            │  agent.py  (rank by composite severity → for each (NPI, year):)
            ▼
  ContextRetriever.get_context(npi, year) ──► ProviderContext
                                                     │
                                       check_rules.evaluate_all(ctx)
                                                     │
                                       generate_brief.generate_brief(...)
                                                     │  Anthropic API call
                                                     ▼
                                  briefs/{NPI}_{YEAR}.md  +  .json
```

Step 1–3 are batch jobs over the silver layer. Step 4 (Claude reasoning) is the only LLM call, and the only step that costs money.

---

## 3. File layout

```
anomaly/
├── __init__.py
├── agent.py                       # orchestrator — rank, retrieve, check, brief
├── schemas.py                     # ProviderContext, RuleCheckResult, InvestigationBrief
├── compute_npi_profiles.py        # silver → 11.5M (NPI, year) profiles
├── compute_benchmarks.py          # profiles → specialty/state/national benchmarks
├── detect_outliers.py             # z-score + Isolation Forest + temporal → flags.parquet
├── retrieve_context.py            # ContextRetriever — assembles ProviderContext
├── check_rules.py                 # 10 fraud-indicator rules (7 evaluable, 3 NOT EVALUABLE)
├── generate_brief.py              # Anthropic API call, prompt-cached system prompt, retry/backoff
├── rules/
│   ├── specialty_scopes.py        # empirical per-specialty HCPCS whitelist
│   ├── em_distribution.py         # per-NPI E&M counts + specialty P-stats (UPCODING)
│   └── revenue_residuals.py       # LightGBM Stage 1 per-row scoring → NPI revenue_ratio (REVENUE_DEVIATION)
└── external/
    └── leie_loader.py             # OIG LEIE downloader
```

Outputs (gitignored) live under `local_pipeline/anomaly/`:

| File | Rows | Notes |
|---|---:|---|
| `npi_profiles.parquet` | 11.52M | 1.76M unique NPIs × up to 11 years |
| `specialty_benchmarks.parquet` | ~1.4K | (specialty, year) — mean + P5/25/50/75/95 |
| `state_specialty_benchmarks.parquet` | ~90K | (specialty, state, year) — same stats |
| `national_benchmarks.parquet` | 11 | one row per year |
| `specialty_scopes.parquet` | 86,924 | (specialty, HCPCS_Cd, in_scope) — 130 specialties, 27% in-scope |
| `em_distributions.parquet` | 5.94M | per-(NPI, year) E&M counts |
| `em_specialty_benchmarks.parquet` | small | (specialty, year) — P50/75/90/95/99 of high-tier share |
| `revenue_residuals.parquet` | ~11.5M | per-(NPI, year) `actual` / `expected` allowed dollars + ratio (Stage 1 LightGBM) |
| `leie_exclusions.parquet` | ~83K | ~8.4K have a valid NPI (others are pre-NPI historical) |
| `flags.parquet` | long format | one row per (NPI, year, method, metric) hit |
| `briefs/{NPI}_{YEAR}.{md,json}` | varies | output artifacts |

---

## 4. Data layer (Stage 0 — `compute_npi_profiles.py`)

The silver-layer grain (NPI × HCPCS × POS × year) is aggregated to **(NPI, year)** with 22 metrics. The 6 dimensions worth knowing:

- **Volume:** `total_services`, `total_beneficiaries`, `total_billing`, `total_allowed`
- **Intensity:** `srvcs_per_bene`, `avg_charge`, `avg_allowed`, `charge_to_allowed_ratio`
- **Mix:** `n_unique_hcpcs`, `herfindahl_index`, `facility_pct`, `bucket_{0..5}_pct`
- **Population acuity:** `risk_score` (CMS HCC, joined from CMS "by Provider" dataset; missing → NaN, NOT imputed at this layer)
- **Trajectory:** `yoy_volume_change`, `yoy_billing_change`, `yoy_bene_change` (within-NPI shift)
- **Keys:** `Rndrng_NPI`, `year`, `specialty`, `state`

HCPCS bucket logic is mirrored verbatim from `notebooks/03_gold_features_local.py` so profile-level analysis is consistent with the rest of the pipeline. Herfindahl is computed as `Σ(service_share²)` over distinct HCPCS within the (NPI, year) group. Bucket 5 is reserved for any HCPCS Level II alphabetic code (A/B/C/E/G/H/J/K/L/M/Q/R/S/T/V — supplies, drugs-by-unit, DME, COVID-era codes); the prompt explicitly tells Claude this bucket frequently produces high `srvcs_per_bene` for benign reasons (unit-of-measure billing).

---

## 5. Detection (Stage 1 — `detect_outliers.py`)

Three methods, each producing rows into the unified `flags.parquet`:

### 5.1 Z-score (Method A)

Six heavy-tailed metrics are z-scored within (specialty, state, year) groups, with a fallback to (specialty, year) when the state-level group has fewer than 30 providers. log1p is applied to five of the six metrics before computing μ/σ to keep the 3σ tail at ~1% rather than over-firing on lognormal distributions; `herfindahl_index` is already bounded in [0, 1] and is z-scored linearly. Threshold default is `|z| > 3.0`. Severity = `min(|z|/10, 1.0)`.

Metrics z-scored: `total_services`, `srvcs_per_bene`, `avg_allowed`, `charge_to_allowed_ratio`, `n_unique_hcpcs`, `herfindahl_index`.

### 5.2 Isolation Forest (Method B)

One model per specialty (pooled across years), `contamination=0.01`, only specialties with ≥ 200 (NPI, year) rows are evaluated. Feature vector: 16 numeric profile metrics (volume + intensity + mix + bucket distribution + risk score). NaN/inf are coerced to 0.0 before fitting — IF can't handle missing values and a missing risk score reflects "below CMS suppression threshold," which the model treats as a separate cluster naturally.

Severity = decision-function score, normalized within specialty so the most-anomalous point gets 1.0.

### 5.3 Temporal rules (Method C)

Right-tail-only spike detection on the precomputed `yoy_*` columns:

- `yoy_volume_change > 2.0` (200%+ jump)
- `yoy_billing_change > 2.5` (250%+ jump)
- "Volume spike with flat benes": `yoy_volume_change > 2.0 AND yoy_bene_change < 0.5` — classic upcoding/unbundling signature.

Two guards keep this from over-firing: (a) **absolute-volume gate** of ≥ 1,000 services rules out new-practice ramp-ups (a provider going 10 → 30 services is +200% but not interesting); (b) **2021 is excluded by default** because everyone's YoY reset coming out of COVID, producing meaningless spikes everywhere.

### 5.4 Output shape

```
flags.parquet  (long-format)
columns: Rndrng_NPI, year, specialty, state,
         flag_type ∈ {z_score, isolation_forest, temporal},
         flag_metric, flag_reason, severity ∈ [0,1],
         value, benchmark_mean, benchmark_std
```

Composite flag rate (NPIs hit by ≥ 2 of the 3 methods): **0.77% of NPI-years.** That's the prefilter the agent ranks on.

---

## 6. Context retrieval (Stage 2 — `retrieve_context.py`)

`ContextRetriever` is a long-lived object designed for batches: it loads profiles, benchmarks, and the four sidecar tables once in `__init__`, then caches per-state silver reads lazily. Calling `get_context(npi, year)` returns a `ProviderContext` containing:

- **Current-year metrics** (16 fields from the profile row).
- **Historical trajectory** — every year that NPI appears in the data, chronologically.
- **Trend classifier** — `"spike" | "increasing" | "decreasing" | "stable" | "insufficient_history"`.
- **National + state benchmarks** — `{metric: {mean, p5, p25, p50, p75, p95}}`.
- **Percentile ranks** within (specialty, year) peer group for 10 metrics. These power the rule triggers (e.g., HIGH_INTENSITY fires at percentile ≥ 99).
- **Top 10 HCPCS codes** (loaded from silver, with descriptions). The agent uses this to narrate the procedure mix in plain English.
- **Bucket distribution** in human-readable form (`{"Anesthesia": 0.0, "Surgery": 0.62, "Radiology": 0.02, ...}`).
- **Out-of-specialty codes** (when the specialty_scopes sidecar is loaded).
- **E&M sidecar** — counts and high-tier share, when the em_distribution sidecar is loaded.
- **LEIE record** — full exclusion row when the OIG LEIE sidecar is loaded and the NPI matches.
- **`data_available` dict** — bool flags for every optional data source. This is the single source of truth the rules and prompt formatter consult when deciding whether to evaluate or skip a check.

The retriever degrades gracefully: every sidecar is optional. If `specialty_scopes.parquet` is missing, OUT_OF_SPECIALTY simply reports NOT EVALUABLE and the brief discloses why. Same for E&M and LEIE.

---

## 7. Rules (Stage 3 — `check_rules.py`)

11 rules total. Each evaluator takes a `ProviderContext` and returns a `RuleCheckResult` with `triggered: bool`, `available: bool`, and a one-line `evidence` string ready for the brief.

### Evaluable (8)

| Rule ID | Trigger | Severity | Reference |
|---|---|---|---|
| **LEIE_EXCLUDED** | NPI is on the OIG LEIE with no reinstatement date | CRITICAL | OIG LEIE / 42 USC 1320a-7 |
| **REVENUE_DEVIATION** | `revenue_ratio > 1.30` AND `expected ≥ $50K` AND `n_rows_scored ≥ 10` | HIGH | LightGBM v2 Stage 1 (R²=0.943) |
| **VOLUME_SPIKE** | `yoy_volume > +200%` AND `total_services ≥ 1,000` | HIGH | CMS FPS Algorithm |
| **HIGH_INTENSITY** | `srvcs_per_bene` at specialty-year percentile ≥ 99 | MEDIUM | OIG OEI-03-17-00470 |
| **PROCEDURE_CONCENTRATION** | `herfindahl_index` at specialty-year percentile ≥ 99 | MEDIUM | derived from CMS pattern analysis |
| **CHARGE_INFLATION** | `charge_to_allowed_ratio > specialty_P95 × 1.5` | LOW | CMS Limiting Charge Policy |
| **OUT_OF_SPECIALTY** | > 20% of services on codes outside empirical specialty scope | MEDIUM | 42 CFR 424.22 |
| **UPCODING** | `(99214+99215) / total_est_visits > specialty_P95`, on ≥ 50 established visits, in a specialty whose P95 is < 99% (i.e., not E&M-saturated) | HIGH | CMS MLN Matters SE1418 |

### NOT EVALUABLE (3) — structural data gaps

| Rule ID | What it would need | Severity if it could fire |
|---|---|---|
| **IMPOSSIBLE_DAY** | claim-level date-of-service field | CRITICAL |
| **UNBUNDLING** | per-encounter claim grouping | HIGH |
| **BENEFICIARY_SHARING** | beneficiary-level claims linkage across NPIs | CRITICAL |

These rules return `available=False` with an evidence string explaining the missing data, so the brief can disclose what was NOT checked rather than silently passing. Future Rule #3 (medical-necessity / diagnosis-code linkage) is BLOCKED on a paid LDS/RIF data subscription.

### REVENUE_DEVIATION — what it actually catches

The other 10 rules ask "is this NPI weird vs. a specialty-aggregate benchmark?" REVENUE_DEVIATION asks something different: "given this NPI's *specific* HCPCS / POS / acuity mix, do they realize more allowed dollars than the production Stage 1 LightGBM predicts?"

Per silver row, the agent runs the same 12-feature vector the API uses at inference (specialty_idx, state_idx, HCPCS_idx, hcpcs_bucket, place_of_srvc_flag, Bene_Avg_Risk_Scre, log_srvcs, log_benes, srvcs_per_bene, specialty_bucket, pos_bucket, hcpcs_target_enc — see `api/services/prediction.build_stage1_features`). Predictions are back-transformed via `expm1` and weighted by `Tot_Srvcs`, then summed to the (NPI, year) grain:

```
expected_total_allowed = Σ Tot_Srvcs_i × expm1(booster.predict(features_i))
revenue_ratio          = actual_total_allowed / expected_total_allowed
```

The model is well-calibrated at population scale (full silver scan: **median ratio = 0.992**, P95 = 1.17, P99 = 1.39, P99.9 = 1.82 across 11.52M NPI-years). The actionable tail is at P99+ with both gates active.

**Full-scan production stats (2026-05, 11.52M rows, 19m 15s wall time):**

| Filter | Survivors |
|---|---:|
| `revenue_ratio > 1.30` | 192,172 (1.67%) |
| `+ expected_total_allowed ≥ $50,000` (volume gate) | 44,266 (0.38%) |
| **`+ n_rows_scored ≥ 10` (row-coverage gate, the actual rule)** | **32,922 (0.29%)** |

The row-coverage gate is the second-most important filter. Without it, the top-by-ratio leaderboard is dominated by `All Other Suppliers` / `Independent Diagnostic Testing Facility` NPIs billing $2–21M on 1–3 distinct silver rows — group / facility / supplier aggregates where the model's per-provider features don't apply. The gate removes 11K such artifacts and leaves 33K individual-physician triggers.

**Specialty composition of the 44K pre-row-gate triggers (which the gate inherits):** Diagnostic Radiology (15,971), Rheumatology (7,174), Pathology (2,916), Allergy/Immunology (1,653), Anesthesiology (1,535), Nurse Practitioner (1,504), Internal Medicine (1,384), Emergency Medicine (1,276), CRNA (995), Gastroenterology (750). Every one of these specialties is literature-documented for modifier-heavy or technical/professional-component billing patterns — the rule lines up with fraud-research priors rather than firing randomly.

**Persistent multi-year hits** are the strongest signal in practice. Example from the production run: NPI `1740240753` (Clinical Laboratory, CA) flagged in 2013/2014/2016/2018 with ratios 1.33–1.47 and $5–17M excess per year. Same NPI, four years of consistent over-realization — the kind of multi-year pattern an analyst would want to see first.

**What it catches that the existing rules miss:**

- **Modifier abuse** — `-22` (increased procedural services), `-50` (bilateral), `-59` (distinct procedural service), `-76` (repeat). Modifiers are not in the model's feature set so any allowed-amount uplift they create surfaces as positive residual.
- **Non-E&M code-family upcoding** — prolonged-service codes (99354/99355), advanced imaging variants, high-tier procedure codes outside the E&M family UPCODING covers.
- **POS misreporting** that pumps the facility differential beyond what the recorded POS predicts.

**Caveats baked into the system prompt** so Claude weighs them: (a) the model partially captures patient acuity via HCC risk score but not perfectly — a clinically complex panel can explain some positive residual; (b) sub-specialization within a coarse CMS specialty label can also produce residual without any fraud; (c) the model was trained on 2013–2023 data, so any fraud already prevalent during training has been partially learned as "expected" — this is a real-world false-negative rate, not a bug.

**Feature parity matters more here than anywhere else in the pipeline.** Any drift between the agent-side feature builder and the API-side feature builder biases every residual. The scorer is intentionally written as a vectorized mirror of `api/services/prediction.build_stage1_features` — the docstring at the top of `revenue_residuals.py` calls this out.

### Two more interpretive nuances

**1. Empirical specialty scope, not regulatory scope.** A code is "in scope" for a specialty if (a) ≥ 1% of that specialty's providers bill it, OR (b) it sits in the specialty's cumulative top-99% of service volume. This is observational. An optometrist billing 66984 (cataract surgery) is NOT flagged because 28% of optometrists in CMS data bill it (co-management, scope-of-practice expansions). Regulatory-scope arguments are explicitly out of scope for this rule — the rule narrates *peer divergence*, the brief narrates *whether divergence is concerning*.

**2. UPCODING short-circuits on saturated specialties.** In ~80% of CMS specialties (Internal Medicine, Cardiology, Neurology, Psychiatry, etc.) the P95 of `(99214+99215) / total_est_visits` is already at or above 99% — i.e. the norm IS billing the top two tiers. The rule cannot discriminate in those specialties and explicitly reports that, rather than silently false-negative or false-positive at scale. This is why UPCODING showed 0 triggers in the top-100 validation run despite being technically "evaluable."

---

## 8. Brief generation (Stage 4 — `generate_brief.py`)

### 8.1 Model

Default: **`claude-sonnet-4-6`**. Sonnet is the right cost/quality tradeoff for a 400–700-word structured narrative task with a heavy reference-material system prompt. Override with `--model` if needed; the orchestrator never assumes Sonnet.

### 8.2 System prompt

A single ~2,600-token system prompt does the heavy lifting. It contains:

1. The agent's role + risk classification rubric (LOW / MEDIUM / HIGH / CRITICAL).
2. The exact 7-section Markdown schema with literal headings — the parser regex-matches these.
3. **Dataset context** — what CMS public data does and doesn't contain, including the structural gaps that make some rules NOT EVALUABLE.
4. **HCPCS bucket reference** — what each of the 6 buckets means, with special attention to bucket 5 alphabetic codes (J-codes billed per drug unit, A-codes for supplies, etc.) so Claude doesn't conflate unit-of-measure billing with upcoding.
5. **Specialty norm heuristics** — primary care vs. surgical subspecialties vs. radiology/path/anesthesia vs. mass immunizers vs. DMEPOS. These prevent over-firing on legitimate specialty patterns.
6. **Rule-by-rule guidance** — for each evaluable rule, when the trigger is meaningful vs. likely benign.
7. **Weighting rubric** — qualitative not additive. One flag with benign explanation rarely exceeds MEDIUM; three flags with a coherent fraud narrative justify CRITICAL.
8. **Output discipline** — return only the brief, no preamble, no JSON wrapper.

The full text is in [`anomaly/generate_brief.py`](anomaly/generate_brief.py:25). It's verbatim-stable across briefs and is sent with `cache_control: ephemeral`, which means a 10-brief batch pays the prefix cost once and reads it back at ~90% off on calls 2–10. This is the single biggest cost lever in the agent.

### 8.3 User prompt

`format_user_prompt(ctx, rules)` assembles the per-(NPI, year) payload. Sections, in order:

1. Provider profile (NPI, specialty, state, year, years-active, risk score vs. specialty median).
2. Current-year metrics with their percentile rank in `[P<rank>]` brackets.
3. National specialty benchmarks (mean + P5/25/50/75/95) for every comparable metric.
4. State-level specialty benchmarks + peer-group size (so Claude knows when small-n state benchmarks are unreliable).
5. Year-by-year historical trajectory (services / srvcs-per-bene / yoy_vol per year).
6. Procedure mix: top 10 HCPCS codes with descriptions + bucket distribution.
7. E&M distribution + specialty benchmark (only when the sidecar is loaded and the provider has E&M volume).
8. OIG LEIE match details (only when matched — including reinstatement and waiver dates).
9. Rule-check results — every rule listed with TRIGGERED / NOT TRIGGERED / NOT EVALUABLE, severity, and one-line evidence.
10. Explicit `data_available=False` items pulled directly from the context — what was NOT checked.

### 8.4 Output parsing

Claude returns Markdown in the exact 7-section schema. `parse_brief_markdown()` regex-extracts:

- `**Risk Classification: <X>**` → `risk_classification`
- `**Composite Risk Score: <0-100>/100**` → `risk_score`
- `## <Section>` blocks via the heading-pattern regex

The full raw Markdown is preserved in `evidence_summary["full_markdown"]` and written alongside the parsed JSON; the web UI renders the Markdown directly so any prose Claude adds is visible even if a heading mismatches.

### 8.5 Reliability + cost

- **Retry with backoff** on `429` (rate limit) and `529` (overloaded) — 5 attempts at 5s / 10s / 20s / 40s / 80s. Other errors propagate immediately. This is necessary because a 100-brief batch can otherwise lose 1–2 briefs to transient capacity issues.
- **API key load order:** existing `ANTHROPIC_API_KEY` env var wins; otherwise the helper reads `--env-path` with `override=True` (necessary because an empty shell var would block default `load_dotenv` behavior).
- **Cost on the top-100 2023 validation run (Sonnet 4.6 with caching):** **$3.82 total**, ~$0.04 per brief. Cache-read tokens dominate after the first call.

---

## 9. Orchestrator (`agent.py`)

```bash
# Dry-run (no API spend — writes formatted prompts to <NPI>_<year>_prompt.md):
python anomaly/agent.py --top-n 10

# Live run against Sonnet 4.6:
python anomaly/agent.py --top-n 100 --year 2023 --live \
    --env-path "C:/Users/rajas/Documents/ADS/coverdrive_pred_11/.env"

# Targeted run (override the ranking):
python anomaly/agent.py --targets 1710906219:2018,1033474374:2018 --live ...
```

Ranking logic (`rank_flags`):

1. Read `flags.parquet`.
2. Group by `(NPI, year)`, aggregate `composite_severity = sum(severity)` and `n_flags`.
3. Sort by `(composite_severity DESC, n_flags DESC)`, take top N.

`--year` is the recommended filter for a real run — analysts care about recent activity; older flags are historical context, not the priority. `--targets` lets an analyst supply an explicit list (e.g., when following up on a specific case).

After ranking, the orchestrator constructs one `ContextRetriever` (amortizing parquet loads across the batch), then loops:

```
get_context(npi, year)  →  evaluate_all(ctx)  →  generate_brief(...)  →  write .md + .json
```

A `summary.json` is written at the end of the batch with token usage, estimated cost, and the parsed risk classifications.

---

## 10. Validation run (2023, top-100)

Run on `claude-sonnet-4-6` with prompt caching, 2026-04. The reference doc for the top-100 batch lives under `local_pipeline/anomaly/briefs_2023_validation/`.

| Dimension | Result |
|---|---|
| Total cost | **$3.82** |
| Severity distribution | 39 CRITICAL / 59 HIGH / 2 MEDIUM |
| Rule trigger counts | VOLUME_SPIKE 66, HIGH_INTENSITY 59, PROCEDURE_CONCENTRATION 6, CHARGE_INFLATION 6, OUT_OF_SPECIALTY 3, UPCODING 0, LEIE_EXCLUDED 0 |
| Top specialties | Nurse Practitioner (38), Gastroenterology (8), Mass Immunizer + Emergency Medicine (7 each) |

Two observations worth carrying into future runs:

1. **Composite flag set ∩ LEIE = ∅.** The statistical screen and the exclusion list pick out different populations. Composite flags catch high-volume operators billing in the open; LEIE catches NPIs already adjudicated as fraudulent. The agent should keep evaluating both — they're complementary, not competing.

2. **UPCODING = 0** is mostly the saturation effect described in §7, not an absence of upcoding. Composite-flagged providers are dominated by NPs and Mass Immunizers (high volume, narrow scope) where 99214/99215 saturation makes the rule indeterminate. To meaningfully target upcoding you'd run the agent on a *different* prefilter (specialties with non-saturated E&M distributions, e.g. dermatology or family practice).

---

## 11. Operational concerns

### 11.1 Dependencies

Project base deps plus:

```
anthropic >= 0.97
python-dotenv  (used in generate_brief._load_api_key)
```

The Claude SDK is the only net-new heavy dep. `scikit-learn`'s `IsolationForest` is already in the base pipeline.

### 11.2 Privacy

- All briefs are generated against **real CMS public data**. NPIs are public (NPPES) but identify named individuals.
- The web UI (`/investigations`) has a runtime NPI redaction toggle (format `1033****74`), and the `sync-briefs.mjs` script accepts `--mask-npis` / `MASK_NPIS=1` for build-time masking. `NEXT_PUBLIC_REDACT_NPIS=1` locks the redaction on for a deploy.
- **Do not** publish briefs publicly without one of those masking paths engaged.

### 11.3 Running the full pipeline cold

```bash
# 1. Build profiles + benchmarks (deterministic, no LLM)
python anomaly/compute_npi_profiles.py            # ~3 min
python anomaly/compute_benchmarks.py              # ~30 s

# 2. Build sidecars (also deterministic; OUT_OF_SPECIALTY, UPCODING, REVENUE_DEVIATION, LEIE unlock here)
python anomaly/rules/specialty_scopes.py
python anomaly/rules/em_distribution.py
python anomaly/rules/revenue_residuals.py        # ~5-10 min full silver; needs api/models/artifacts/
python anomaly/external/leie_loader.py --insecure   # OIG TLS chain fails Python defaults

# 3. Run statistical detection
python anomaly/detect_outliers.py                  # ~3 min

# 4. Generate briefs (LLM, costs money)
python anomaly/agent.py --top-n 100 --year 2023 --live \
    --env-path /path/to/.env
```

The `--insecure` flag on `leie_loader.py` is intentional: OIG's TLS chain doesn't validate against Python's default CA bundle locally. The loader sets a single-shot `verify=False` for that one fetch — don't generalize it.

### 11.4 Extending the rule set

A new rule is roughly four touchpoints:

1. **Sidecar table(s) if needed** — write a one-shot scan of silver that produces a parquet keyed by `(NPI, year)` or `(specialty, year)`.
2. **`ContextRetriever`** — load the sidecar in `__init__`, surface the new field on the `ProviderContext.metrics` dict (or `data_available` flag), guard with `os.path.exists(...)` so the retriever stays robust when the sidecar is missing.
3. **`check_rules.py`** — add the rule evaluator, return `available=False` with a clear reason if the sidecar isn't loaded. Append to `RULE_CHECKS`.
4. **`SYSTEM_PROMPT` in `generate_brief.py`** — add the new rule to the "Rule-by-rule guidance" section so Claude knows how to interpret it (the structured user-prompt section will already list it).

The optional `format_user_prompt` block at the end of §8.3 is a good template: only emit the section when the data is actually present (`if ctx.metrics.get("...") is not None: ...`).

---

## 12. Known limits

| Limit | Root cause | Path to fix |
|---|---|---|
| 3 rules permanently NOT EVALUABLE | CMS Provider & Service is annual-aggregate; no claim-level dates, no diagnosis codes, no beneficiary linkage | Paid LDS/RIF subscription |
| UPCODING fires in narrow window | ~80% of specialties have P95 already saturated at high-tier E&M | Use YoY drift within-NPI as a complementary signal, or restrict to non-saturated specialties |
| Small-state / small-specialty benchmarks unreliable | Group-size variance | Z-score already falls back to (specialty, year) at n < 30; Isolation Forest skips specialties with < 200 providers |
| 2021 noise | COVID YoY reset | `detect_temporal` excludes 2021 by default; the prompt also tells Claude 2020–2021 telehealth waivers can create benign spikes |
| Rural-solo false positives | Tiny peer groups → unstable percentiles | Acknowledged in the prompt; brief reasoning is expected to weigh this rather than the threshold being changed |

---

## 13. Pointers

- Spec / design: [`PROVIDER_ANOMALY_AGENT_SPEC.md`](PROVIDER_ANOMALY_AGENT_SPEC.md)
- Orchestrator entry point: [`anomaly/agent.py`](anomaly/agent.py)
- The system prompt (the single most important file for brief quality): [`anomaly/generate_brief.py:25`](anomaly/generate_brief.py:25)
- Rule evaluators: [`anomaly/check_rules.py`](anomaly/check_rules.py)
- Context assembly: [`anomaly/retrieve_context.py`](anomaly/retrieve_context.py)
- Web UI route: `/investigations` and `/investigations/[id]` in [`web/src/app`](web/src/app)
- Sync script (briefs → web public dir): [`web/scripts/sync-briefs.mjs`](web/scripts/sync-briefs.mjs)
