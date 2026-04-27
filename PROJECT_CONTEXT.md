---
project: Medicare Provider Utilization & Cost Analysis
codename: AllowanceMap
owner: Raj Vedire (Cobbles & Currents Studios)
status: Production — V2 deployed, Phase 9 shipped
last_updated: 2026-04-27
tags: [medicare, ml, fraud-detection, time-series, fastapi, nextjs, railway, vercel]
related: [[CoverDrive Pred 11]], [[DataSkrive Cohort]]
---

# Medicare Provider Utilization & Cost Analysis — Project Context Pack

> Single-file project context for use in claude.ai chat sessions and as an Obsidian project hub note.
> Source files: `CLAUDE.md`, `~/.claude/.../memory/MEMORY.md` and member files, `PROGRESS.md`, `V2_MODEL_SPEC.md`, `PROVIDER_ANOMALY_AGENT_SPEC.md`. Re-derive from those if drift suspected.

---

## TL;DR

End-to-end Medicare cost-analysis platform built on CMS Physician & Practitioners data 2013-2023 (~126.8M rows). Three production tracks running:

1. **Stage 1 — Allowed-amount prediction.** LightGBM V2 no-charge, R² 0.943, MAE ~$7. Live on Railway.
2. **Stage 2 — Patient OOP quantile prediction.** CatBoost monotonic P10/P50/P90 on synthetic-MCBS Track B. Live on Railway.
3. **Forecast — 2024-2026 specialty rates.** LightGBM Stacker V2_12, R² 0.8852. Signal ceiling at annual resolution confirmed via Multivariate TFT (R² 0.8691).

Plus **Phase 9 Provider Anomaly Investigation Agent** — 10-rule fraud-detection layer with Claude Sonnet 4.6 brief generation, surfaced via `/investigations` UI.

Frontend: Next.js + MUI on Vercel. Backend: FastAPI on Railway. Reference data: Supabase (proxied server-side).

Today's status: **complete and a bit more than spec'd.** Only outstanding work is medical-necessity Dx-linkage (Rule #3), blocked on paid LDS/RIF data subscription.

---

## 1. Project goals & scope

**Original goal:** Predict average Medicare allowed amount per service (`Avg_Mdcr_Alowd_Amt`) by HCPCS code × provider specialty × state, then extend to patient out-of-pocket cost (Stage 2) using MCBS data.

**Scope expansion (over the build):**
- Forecast track to project specialty-level rates 3 years forward (Phase 8).
- Provider anomaly / fraud investigation agent on top of silver layer (Phase 9).
- Live API + frontend so end users get real-time inference, not pre-computed lookups.
- Knowledge-base HTML reports (`docs/knowledge/`) and design handoff assets for product polish.

**Not in scope:** Per-claim or per-beneficiary analytics (would require LDS/RIF — paid data). Three rules in the anomaly agent are blocked on this.

---

## 2. Architecture overview

**Medallion data pipeline + dual execution mode (Databricks production / local development).**

```
CMS API (by Provider & Service) ─┐
                                 ├─→ Bronze ─→ Silver ─→ Gold ─→ Models
CMS API (by Provider, risk)   ───┘                ├─→ LSTM sequences
                                                  └─→ NPI profiles → Anomaly detection → Briefs
CMS MCBS PUF ─→ MCBS Bronze ─→ MCBS Silver ─→ Crosswalk (Track A: real PUF national)
                                                       ↓
generate_synthetic_mcbs.py   ─→  Crosswalk (Track B: synthetic per-service regional)
                                                       ↓
                                              OOP quantile training
```

**Serving layer:**
```
Browser → Vercel (Next.js)  ──HTTP──→  Railway (FastAPI)  ──proxy──→  Supabase (reference data)
                                              │
                                              ├─ LightGBM no-charge    (Stage 1)
                                              ├─ CatBoost monotonic    (Stage 2 OOP P10/P50/P90)
                                              └─ Stacker forecast      (parquet served)
```

---

## 3. Data sources

| Dataset | Source | Years | Rows | Use |
|---|---|---|---|---|
| Medicare Physician & Practitioners — by Provider & Service | CMS data.cms.gov | 2013-2023 | 126.8M | Bronze/Silver/Gold; Stage 1 modeling; anomaly detection |
| Medicare Physician & Practitioners — by Provider | CMS data.cms.gov | 2013-2023 | ~1.7M NPIs/yr | `Bene_Avg_Risk_Scre` join; NPI profiles |
| MCBS Public Use File (Survey + Cost) | CMS PUF | 2017-2021 | ~12K respondents/yr | Track A — national OOP crosswalks |
| Synthetic LDS-style per-service OOP | `generate_synthetic_mcbs.py` | 2013-2023 | ~10% sample | Track B — regional per-service OOP for Stage 2 training |
| OIG LEIE (List of Excluded Individuals/Entities) | OIG.HHS.gov | rolling | ~83K rows, 8.4K NPIs | Phase 9 LEIE_EXCLUDED rule |

**Notes:**
- Medicare data is public (Provider-level aggregates, no claims). NPIs are public via NPPES.
- LEIE OIG TLS chain fails Python's default CA validation locally — use `--insecure` flag on the loader.
- Data privacy on briefs: real CMS data, real NPIs. Use redaction toggle for any public deploy.

See [`DATA_SOURCES.md`](DATA_SOURCES.md) for the full inventory.

---

## 4. Pipeline (Bronze → Silver → Gold)

**Output convention:** `local_pipeline/` (gitignored) for local dev; S3/Delta on Databricks.

| Stage | Code | Output | Notes |
|---|---|---|---|
| Bronze | `notebooks/01_bronze_ingest_local.py` | `bronze/bronze.parquet` | Raw ingest with year injected by `partition_medicare_data.py` |
| Silver | `notebooks/02_silver_clean_local.py` | `silver/{STATE}.parquet` | Typed, IQR clipping on allowed amount, partitioned by state |
| Gold | `notebooks/03_gold_features_local.py` | `gold/{STATE}.parquet`, `gold/label_encoders.json` | Encoded features + risk score join |
| LSTM sequences | `notebooks/05_lstm_sequences_local.py` | `lstm/sequences.parquet` | Time-series sequences for Phase 3 LSTM |
| MCBS Bronze | `notebooks/06_mcbs_bronze_local.py` | `mcbs_bronze/{survey,cost}_{YEAR}.parquet` | — |
| MCBS Silver | `notebooks/07_mcbs_silver_local.py` | `mcbs_silver/{YEAR}.parquet` | Survey + Cost joined |
| MCBS Crosswalk | `notebooks/08_mcbs_crosswalk_local.py` | `mcbs_crosswalk/crosswalk.parquet` | `--mode puf` or `--mode synthetic` |
| Synthetic OOP | `generate_synthetic_mcbs.py` | `mcbs_synthetic/synthetic_oop.parquet` | Track B — per-service OOP with region |

### Feature set (10) — used by Stage 1 tree models
- `Rndrng_Prvdr_Type_idx` — encoded specialty
- `Rndrng_Prvdr_State_Abrvtn_idx` — encoded state
- `HCPCS_Cd_idx` — encoded HCPCS (~6K unique)
- `hcpcs_bucket` — 0=Anesthesia, 1=Surgery, 2=Radiology, 3=Lab, 4=Medicine/E&M, 5=HCPCS Level II
- `place_of_srvc_flag` — 1=facility, 0=office
- `Bene_Avg_Risk_Scre` — NPI HCC risk score
- `log_srvcs`, `log_benes` — log1p volume
- `Avg_Sbmtd_Chrg` — submitted charge (excluded in no-charge variant)
- `srvcs_per_bene` — services per beneficiary

**Removed (data leakage):** `Avg_Mdcr_Pymt_Amt`, `Avg_Mdcr_Stdzd_Amt` (both derived from allowed amount).
**Year column:** preserved as int16 metadata, not a feature for tree/GLM models, used for LSTM sequence grouping and temporal splits.

### Census regions (batch-mode training)
```
NORTHEAST:   CT ME MA NH RI VT NJ NY PA
SOUTH:       DE FL GA MD NC SC VA DC WV AL KY MS TN AR LA OK TX
MIDWEST:     IL IN MI OH WI IA KS MN MO NE ND SD
WEST:        AZ CO ID MT NV NM UT WY AK CA HI OR WA
TERRITORIES: AA AE AP AS FM GU MP PR PW VI
```

---

## 5. Stage 1 — Allowed-amount models (PRODUCTION: LightGBM V2 no-charge)

V2 trained on Colab Pro (A100 / T4) on 2026-04-11. Full spec: [`V2_MODEL_SPEC.md`](V2_MODEL_SPEC.md).

| Model | R² | MAE | Notes |
|---|---|---|---|
| LightGBM V2 (full) | 0.9575 | $6.73 | Charge-aware |
| **LightGBM V2 no-charge** | **0.943** | **~$7** | **PRODUCTION** — Railway API. Auto-detected from feature names. |
| Ensemble V2 | 0.9580 | — | Only +0.0004 over LightGBM. Not worth deploying. |
| XGBoost V2 | 0.9452 | $7.73 | — |
| CatBoost V2 | 0.9070 | $10.88 | — |
| GLM (SGD baseline) | ~0.75 | — | Phase 1 baseline |
| Random Forest | ~0.92 | — | Phase 1 (warm_start by region or full RandomizedSearchCV) |

**Why no-charge wins production:** API users don't reliably know `Avg_Sbmtd_Chrg`. Dropping it costs only 0.01-0.02 R². Robustness > marginal accuracy.

**Training modes:** XGB / RF / CatBoost / LightGBM all support `--mode batch|full`. Batch is incremental by census region (memory-efficient, default). Full is single-pass with early stopping (more RAM).

**Artifacts:** `api/models/artifacts/lgbm_v2_no_charge.txt` (1000 trees, ~26 MB).

---

## 6. Stage 2 — Patient OOP quantile (PRODUCTION: CatBoost monotonic)

Synthetic-MCBS Track B used for training (real MCBS PUF doesn't go per-service).

| Model | R² | Status |
|---|---|---|
| XGB Quantile V1 | 0.400 | Best raw R² |
| **CatBoost Mono V2 (P10/P50/P90)** | **0.173** | **PRODUCTION** — monotonicity valued for product safety |
| CatBoost ZI V2 | -0.054 | Gate+regression compounded errors |

**Why CatBoost Mono wins production despite lower R²:** product safety. Monotone constraints (e.g., higher allowed → higher OOP) prevent the API from returning paradoxical answers under user input variation.

**Constraint quirk:** CatBoost GPU does **not** support `monotone_constraints` — must train on CPU.

**Calibration sidecar (`oop_calibration.json`):** Raw CatBoost quantile output is asymmetrically miscalibrated — P10 marginal coverage 14.5% (target 10%, slightly conservative), P90 marginal coverage 67.5% (target 90%, materially under-covering). Asymmetric CQR per [`modeling/calibrate_oop.py`](modeling/calibrate_oop.py) computes `q_lo = $0.0004` (essentially zero — leaves P10 alone) and `q_hi = $14.47` (added to P90). Held-out test coverage post-calibration: 80.1% interval, 90.0% P90 marginal. Symmetric q_hat = $3.6884 reproduces V2_04's MLflow log exactly. Loader auto-applies on startup; no calibration sidecar = bands served raw with WARNING logged.

**Artifacts:** `api/models/artifacts/oop_mono_{p10,p50,p90}.cbm` + `oop_calibration.json`. Categorical indices: `OOP_CAT_IDX = [2,3,4,5]`.

---

## 7. Forecast track — Phase 8 (CLOSED — PRODUCTION: LightGBM Stacker V2_12)

Three-track investigation through V2_09 → V2_13. Production: `stacker_forecast_2024_2026.parquet` (LightGBM stacker over LSTM + Chronos + persistence).

### Final leaderboard (2022-2023 fair temporal holdout, N=32,481)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **LGB Stacker V2_12** | **8.74** | **17.69** | **0.8852** ← PRODUCTION |
| LSTM V1 (autoregressive, fair) | 9.82 | 18.91 | 0.8689 |
| Multivariate TFT V2_13 | 9.23 | 18.79 | 0.8691 |
| Chronos Bolt cpi_cf_deflated | 9.39 | 19.71 | 0.8576 |
| LSTM V1 (teacher-forced, INFLATED) | 8.84 | 36.42 | 0.8860 |

### Key findings
1. **Signal ceiling ≈ 0.885 at annual resolution.** Three independent approaches (LSTM AR, Chronos, TFT multivariate) cluster R² 0.857-0.869. Only the ensemble stacker breaks above (+0.0163 via prediction diversity).
2. **Multivariate covariates didn't help.** TFT V2_13 with 5 observed covariates (log_srvcs, log_benes, Avg_Sbmtd_Chrg, Bene_Avg_Risk_Scre, srvcs_per_bene) tied with univariate LSTM (+0.0002 R²). At annual resolution these features move in near-lockstep with the target.
3. **Stacker feature importance:** lstm_pred 70.5%, last_history_value 16.0%, chronos_pred 3.6%, history_mean 2.7%, ptype/state 3.1%, history_cv/trend 2.8%, n_history_years 0.8%, hcpcs_bucket/forecast_year 0.6%, cpi_factor/cf_factor ≈0%. Stacker is mostly LSTM + persistence anchor + history conditioning.
4. **TFT forecast shape is more coherent** through 2024-2026 (joint $70 → $68 → $68 vs stacker mean-reversion $72 → $62 → $62) but no ground truth on the actual horizon to validate.

### Only realistic path above 0.89 (BACKLOG, not pursued)
**Quarterly CMS data ingestion.** 11 annual points → 44 quarterly per group. Would unlock CNN/TCN/PatchTST architectures, seasonal decomposition, proper foundation-model fine-tuning. Estimated 2-4 weeks of data engineering. Target R² 0.91-0.93.

### Known bug (cosmetic)
`modeling/train_lstm_local.py` `evaluate()` (~line 340) uses 1-step teacher-forced prediction, not autoregressive rollout. V1 reported R² ≈0.886 inflated by ~0.017. Production forecast doesn't use this code path (V2 stacker takes over) — but fix it if anyone re-runs the local LSTM.

---

## 8. Phase 9 — Provider Anomaly Investigation Agent (SHIPPED)

Fraud-detection layer on top of silver. Spec designed 2026-04-08, built 2026-04-23 → 2026-04-24 (Phases A-E). Phases A-D shipped via PRs #2-4 (tag `phase-9d-complete`); Phase 9E shipped direct to main 2026-04-24 (commit 423a26f).

### Pipeline (`anomaly/`)
- `compute_npi_profiles.py` — silver → 11.52M NPI-year profiles × 22 metrics (volume, intensity, charge ratios, Herfindahl, bucket distribution, YoY changes, risk score)
- `compute_benchmarks.py` — specialty / state / national benchmark tables (mean + P5/P25/P50/P75/P95)
- `detect_outliers.py` — z-score (log1p) + Isolation Forest + temporal-rule detection → `flags.parquet`. Composite ≥2-method hit: 0.77% of NPI-years.
- `retrieve_context.py` — `ContextRetriever` builds evidence package per (NPI, year); loads E&M and LEIE sidecars
- `check_rules.py` — 10 rules
- `generate_brief.py` — Claude Sonnet 4.6 with 2,584-token cache-engaged system prompt; 429/529 retry-with-backoff
- `agent.py` — orchestrator with `--year` filter for targeted runs
- `rules/specialty_scopes.py` — `specialty_scopes.parquet` (86,924 rows, 130 specialties, 27% in-scope)
- `rules/em_distribution.py` — `em_distributions.parquet` (5.94M rows) + `em_specialty_benchmarks.parquet`
- `external/leie_loader.py` — OIG LEIE → `leie_exclusions.parquet` (8,375 NPIs total, 2,535 overlap with CMS providers)

### 10 rules — status
**Evaluable (7):**
1. **LEIE_EXCLUDED** — CRITICAL override under 42 USC 1320a-7. Position 1 in RULE_CHECKS.
2. VOLUME_SPIKE
3. HIGH_INTENSITY
4. PROCEDURE_CONCENTRATION
5. CHARGE_INFLATION
6. OUT_OF_SPECIALTY (uses empirical population scopes — see interpretive note below)
7. UPCODING (high-tier E&M share > specialty P95 on ≥50 visits; reports "saturated P95" for the 68/84 specialties where the rule can't discriminate)

**NOT EVALUABLE (3):** IMPOSSIBLE_DAY, UNBUNDLING, BENEFICIARY_SHARING — all require per-claim/date-of-service/beneficiary linkage absent from CMS public PUF.

### Validation run (n=100 top composite-flagged 2023, Sonnet 4.6 cached)
- **Severity:** 39 CRITICAL, 59 HIGH, 2 MEDIUM
- **Cost:** $3.82 total (cache_read=282,942)
- **Triggers:** VOLUME_SPIKE 66, HIGH_INTENSITY 59, PROCEDURE_CONCENTRATION 6, CHARGE_INFLATION 6, OUT_OF_SPECIALTY 3, UPCODING 0, LEIE_EXCLUDED 0
- **Top specialties:** Nurse Practitioner (38), Gastroenterology (8), Mass Immunizer + Emergency Medicine (7 each)
- **Why UPCODING/LEIE = 0:** top-100 dominated by NPs and Mass Immunizers; E&M saturated at P95=100% across 68/84 specialties; LEIE-overlap with composite flags = 0

### Interpretive note (worth remembering)
**"Specialty scope" is empirical, not regulatory.** The optometrist-bills-cataract-surgery (66984) case is NOT flagged by OUT_OF_SPECIALTY because 28% of optometrists in the data bill that code (co-management). That NPI still surfaces CRITICAL via HIGH_INTENSITY + PROCEDURE_CONCENTRATION. Don't promise scope-of-practice fraud detection — promise outlier vs population.

### Cost-tracking
- Initial 10-brief run (no cache): $0.36
- Re-gen with padded prompt + cache: $0.42 total ($0.04/brief amortized after first)
- Validation 100-brief: $3.82

### Data privacy
All briefs are real CMS data + real NPIs. Use the runtime redaction toggle in `/investigations` (or env `NEXT_PUBLIC_REDACT_NPIS=1`) for any demo, screenshot, or public deploy. Format: `1033****74`.

### API key location
`C:/Users/rajas/Documents/ADS/coverdrive_pred_11/.env` (cross-project). Helper loads with `override=True` because an empty shell var would otherwise block dotenv.

---

## 9. Backend — FastAPI on Railway (LIVE)

URL: `medicare-provider-utilization-cost-analysis-production.up.railway.app`

Replaces a pre-computed Supabase 33K-row group-average table. Real-time inference lets users plug in custom inputs and get actual model predictions.

### Layout
```
api/
├── Dockerfile, main.py, config.py, requirements.txt
├── models/
│   ├── loader.py
│   └── artifacts/                # lgbm_v2_no_charge.txt, oop_mono_p{10,50,90}.cbm (~26MB total)
├── routers/
│   ├── health.py
│   ├── predict.py                # Stage 1 + Stage 2
│   ├── forecast.py               # 2024-2026 specialty forecasts
│   └── reference.py              # Supabase-proxied lookups
├── services/
│   ├── prediction.py
│   ├── supabase.py
│   └── specialty_canonicalization.py   # Collapses CMS name-splits (Cardiology, Colorectal, Oral Surgery)
└── schemas/
```

### Key behaviors
- Stage 1 model auto-detects charge / no-charge variant from feature names.
- Stage 2 OOP categorical indices: `OOP_CAT_IDX = [2, 3, 4, 5]`.
- Reference data (forecasts, labels, metrics) proxied from Supabase server-side (anon key currently — should switch to service-role key for production hardening).
- Specialty canonicalization shipped in commit cc248ae handles CMS upstream splitting Cardiology/Colorectal/Oral Surgery into multiple display names.

---

## 10. Frontend — Next.js + MUI on Vercel (LIVE)

`web/` directory. Reads `NEXT_PUBLIC_API_URL` (Vercel env + `web/.env.local`) → Railway.

### Routes
| Path | Purpose |
|---|---|
| `/` | Estimator (Stage 1 + Stage 2 inference inputs) |
| `/forecast` | Specialty-level 2024-2026 forecast viewer |
| `/investigations` | Phase 9 brief list with severity filter, NPI redaction toggle |
| `/investigations/[id]` | Brief detail + Approve/Escalate/Dismiss + analyst notes (localStorage) |
| `/demo` | Product demo page (`AllowanceMap_ProductDemo.mp4`) — UNTRACKED, not committed yet |
| `/about` | Project info |

### Conventions
- `web/scripts/sync-briefs.mjs` copies Phase 9 briefs to `web/public/data/investigations/` and parses per-rule summaries. Supports `--mask-npis` flag and `MASK_NPIS=1` env for public deploys.
- Runtime NPI redaction: localStorage with cross-tab `CustomEvent` sync. Format `1033****74`. `NEXT_PUBLIC_REDACT_NPIS=1` locks it on.
- `web/CLAUDE.md` warns: this is **not the Next.js you know** — read `node_modules/next/dist/docs/` before writing app code.

---

## 11. Auxiliary deliverables (ungated)

- `docs/knowledge/` — 9-part HTML knowledge base (project overview through anomaly agent). Untracked.
- `design_handoff/` — animations, scenes, swipedeck, onepager, `AllowanceMap_Animation.mp4`. Untracked.
- `report_build/build_report.js` — report builder. Untracked.
- `AllowanceMap_Project_Report.docx`, `medicare_oop_project_spec.html`, `medicare_v2_model_analysis.html` — deliverables. Untracked.
- `medicare_knowledge_report.html` — committed.

---

## 12. Lessons learned & feedback (worth carrying forward)

### Compute estimates
**Training time on 100M+ rows is 3-5x worse than naive estimates suggest.** Empirical multipliers from V2:

| Workload | Estimate | Actual |
|---|---|---|
| XGBoost 1000 rounds GPU | ~8 min | ~8 min ✓ |
| CatBoost 3000 iters GPU | ~30 min | ~105 min (3.5x) |
| LightGBM 1000 rounds CPU | ~15 min | ~50 min (3.3x) |
| Ensemble 5-fold (15 runs) | 6-8 hrs / 12-16 CU | 13.3 hrs / 149 CU (10x CU) |

**Rule of thumb:** for 100M+ row datasets, multiply initial estimates by 3-4x. Never estimate CatBoost GPU at <30 min per 1000 iters on 100M rows. Per-fold × n_folds is the real ensemble baseline.

### Modeling
- **CatBoost GPU does not support `monotone_constraints`** — must use CPU.
- **Monotonicity hurts when the assumed relationship doesn't hold** in the data (Stage 2 OOP V2: R² dropped from 0.40 to 0.17).
- **TFT needs long sequences** (100+ steps) to outperform LSTM. With 11 annual points, attention has nothing to attend to.
- **Teacher-forcing in `evaluate()` inflates reported metrics** by ~0.017 R² and 17 RMSE points. Always evaluate autoregressively for forecast comparison.

### Engineering / Colab
- **Drive FUSE timeouts on Colab.** Always copy artifacts to `/content/` local SSD before training.
- **PyTorch 2.6 `weights_only=True` breaks pytorch-forecasting checkpoint loading** — pin earlier or override.
- **Compatible PyTorch-Forecasting pair:** `pytorch-forecasting==1.1.1` + `lightning==2.2.5`.

### Anomaly agent
- **Specialty scope is empirical, not regulatory.** Don't promise scope-of-practice detection; promise outlier vs specialty population.
- **System prompt padded to 2,584 tokens** to engage Sonnet 4.6 prompt cache. Cache_read=2,578 on every post-first call. Material cost driver.
- **OIG LEIE TLS chain** fails Python's default CA validation locally — `--insecure` flag required.

### General preference
- **Real production models > marginal R² wins.** Ensemble V2 was +0.0004 over LightGBM; not deployed. CatBoost Mono is lower R² than XGB Quantile but won production for monotonicity safety.

---

## 13. Backlog

| Item | Status | Effort |
|---|---|---|
| Rule #3 — medical-necessity Dx linkage (UPCODING/UNBUNDLING extension) | BLOCKED on paid LDS/RIF data | unknown |
| Quarterly data ingestion (forecast ceiling break) | BACKLOG, not pursued | 2-4 weeks |
| Fix `train_lstm_local.py` `evaluate()` teacher-forcing bug | KNOWN, cosmetic | 1 hr |
| Train 3-head quantile stacker (real P10/P50/P90 forecast bounds) | NICE-TO-HAVE | 20-line patch to V2_12 Cell 10 |
| Switch Supabase from anon → service-role key | HARDENING | 30 min + redeploy |
| Commit untracked deliverables (`/demo`, `docs/knowledge/`, `design_handoff/`) | PENDING USER DECISION | 5 min |

---

## 14. Quickstart commands

### Pipeline (one-time)
```bash
python pull_provider_data.py --output-dir data/
python partition_medicare_data.py --input-dir data --output-dir partitioned_data
python csv_to_parquet.py --dir partitioned_data
python notebooks/01_bronze_ingest_local.py
python notebooks/02_silver_clean_local.py
python notebooks/03_gold_features_local.py --provider-data-dir data/
python notebooks/05_lstm_sequences_local.py
python notebooks/04_eda_local.py
```

### Stage 1 training (batch mode default)
```bash
python modeling/train_lgbm_local.py --mode batch
python modeling/train_xgb_local.py --mode batch
python modeling/train_catboost_local.py --mode batch
python modeling/train_rf_local.py --mode batch
python modeling/train_glm_local.py
python modeling/compare_models_local.py
```

### Stage 2 OOP training
```bash
python pull_mcbs_data.py --type both
python notebooks/06_mcbs_bronze_local.py
python notebooks/07_mcbs_silver_local.py
python generate_synthetic_mcbs.py --sample 0.1
python notebooks/08_mcbs_crosswalk_local.py --mode synthetic
python modeling/train_oop_local.py --sample 0.3 --rounds 500
```

### Phase 9 anomaly pipeline
```bash
python anomaly/compute_npi_profiles.py
python anomaly/compute_benchmarks.py
python anomaly/rules/specialty_scopes.py
python anomaly/rules/em_distribution.py
python anomaly/external/leie_loader.py --insecure
python anomaly/detect_outliers.py

# Dry-run (no API spend)
python anomaly/agent.py --top-n 10

# Live run with cached system prompt
python anomaly/agent.py --top-n 100 --year 2023 --live \
    --env-path "C:/Users/rajas/Documents/ADS/coverdrive_pred_11/.env"
```

### Frontend / backend
```bash
# API (Railway deploys from main automatically)
cd api && uvicorn main:app --reload

# Web (Vercel deploys from main automatically)
cd web && npm run dev
```

---

## 15. File map (top level)

```
medicare-provider-utilization-cost-analysis-/
├── CLAUDE.md                             # Source-of-truth project context (this doc's parent)
├── PROJECT_CONTEXT.md                    # ← you are here
├── PROGRESS.md                           # Phase-by-phase execution log
├── V2_MODEL_SPEC.md                      # V2 model design + Colab training spec
├── PROVIDER_ANOMALY_AGENT_SPEC.md        # Phase 9 design
├── DATA_SOURCES.md, MODELING.md, IMPROVEMENTS.md, FRONTEND_TODOS.md, README.md
│
├── pull_medicare_data.py / pull_provider_data.py / pull_mcbs_data.py
├── partition_medicare_data.py / csv_to_parquet.py / generate_synthetic_mcbs.py
│
├── notebooks/                            # 01-08 + LSTM sequences (local + Databricks variants)
├── modeling/                             # train_glm/rf/xgb/catboost/lgbm/lstm/oop + compare
├── anomaly/                              # Phase 9 — profiles, benchmarks, detection, brief gen
│   ├── rules/                            # specialty_scopes, em_distribution
│   └── external/                         # leie_loader
│
├── api/                                  # FastAPI on Railway
│   ├── Dockerfile, main.py, config.py
│   ├── models/{loader, artifacts/}
│   ├── routers/{health, predict, forecast, reference}.py
│   ├── services/{prediction, supabase, specialty_canonicalization}.py
│   └── schemas/
│
├── web/                                  # Next.js on Vercel
│   ├── src/app/{page.tsx, forecast, investigations, demo, about}
│   ├── src/lib/constants.ts
│   ├── public/data/investigations/       # synced briefs
│   └── scripts/sync-briefs.mjs
│
├── docs/knowledge/                       # 9-part HTML knowledge base
├── design_handoff/                       # animations + scenes + swipedeck + product demo mp4
├── report_build/                         # build_report.js
│
└── local_pipeline/                       # gitignored — bronze/silver/gold/lstm/mcbs/anomaly
```

---

## 16. Environment & dependencies

**Platform:** Windows 11 + WSL2 Ubuntu, NVIDIA 5070 Ti (16GB VRAM).
**Python core:** pandas, pyarrow, scikit-learn, xgboost, mlflow, matplotlib, seaborn, scipy, requests, torch, pyspark (Databricks only).
**Optional GPU:** cudf-cu12 (RAPIDS for gold features), cuml-cu12 (RAPIDS for RF full mode), torch CUDA (for LSTM).
**Phase 9 add-ons:** anthropic ≥ 0.97, python-dotenv.
**Colab pinned pair:** pytorch-forecasting==1.1.1 + lightning==2.2.5.

**Env vars (`.env`, gitignored):** `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `ANTHROPIC_API_KEY` (cross-project, in `coverdrive_pred_11/.env`).

**MLflow experiment:** local runs log to `{user_home}/medicare_models` (unified experiment).

---

## 17. Cross-project links (Obsidian)

- [[CoverDrive Pred 11]] — shares the `ANTHROPIC_API_KEY` location used by the anomaly agent (`coverdrive_pred_11/.env`).
- [[DataSkrive Cohort]] — separate cohort/segmentation work; no direct dependency.
- See `~/.claude/projects/.../memory/MEMORY.md` for the full live memory index.

---

## 18. References — source files in repo

| File | Use |
|---|---|
| `CLAUDE.md` | Project context for Claude Code sessions |
| `PROGRESS.md` | Phase-by-phase execution log + changelog |
| `V2_MODEL_SPEC.md` | V2 model design |
| `PROVIDER_ANOMALY_AGENT_SPEC.md` | Phase 9 design (designed 2026-04-08) |
| `MODELING.md` | Detailed modeling decisions |
| `DATA_SOURCES.md` | Full data inventory |
| `IMPROVEMENTS.md` | Future improvements brainstorm |
| `FRONTEND_TODOS.md` | Frontend backlog |
| `README.md` | Public-facing project intro |
| `medicare_knowledge_report.html` | Built knowledge report |
| `docs/knowledge/*.html` | 9-part HTML knowledge base |

---

*End of context pack. Re-derive from source files if drift suspected — this is a snapshot.*
