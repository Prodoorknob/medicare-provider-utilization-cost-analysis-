# CLAUDE.md

## Project Overview

Medicare Provider Utilization & Cost Analysis ("AllowanceMap") — end-to-end data pipeline, ML, forecasting, fraud-detection, API, and frontend for CMS Physician & Practitioners data (2013-2023).

- **Stage 1 (allowed amount):** LightGBM V2 no-charge — R² 0.943, MAE $6.73. Live on Railway.
- **Stage 2 (patient OOP):** CatBoost monotonic quantile (P10/P50/P90) on synthetic-MCBS Track B. Live on Railway.
- **Forecast (2024-2026):** LightGBM Stacker V2_12 — R² 0.8852 on temporal holdout. Signal ceiling at annual resolution confirmed via TFT V2_13.
- **Provider Anomaly Investigation Agent (Phase 9):** 10 rules (7 evaluable), Claude Sonnet 4.6 brief generation, `/investigations` UI on web app.
- **Frontend:** Next.js + MUI on Vercel; routes `/`, `/forecast`, `/investigations`, `/demo`, `/about`.
- **Backend:** FastAPI on Railway (real-time inference + Supabase reference proxy).

## Architecture

**Medallion pipeline** with two execution modes:

1. **Databricks (production):** Notebooks in `notebooks/` (01–04) use PySpark + Delta Lake on S3. MLflow experiments log to Databricks workspace.
2. **Local (development):** `*_local.py` variants in `notebooks/` and `modeling/` use pandas/pyarrow on `local_pipeline/` parquet files. These also log to Databricks MLflow via `DATABRICKS_HOST`/`DATABRICKS_TOKEN` env vars.

### Data flow

```
CMS API (by Provider & Service) → pull_medicare_data.py → partition_medicare_data.py (injects year) → csv_to_parquet.py
CMS API (by Provider)           → pull_provider_data.py (Bene_Avg_Risk_Scre)
  ↓
Bronze (raw ingest + year) → Silver (typed, cleaned, IQR on allowed amt) → Gold (features + encoding, per-state) → EDA + Modeling
                                                                             ↓
                                                                          LSTM sequences (05_lstm_sequences)

CMS MCBS PUF (Survey + Cost) → pull_mcbs_data.py → 06_mcbs_bronze → 07_mcbs_silver → 08_mcbs_crosswalk
                                                                                        ↓
                                              generate_synthetic_mcbs.py → 08_mcbs_crosswalk --mode synthetic
```

### Local pipeline outputs (`local_pipeline/`, gitignored)

- `bronze/bronze.parquet` — consolidated raw data
- `silver/{STATE}.parquet` — cleaned, partitioned by state (year column preserved)
- `gold/{STATE}.parquet` — model-ready features, one file per state
- `gold/label_encoders.json` — persisted LabelEncoder classes for consistent encoding
- `lstm/sequences.parquet` — LSTM-ready time-series sequences (Phase 3 input)
- `lstm/forecast_2024_2026.parquet` — LSTM forecast output with confidence bounds
- `lstm/plots/` — specialty trend and forecast visualizations
- `mcbs_bronze/survey_{YEAR}.parquet`, `cost_{YEAR}.parquet` — raw MCBS ingest
- `mcbs_silver/{YEAR}.parquet` — cleaned MCBS (survey + cost joined)
- `mcbs_crosswalk/crosswalk.parquet` — region x specialty bridge table
- `mcbs_synthetic/synthetic_oop.parquet` — synthetic per-service OOP with region (Track B)
- `mcbs_synthetic/synthetic_metadata.json` — provenance and replacement instructions
- `eda/` — plots and summaries

## Key Conventions

- **Target variable:** `Avg_Mdcr_Alowd_Amt` (Medicare allowed amount — Stage 1 target)
- **Feature set (10 features):**
  - `Rndrng_Prvdr_Type_idx` — encoded provider specialty
  - `Rndrng_Prvdr_State_Abrvtn_idx` — encoded state
  - `HCPCS_Cd_idx` — encoded raw HCPCS code (~6K unique)
  - `hcpcs_bucket` — coarse clinical category (0=Anesthesia, 1=Surgery, 2=Radiology, 3=Lab, 4=Medicine/E&M, 5=HCPCS Level II)
  - `place_of_srvc_flag` — binary (1=facility, 0=office)
  - `Bene_Avg_Risk_Scre` — NPI-level HCC risk score (from "by Provider" dataset)
  - `log_srvcs` — log1p(Tot_Srvcs)
  - `log_benes` — log1p(Tot_Benes)
  - `Avg_Sbmtd_Chrg` — submitted charge amount
  - `srvcs_per_bene` — services per beneficiary ratio
- **Temporal metadata:** `year` column (int16, 2013-2023) — not used as feature by tree/GLM models, used for LSTM sequence grouping
- **Removed (data leakage):** `Avg_Mdcr_Pymt_Amt`, `Avg_Mdcr_Stdzd_Amt` (derived from allowed amount)
- **Train/test split:** 80/20, `random_state=42` (consistent across all models)
- **Local scripts use `log1p` target transform** for skew correction
- **Six models:** GLM (SGD baseline), Random Forest (warm_start or RandomizedSearchCV), XGBoost (incremental or early stopping), CatBoost (native categoricals via ordered target statistics), LightGBM (GOSS + leaf-wise growth), LSTM (PyTorch, temporal split + MC Dropout forecasting)
- **LSTM specifics:** Temporal split (train ≤ 2021, val 2022-2023), autoregressive forecast 2024-2026, MC Dropout confidence bounds, static embeddings for group keys
- **MLflow experiment:** All local runs log to `{user_home}/medicare_models` (unified experiment)

### Training modes

XGBoost and RF support `--mode batch|full`:

| Mode | XGBoost | Random Forest | CatBoost | LightGBM |
|------|---------|---------------|----------|----------|
| `batch` (default) | Incremental by region via `xgb_model`, 125 rounds/region | warm_start by region, 125 trees/region, sklearn only | Incremental by region via `init_model`, 125 iters/region | Incremental by region via `init_model`, 125 rounds/region |
| `full` | Single DMatrix, early stopping, 500 rounds | RandomizedSearchCV, cuML/sklearn auto-detect | Single Pool, early stopping, 500 iters | Single Dataset, early stopping, 500 rounds, GOSS |

### Census regions (for batch training)

```
NORTHEAST:   CT, ME, MA, NH, RI, VT, NJ, NY, PA
SOUTH:       DE, FL, GA, MD, NC, SC, VA, DC, WV, AL, KY, MS, TN, AR, LA, OK, TX
MIDWEST:     IL, IN, MI, OH, WI, IA, KS, MN, MO, NE, ND, SD
WEST:        AZ, CO, ID, MT, NV, NM, UT, WY, AK, CA, HI, OR, WA
TERRITORIES: AA, AE, AP, AS, FM, GU, MP, PR, PW, VI
```

## File Layout

```
├── pull_medicare_data.py          # CMS API download (by Provider & Service)
├── pull_provider_data.py          # CMS download (by Provider — risk scores)
├── partition_medicare_data.py     # Split CSVs by STATE/PROVIDER_TYPE, inject year
├── csv_to_parquet.py              # Convert CSV partitions to Parquet
├── notebooks/
│   ├── 01_bronze_ingest.py / _local.py
│   ├── 02_silver_clean.py / _local.py
│   ├── 03_gold_features.py / _local.py   # Per-state gold + risk score join
│   ├── 04_eda.py / _local.py
│   └── 05_lstm_sequences_local.py        # LSTM sequence preparation
├── modeling/
│   ├── train_glm.py / _local.py
│   ├── train_rf.py  / _local.py          # --mode batch|full
│   ├── train_xgb.py / _local.py          # --mode batch|full
│   ├── train_catboost_local.py           # --mode batch|full, native categoricals
│   ├── train_lgbm_local.py              # --mode batch|full, GOSS + leaf-wise
│   ├── train_lstm_local.py               # PyTorch LSTM forecasting
│   ├── train_oop_local.py                # Stage 2 OOP quantile regression (P10/P50/P90)
│   └── compare_models.py / _local.py
├── pull_mcbs_data.py                     # MCBS PUF download (Survey + Cost)
├── notebooks/
│   ├── 06_mcbs_bronze_local.py           # MCBS Bronze ingest
│   ├── 07_mcbs_silver_local.py           # MCBS Silver cleaning + join
│   └── 08_mcbs_crosswalk_local.py        # Region × specialty crosswalk (--mode puf|synthetic)
├── generate_synthetic_mcbs.py            # Synthetic per-service OOP (Track B)
├── anomaly/                              # Phase 9 fraud-detection pipeline
│   ├── compute_npi_profiles.py
│   ├── compute_benchmarks.py
│   ├── detect_outliers.py
│   ├── retrieve_context.py
│   ├── check_rules.py
│   ├── generate_brief.py
│   ├── agent.py
│   ├── schemas.py
│   ├── rules/
│   │   ├── specialty_scopes.py
│   │   └── em_distribution.py
│   └── external/
│       └── leie_loader.py
├── api/                                  # FastAPI on Railway
│   ├── Dockerfile, main.py, config.py
│   ├── models/{loader.py, artifacts/}
│   ├── routers/{health,predict,forecast,reference}.py
│   ├── services/{prediction,supabase,specialty_canonicalization}.py
│   └── schemas/
├── web/                                  # Next.js on Vercel
│   ├── src/app/{page.tsx, forecast, investigations, demo, about}
│   ├── src/lib/constants.ts
│   ├── public/data/investigations/       # synced briefs
│   └── scripts/sync-briefs.mjs
├── docs/knowledge/                       # 9-part HTML knowledge base
└── local_pipeline/                       # gitignored
```

## Commands

```bash
# Pull provider-level data (one-time, for risk scores)
python pull_provider_data.py --output-dir data/

# Re-partition raw data (required once to inject year column)
python partition_medicare_data.py --input-dir data --output-dir partitioned_data
python csv_to_parquet.py --dir partitioned_data

# Run the full local pipeline
python notebooks/01_bronze_ingest_local.py
python notebooks/02_silver_clean_local.py
python notebooks/03_gold_features_local.py --provider-data-dir data/
python notebooks/05_lstm_sequences_local.py
python notebooks/04_eda_local.py

# Train models — batch mode (default, memory-efficient)
python modeling/train_glm_local.py
python modeling/train_rf_local.py --mode batch
python modeling/train_xgb_local.py --mode batch
python modeling/train_catboost_local.py --mode batch
python modeling/train_lgbm_local.py --mode batch

# Train models — full mode (needs more RAM/VRAM)
python modeling/train_rf_local.py --mode full --sample 0.5
python modeling/train_xgb_local.py --mode full --sample 0.5
python modeling/train_catboost_local.py --mode full --sample 0.5
python modeling/train_lgbm_local.py --mode full --sample 0.5

# Train LSTM (Phase 3 — time-series forecasting)
python modeling/train_lstm_local.py
python modeling/train_lstm_local.py --epochs 100 --hidden-size 128

# Compare models
python modeling/compare_models_local.py

# MCBS pipeline — Track A (real PUF, national)
python pull_mcbs_data.py --type both
python notebooks/06_mcbs_bronze_local.py
python notebooks/07_mcbs_silver_local.py
python notebooks/08_mcbs_crosswalk_local.py

# MCBS pipeline — Track B (synthetic LDS, regional per-service OOP)
python generate_synthetic_mcbs.py --sample 0.1
python notebooks/08_mcbs_crosswalk_local.py --mode synthetic

# Train Stage 2 OOP quantile regression (P10/P50/P90)
python modeling/train_oop_local.py
python modeling/train_oop_local.py --sample 0.3 --rounds 500
```

## Environment

- **Python deps:** pandas, pyarrow, scikit-learn, xgboost, mlflow, matplotlib, seaborn, scipy, requests, torch, pyspark (Databricks only)
- **Optional GPU deps:** cudf-cu12 (RAPIDS for gold features), cuml-cu12 (RAPIDS for RF full mode), torch CUDA (for LSTM)
- **Env vars (`.env`, gitignored):** `DATABRICKS_HOST`, `DATABRICKS_TOKEN`
- **Platform:** Windows 11 + WSL2 Ubuntu, NVIDIA 5070 Ti (16GB VRAM)

## Progress

See `PROGRESS.md` for detailed milestone tracking, changelogs, model results, and phase-by-phase execution log. Update PROGRESS.md after every phase/milestone completion.

**Current status (2026-04-27):** Project complete and in production.

- **Phases 1-7 (pipeline + Stage 1/Stage 2 modeling):** done. V2 trained on Colab (A100/T4). LightGBM V2 no-charge + CatBoost monotonic OOP deployed.
- **Phase 8 (forecast track):** closed. LightGBM Stacker V2_12 wins (R² 0.8852); multivariate TFT V2_13 confirms signal ceiling ≈0.885 at annual resolution.
- **Phase 9 (Provider Anomaly Investigation Agent):** A-E complete on `main`. 10 rules, 7 evaluable; UPCODING + LEIE_EXCLUDED unlocked in commit 423a26f. Validation run on top-100 2023 flags: 39 CRITICAL / 59 HIGH / 2 MEDIUM, $3.82 total spend.
- **API + Frontend:** FastAPI on Railway (`api/`), Next.js on Vercel (`web/`). Real-time inference replaces pre-computed Supabase lookups.
- **Only outstanding item:** Rule #3 (medical-necessity diagnosis-code linkage) BLOCKED on paid LDS/RIF data subscription. Documented as NOT EVALUABLE in briefs.

## V2 Production Models (Stage 1 + Stage 2)

Trained on Colab Pro (A100/T4) on 2026-04-11. Full spec in [`V2_MODEL_SPEC.md`](V2_MODEL_SPEC.md).

**Stage 1 (allowed amount) leaderboard — fair temporal holdout:**

| Model | R² | MAE | Notes |
|---|---|---|---|
| LightGBM V2 (full) | 0.9575 | $6.73 | Charge-aware variant |
| **LightGBM V2 no-charge** | **0.943** | **~$7** | **PRODUCTION — used by Railway API** |
| Ensemble V2 | 0.9580 | — | +0.0004 over LightGBM, not worth deploying |
| XGBoost V2 | 0.9452 | $7.73 | — |
| CatBoost V2 | 0.9070 | $10.88 | — |

Why no-charge wins: API users don't reliably know `Avg_Sbmtd_Chrg`; charge ablation cost only 0.01-0.02 R². Robustness > marginal accuracy.

**Stage 2 (patient OOP) leaderboard:**

| Model | R² | Status |
|---|---|---|
| XGB Quantile V1 | 0.400 | Best raw R² |
| CatBoost Mono V2 (P10/P50/P90) | 0.173 | **PRODUCTION — monotone constraints valued for product safety** |
| CatBoost ZI V2 | -0.054 | Gate+regression compounded errors |

Notes:
- CatBoost GPU does **not** support `monotone_constraints` — must train on CPU.
- **Asymmetric CQR calibration applied at inference** via `api/models/artifacts/oop_calibration.json` (q_lo=$0.0004, q_hi=$14.47 for 80% interval). Raw P90 only covered 67.5% of actuals — sidecar widens the upper tail to hit the nominal 90% marginal. Reproduce with `python modeling/calibrate_oop.py`. Symmetric q_hat=$3.6884 reproduces V2_04's MLflow log exactly.

**Forecast track (Phase 8) — temporal holdout 2022-2023, N=32,481:**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **LGB Stacker V2_12** | **8.74** | **17.69** | **0.8852** ← PRODUCTION |
| LSTM V1 (autoregressive, fair) | 9.82 | 18.91 | 0.8689 |
| Multivariate TFT V2_13 | 9.23 | 18.79 | 0.8691 |
| Chronos Bolt cpi_cf_deflated | 9.39 | 19.71 | 0.8576 |
| LSTM V1 (teacher-forced, INFLATED) | 8.84 | 36.42 | 0.8860 |

Stacker feature importance: lstm_pred 70.5%, last_history_value 16.0%, chronos_pred 3.6%, history_mean 2.7%, ptype/state 3.1%, history_cv/trend 2.8%. Output: `stacker_forecast_2024_2026.parquet`.

**Known bug:** [`modeling/train_lstm_local.py`](modeling/train_lstm_local.py) `evaluate()` (~line 340) uses 1-step teacher-forced prediction, not autoregressive rollout. Reported R² ≈0.886 is inflated by ~0.017. Cosmetic (V2 stacker is the production forecast model anyway), but should be fixed if anyone re-runs the local LSTM.

## Provider Anomaly Investigation Agent (Phase 9)

End-to-end fraud investigation pipeline on top of the existing silver layer. See [`PROVIDER_ANOMALY_AGENT_SPEC.md`](PROVIDER_ANOMALY_AGENT_SPEC.md) for the full design. Phases A-E complete on `main`.

### Anomaly pipeline files (`anomaly/`)
- `compute_npi_profiles.py` — silver → 11.52M NPI-year profiles with 22 metrics (volume, intensity, charge ratios, Herfindahl, bucket distribution, YoY changes, risk score)
- `compute_benchmarks.py` — specialty/state/national benchmark tables (mean + P5/P25/P50/P75/P95)
- `detect_outliers.py` — z-score (log1p-transformed) + Isolation Forest + temporal-rule detection → `flags.parquet`
- `schemas.py` — `ProviderContext`, `RuleCheckResult`, `InvestigationBrief` dataclasses
- `retrieve_context.py` — `ContextRetriever` builds the evidence package for a (NPI, year); loads E&M and LEIE sidecars
- `check_rules.py` — 10 fraud-indicator rules; 7 evaluable, 3 structurally NOT EVALUABLE with reason
- `generate_brief.py` — Claude API call (`claude-sonnet-4-6` default), 2,584-token system prompt for cache hit, 429/529 retry-with-backoff
- `agent.py` — orchestrator: rank flags → context → rules → brief → markdown + JSON; `--year` filter for targeted runs
- `rules/specialty_scopes.py` — builds `specialty_scopes.parquet` (86,924 rows, 130 specialties, 27% in-scope) for OUT_OF_SPECIALTY rule
- `rules/em_distribution.py` — pre-computes per-NPI E&M counts → `em_distributions.parquet` (5.94M rows) + `em_specialty_benchmarks.parquet` for UPCODING rule
- `external/leie_loader.py` — downloads OIG LEIE UPDATED.csv (~83K rows, 8.4K with NPI) → `leie_exclusions.parquet`. **Use `--insecure`** — OIG TLS chain fails Python's default CA validation locally.

### Rule status (10 total)
**Evaluable (7):** LEIE_EXCLUDED (CRITICAL override under 42 USC 1320a-7), VOLUME_SPIKE, HIGH_INTENSITY, PROCEDURE_CONCENTRATION, CHARGE_INFLATION, OUT_OF_SPECIALTY, UPCODING.
**NOT EVALUABLE (3):** IMPOSSIBLE_DAY, UNBUNDLING, BENEFICIARY_SHARING — all require per-claim/date-of-service/beneficiary linkage not in CMS public PUF. Future Rule #3 (medical-necessity Dx linkage) blocked on paid LDS/RIF.

### Anomaly outputs (`local_pipeline/anomaly/`, gitignored)
- `npi_profiles.parquet` — 11.52M rows (1.76M unique NPIs × 11 years)
- `specialty_benchmarks.parquet`, `state_specialty_benchmarks.parquet`, `national_benchmarks.parquet`
- `specialty_scopes.parquet`, `em_distributions.parquet`, `em_specialty_benchmarks.parquet`, `leie_exclusions.parquet`
- `flags.parquet` — long-format detection flags; composite (≥2 methods) hits 0.77% of NPI-years
- `briefs/{NPI}_{YEAR}.md` / `.json` and `briefs_2023_validation/` — generated investigation briefs

### Validation run (n=100, year=2023, Sonnet 4.6 with caching)
- **Severity:** 39 CRITICAL, 59 HIGH, 2 MEDIUM. Total spend: $3.82.
- **Rule triggers:** VOLUME_SPIKE 66, HIGH_INTENSITY 59, PROCEDURE_CONCENTRATION 6, CHARGE_INFLATION 6, OUT_OF_SPECIALTY 3, UPCODING 0, LEIE_EXCLUDED 0.
- **Why UPCODING/LEIE = 0:** Top-100 composite flags are dominated by Nurse Practitioners and Mass Immunizers (E&M saturated at P95=100% across 68/84 specialties; LEIE intersect with composite flags = 0).
- **Top specialties:** Nurse Practitioner (38), Gastroenterology (8), Mass Immunizer + Emergency Medicine (7 each).

### Key interpretive note
"Specialty scope" is defined empirically from what a specialty's population actually bills, NOT from regulatory scope-of-practice rules. Optometrist billing cataract surgery (66984) is NOT flagged by OUT_OF_SPECIALTY because 28% of optometrists in the data bill it (co-management). Same NPI still surfaces CRITICAL via HIGH_INTENSITY + PROCEDURE_CONCENTRATION.

### Anomaly commands
```bash
# 1. Build NPI profiles (~3 min) and benchmarks (~30s)
python anomaly/compute_npi_profiles.py
python anomaly/compute_benchmarks.py

# 2. Build sidecars
python anomaly/rules/specialty_scopes.py
python anomaly/rules/em_distribution.py
python anomaly/external/leie_loader.py --insecure

# 3. Run detection (~3 min)
python anomaly/detect_outliers.py

# 4. Generate investigation briefs
# Dry-run (formats prompts, no API spend)
python anomaly/agent.py --top-n 10

# Live run (Claude API; ~$0.04/brief on Sonnet 4.6 with cache)
python anomaly/agent.py --top-n 100 --year 2023 --live \
    --env-path "C:/Users/rajas/Documents/ADS/coverdrive_pred_11/.env"
```

### Anomaly agent dependencies
- `anthropic >= 0.97`, `python-dotenv` (in addition to base pipeline deps)
- `ANTHROPIC_API_KEY` env var; helper loads from `--env-path` with `override=True` (necessary when an empty shell var would otherwise block dotenv)
- API key location: `C:/Users/rajas/Documents/ADS/coverdrive_pred_11/.env` (cross-project)
- **Privacy:** All briefs use real CMS public data, not synthetic. NPIs are public via NPPES but briefs reference individuals — use the redaction toggle (or `NEXT_PUBLIC_REDACT_NPIS=1`) for any demo, screenshot, or public deploy.

## Backend (FastAPI on Railway)

Live at `medicare-provider-utilization-cost-analysis-production.up.railway.app`. Real-time inference replaces pre-computed Supabase group-average lookups.

- **Stage 1:** `lgbm_v2_no_charge.txt` (1000 trees, R²~0.943). Auto-detects charge/no-charge from feature names.
- **Stage 2 OOP:** `oop_mono_{p10,p50,p90}.cbm` with `OOP_CAT_IDX = [2,3,4,5]`.
- **Reference data** (forecasts, labels, metrics) proxied from Supabase server-side.
- **Specialty canonicalization** in `api/services/specialty_canonicalization.py` — collapses CMS specialty-name splits (Cardiology, Colorectal, Oral Surgery) to a single canonical form (commit cc248ae).
- Model artifacts bundled in `api/models/artifacts/` (~26 MB, committed to git).
- Supabase uses anon key for now; should switch to service role key for production hardening.

### API structure
```
api/
├── Dockerfile
├── main.py
├── config.py
├── requirements.txt
├── models/
│   ├── loader.py
│   └── artifacts/        # lgbm_v2_no_charge.txt, oop_mono_*.cbm
├── routers/
│   ├── health.py
│   ├── predict.py
│   ├── forecast.py
│   └── reference.py
├── services/
│   ├── prediction.py
│   ├── supabase.py
│   └── specialty_canonicalization.py
└── schemas/
```

## Frontend (Next.js + MUI on Vercel)

`web/` — Next.js app. Routes:
- `/` — Estimator (Stage 1 + Stage 2 inference)
- `/forecast` — Specialty-level 2024-2026 forecast viewer
- `/investigations` + `/investigations/[id]` — Phase 9 brief list + detail
- `/demo` — Product demo page (`AllowanceMap_ProductDemo.mp4`)
- `/about` — Project info

Key conventions:
- Reads `NEXT_PUBLIC_API_URL` (Vercel env + `web/.env.local`) → Railway.
- `web/scripts/sync-briefs.mjs` copies Phase 9 briefs to `web/public/data/investigations/` and parses per-rule summaries. Supports `--mask-npis` flag and `MASK_NPIS=1` env for public deploys.
- Runtime NPI redaction toggle in `/investigations` list + detail headers (format `1033****74`). Preference in localStorage with cross-tab `CustomEvent` sync. `NEXT_PUBLIC_REDACT_NPIS=1` locks it on.
- `web/CLAUDE.md` notes: this is **not the Next.js you know** — read `node_modules/next/dist/docs/` before writing app code.

## Notes

- `.env` contains Databricks credentials — never commit or expose
- `local_pipeline/` and `data/` / `partitioned_data/` are gitignored
- Databricks notebooks use `# COMMAND ----------` cell separators
- XGBoost auto-detects CUDA; RF batch mode is sklearn-only (cuML lacks warm_start)
- `--sample` flag on training scripts controls fraction of data loaded (default 0.3)
- Gold parquets use float32 for RF (halves VRAM) and float64 for XGBoost/GLM
- Year column must be re-injected by re-running `partition_medicare_data.py` from raw CSVs if missing
- Provider risk scores (`Bene_Avg_Risk_Scre`) come from a separate CMS dataset; missing scores imputed with global median (~1.0)
- **Compute estimates were 3-5x too low** on V2 training (CatBoost 3000 iters on 126.8M rows = ~105 min on A100, ensemble 5-fold = 13.3 hrs). For 100M+ row training, multiply initial estimates by 3-4x.
