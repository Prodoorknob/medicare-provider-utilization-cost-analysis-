# Medicare Provider Cost & Patient OOP Prediction Pipeline — Progress Log

## Project Timeline & Milestones

> Last updated: 2026-04-04
> Author: Raj Vedire (rvedire@iu.edu)
> Repo: medicare-provider-utilization-cost-analysis-

---

## Phase 1 — Academic Baseline ✅ COMPLETE

**Goal:** Full Medallion pipeline on representative data, 3 model families trained with MLflow tracking.

### Milestone 1.1 — Data Infrastructure (Early 2026)
- [x] CMS API pull script (`pull_medicare_data.py`) — downloads 2013-2023 "by Provider & Service" data
- [x] Partitioning script (`partition_medicare_data.py`) — splits raw CSVs by STATE/PROVIDER_TYPE
- [x] Parquet conversion (`csv_to_parquet.py`) — converts partitioned CSVs to columnar format
- [x] S3 bucket configured for Databricks
- [x] Databricks Git integration live
- [x] Project proposal delivered

### Milestone 1.2 — Medallion Pipeline (April 4, 2026)
- [x] Bronze layer — raw ingest with schema preservation (`01_bronze_ingest.py` / `_local.py`)
- [x] Silver layer — type casting, null handling, IQR outlier removal (`02_silver_clean.py` / `_local.py`)
- [x] Gold layer — feature engineering, label encoding, per-state parquets (`03_gold_features.py` / `_local.py`)
- [x] EDA — distribution plots, correlation heatmap, provider/state summaries (`04_eda.py` / `_local.py`)

### Milestone 1.3 — Model Training & Comparison (April 4, 2026)
- [x] GLM baseline — SGDRegressor with partial_fit streaming, Huber loss
- [x] Random Forest — RandomizedSearchCV, cuML/sklearn auto-detection
- [x] XGBoost — early stopping, CUDA auto-detection
- [x] MLflow experiment tracking — all runs logged to Databricks workspace
- [x] Model comparison script with paired t-test (`compare_models_local.py`)

### Key Decisions Made (Phase 1)
| Decision | Rationale |
|---|---|
| Removed `Avg_Mdcr_Alowd_Amt`, `Avg_Mdcr_Stdzd_Amt` as features | Data leakage — these are derived from the target (R² was 0.9996 with them) |
| Removed `pymt_to_charge_ratio`, `stdz_to_pymt_ratio` | Also derived from leaky columns |
| Used `log1p` target transform | Right-skewed cost distribution spans orders of magnitude |
| 80/20 train/test split, `random_state=42` | Consistency across all models |

---

## Phase 2 — National Scale ✅ COMPLETE

**Goal:** Retrain all models on complete 50-state + territories dataset with production-grade features aligned to project spec.

### Milestone 2.1 — Target Variable Correction (April 4, 2026)
- [x] Swapped target from `Avg_Mdcr_Pymt_Amt` (payment) to `Avg_Mdcr_Alowd_Amt` (allowed amount)
- [x] Rationale: Stage 1 predicts what Medicare *allows*, Stage 2 predicts patient OOP from that
- [x] Updated all 7 modeling scripts + silver cleaning + EDA + CLAUDE.md
- [x] IQR outlier bounds now computed on allowed amount

### Milestone 2.2 — New Feature Engineering (April 4, 2026)
- [x] `hcpcs_bucket` — coarse clinical category from CPT code ranges (0-5: Anesthesia, Surgery, Radiology, Lab, Medicine, HCPCS-II)
- [x] `place_of_srvc_flag` — binary facility/office indicator (already in CMS data, just not extracted before)
- [x] `Bene_Avg_Risk_Scre` — HCC risk score from separate CMS "by Provider" dataset, joined on NPI+year
- [x] `log_srvcs`, `log_benes` — log-transformed service/beneficiary counts (replaced raw counts)
- [x] Kept `HCPCS_Cd_idx` (raw) AND `hcpcs_bucket` (coarse) — models can use either granularity

### Milestone 2.3 — Provider Risk Score Integration (April 4, 2026)
- [x] Created `pull_provider_data.py` — downloads CMS "by Provider" dataset (2013-2023, ~470 MB/year)
- [x] Direct CSV download from CMS catalog (no API pagination needed)
- [x] NPI + year join in gold features script
- [x] Missing risk scores imputed with global median (~1.0)

### Milestone 2.4 — Regional Batch Training Architecture (April 4, 2026)
- [x] Gold layer outputs per-state parquets (`gold/{STATE}.parquet`) instead of monolithic file
- [x] `label_encoders.json` persisted for cross-script consistency
- [x] XGBoost incremental training by Census region (125 rounds/region, booster continuation via `xgb_model`)
- [x] RF warm_start training by Census region (125 trees/region, sklearn only — cuML lacks warm_start)
- [x] GLM streaming via directory-based `iter_row_groups()` across all state parquets
- [x] `--mode batch|full` flag on XGB and RF scripts
- [x] Census regions defined: Northeast, South, Midwest, West, Territories

### Milestone 2.5 — Temporal Data Preservation (April 4, 2026)
- [x] `year` column injected at partition stage (extracted from CSV filename via regex)
- [x] Carried through Bronze → Silver → Gold as int16
- [x] Year preserved in gold schema as metadata (not used as model feature for tree/GLM)
- [x] Essential for Phase 3 LSTM time-series modeling

### Milestone 2.6 — LSTM Sequence Preparation (April 4, 2026)
- [x] Created `05_lstm_sequences_local.py`
- [x] Groups gold data by (provider_type × hcpcs_bucket × state)
- [x] Produces year-ordered target vectors per group
- [x] Output: `lstm/sequences.parquet` — 23,672 groups, avg 7.6 years/group, 10,540 with all 11 years
- [x] Ready for Phase 3 LSTM model consumption

### Milestone 2.7 — National Scale Training Results (April 4, 2026)
- [x] All 3 models trained on full national dataset (~103M rows)
- [x] All runs logged to unified MLflow experiment: `/Users/rvedire@iu.edu/medicare_models`

**Final Gold Schema (12 columns):**
```
year (int16) | Rndrng_Prvdr_Type_idx | Rndrng_Prvdr_State_Abrvtn_idx | HCPCS_Cd_idx
hcpcs_bucket | place_of_srvc_flag | Bene_Avg_Risk_Scre | log_srvcs | log_benes
Avg_Sbmtd_Chrg | srvcs_per_bene | Avg_Mdcr_Alowd_Amt (TARGET)
```

**National Scale Model Results:**
| Model | Test MAE | Test RMSE | Test R² | Notes |
|---|---|---|---|---|
| Random Forest | $12.04 | $22.73 | 0.8843 | Best performer, 625 trees, warm_start batch |
| XGBoost | $11.83 | $21.18 | 0.8331 | Incremental by region, 500 rounds |
| GLM (SGD) | $51.77 | $521.90 | -102.80 | Diverged — needs tuning (Huber loss, adaptive LR) |

**Feature Importances (RF, top 5):**
1. `Avg_Sbmtd_Chrg` — 0.618 (submitted charge dominates)
2. `HCPCS_Cd_idx` — 0.202 (procedure code)
3. `hcpcs_bucket` — 0.064 (clinical category)
4. `Rndrng_Prvdr_Type_idx` — 0.044 (provider specialty)
5. `srvcs_per_bene` — 0.032 (utilization intensity)

### GPU & Infrastructure Notes
- XGBoost CUDA auto-detection added (`_detect_device()`)
- cuML auto-detection for RF full mode
- RF batch mode runs on CPU (sklearn warm_start) — ran on Windows (32GB RAM) due to WSL OOM at 24GB
- Bronze ingest OOM on full dataset (6,313 files) — skippable since silver reads from partitioned_data directly
- Provider data pull: 11 years × ~470 MB = ~5 GB CSV downloads

---

## Phase 3 — LSTM Time-Series Forecasting 🏗️ CODE COMPLETE (pending training)

**Goal:** Build LSTM model on year-indexed sequences to forecast allowed amounts 2-3 years forward by specialty × HCPCS bucket × state.

### Milestone 3.1 — PyTorch LSTM Architecture (April 7, 2026)
- [x] `MedicareLSTM` model: static embeddings (provider_type, state, hcpcs_bucket) + 2-layer LSTM + linear head
- [x] Teacher forcing: input=seq[:-1] → predict seq[1:]
- [x] ~50-80K params, CUDA auto-detection, batch_first=True
- [x] MC Dropout for inference-time uncertainty

### Milestone 3.2 — Temporal Train/Val Split (April 7, 2026)
- [x] Train on years ≤ 2021, validate on 2022-2023 predictions
- [x] `MedicareSequenceDataset` with variable-length padding and val_mask
- [x] Early stopping (patience=10), ReduceLROnPlateau, gradient clipping
- [x] Masked MSE loss (ignores padding positions)

### Milestone 3.3 — Confidence-Bounded Forecasting (April 7, 2026)
- [x] Autoregressive rollout 3 years forward (2024-2026)
- [x] MC Dropout: 50 stochastic passes per group for uncertainty bounds
- [x] Output: forecast_mean, forecast_std, P10/P50/P90 per group per year
- [x] Saved to `local_pipeline/lstm/forecast_2024_2026.parquet`

### Milestone 3.4 — Specialty Trend Visualization (April 7, 2026)
- [x] `specialty_trends.png` — 4x3 grid, top 12 specialties with P10-P90 forecast bands
- [x] `forecast_distribution.png` — histogram of 2026 forecast means
- [x] `top_growth_specialties.png` — top 15 specialties by projected cost growth %

### Milestone 3.5 — MLflow Logging (April 7, 2026)
- [x] Run name: `lstm_local`, same experiment as Stage 1 models
- [x] Logs: all hyperparams, test_mae/rmse/r2, PyTorch model, forecast parquet, plots
- [x] Integrated into `compare_models_local.py` (LSTM added to MODEL_RUN_NAMES)

### Milestone 3.6 — Forecast Output (April 7, 2026)
- [x] Schema: group keys + forecast_year + forecast_mean/std/p10/p50/p90 + last_known_year/value + n_history_years

### Input Data Ready
- `local_pipeline/lstm/sequences.parquet` — 23,672 groups
- 10,540 groups with all 11 years (2013-2023)
- 2,771 groups with < 3 years (will be filtered)

### Status
**Code complete.** Training deferred — run `python modeling/train_lstm_local.py` when GPU is available.

---

## Phase 4 — MCBS Integration 🏗️ CODE COMPLETE (pending data download)

**Goal:** Ingest Medicare Current Beneficiary Survey data, build parallel Bronze/Silver pipeline, create crosswalk.

### Milestone 4.1 — MCBS PUF Download Script (April 7, 2026)
- [x] `pull_mcbs_data.py` — downloads Survey File + Cost Supplement ZIPs from data.cms.gov
- [x] Survey File URLs: 2015-2022 (8 years)
- [x] Cost Supplement URLs: 2018-2022 (5 years)
- [x] Streaming download with resume detection, ZIP extraction to CSV
- [x] `--url` override for when CMS URLs change
- [x] No Data Use Agreement required (PUF is public)

### Milestone 4.2 — MCBS Bronze Ingest (April 7, 2026)
- [x] `notebooks/06_mcbs_bronze_local.py` — CSV → per-year Parquet with provenance
- [x] Reads with `dtype=str` to prevent type inference (same as provider pipeline)
- [x] BASEID validation (auto-detects variant column names)
- [x] Output: `mcbs_bronze/survey_{YEAR}.parquet`, `mcbs_bronze/cost_{YEAR}.parquet`

### Milestone 4.3 — MCBS Silver Cleaning (April 7, 2026)
- [x] `notebooks/07_mcbs_silver_local.py` — Cost Supplement cleaning (primary Stage 2 input)
- [x] COST_RENAME: maps actual CMS columns (PAMTOOP, CSP_AGE, etc.) to canonical names
- [x] Derived features: age_band, oop_share, has_medicaid, has_private_ins
- [x] Global IQR outlier removal on pay_oop (3x IQR, non-zero values)
- [x] Survey weight preserved (CSPUFWGT) for population estimates
- [x] Output: `mcbs_silver/{YEAR}.parquet`

### Milestone 4.4 — Specialty × MCBS Crosswalk (April 7, 2026)
- [x] `notebooks/08_mcbs_crosswalk_local.py` — national-level bridge table
- [x] Provider aggregation: (specialty × bucket × year) → mean allowed amt, risk, charge
- [x] MCBS aggregation: (year) → mean OOP, OOP share, age, chronic, income, Medicaid/private rates
- [x] Join on year (national level — see PUF constraints below)
- [x] Output: `mcbs_crosswalk/crosswalk.parquet`

### PUF Constraints Discovered
- **No Census region** in MCBS PUF (suppressed for privacy) — crosswalk is national, not regional
- **Survey File and Cost Supplement PUF_IDs do NOT overlap** — independent samples, cannot join at beneficiary level
- **Survey File** has 3 seasonal rounds (fall/winter/summer) with different columns per round, merged on PUF_ID
- **Cost Supplement** has its own demographics (CSP_AGE, CSP_SEX, CSP_RACE, CSP_INCOME, CSP_NCHRNCND) — self-contained for Stage 2
- **OOP target variable:** `PAMTOOP` (renamed to `pay_oop` in Silver)

### Milestone 4.5 — Dual-Track Architecture (April 7, 2026)
- [x] **Track A (Real PUF):** National-level annual OOP from public data — already built
- [x] **Track B (Synthetic LDS):** `generate_synthetic_mcbs.py` — per-service OOP with Census region
- [x] Synthetic data derived from real MCBS distributions + real provider gold data
- [x] Demographic sampling: age, sex, income, chronic_count from real MCBS year distributions
- [x] OOP modulation: dual_eligible (-85%), supplemental (-35%), chronic count scaling, log-normal noise
- [x] `08_mcbs_crosswalk_local.py` now has `--mode puf|synthetic` flag
- [x] Synthetic mode: aggregates by (region, specialty, bucket, year) instead of just (year)
- [x] Metadata + clear "SYNTHETIC" labeling for reproducibility
- [x] Drop-in replacement: users with real MCBS LDS can swap the synthetic parquet and run the same pipeline

### Status
**Code complete.** Run pipeline:
- Track A: `python pull_mcbs_data.py` → `06_mcbs_bronze` → `07_mcbs_silver` → `08_mcbs_crosswalk --mode puf`
- Track B: `python generate_synthetic_mcbs.py` → `08_mcbs_crosswalk --mode synthetic`

---

## Phase 5 — Stage 2 OOP Model 🏗️ CODE COMPLETE (pending training)

**Goal:** Train quantile regression on MCBS features to predict patient out-of-pocket costs (P10/P50/P90).

### Milestone 5.1 — Stage 2 Feature Matrix (April 7, 2026)
- [x] 12-feature matrix combining provider-side and beneficiary-side columns
- [x] Provider features: Avg_Mdcr_Alowd_Amt (Stage 1 target → Stage 2 feature), risk score, specialty, bucket, POS
- [x] Beneficiary features: census_region, age, sex, income, chronic_count, dual_eligible, has_supplemental
- [x] Data source: `mcbs_synthetic/synthetic_oop.parquet` (10.3M rows) — drop-in replaceable with real LDS

### Milestone 5.2 — Quantile XGBoost Training (April 7, 2026)
- [x] `modeling/train_oop_local.py` — 3 separate XGBoost boosters (P10, P50, P90)
- [x] Objective: `reg:quantileerror` with `quantile_alpha` = 0.1, 0.5, 0.9
- [x] Same hyperparams as Stage 1: lr=0.05, max_depth=6, subsample=0.8, hist, CUDA auto-detect
- [x] Early stopping (patience=30), 300 rounds default

### Milestone 5.3 — Evaluation (April 7, 2026)
- [x] 80/20 random split (random_state=42, matches Stage 1)
- [x] Metrics per quantile: MAE, RMSE, R2, coverage (should be ~10/50/90%), pinball loss
- [x] P50 metrics logged as `test_mae/rmse/r2` for comparison table compatibility

### Milestone 5.4 — MLflow Logging (April 7, 2026)
- [x] Run name: `xgb_quantile_oop_local`, same experiment as Stage 1
- [x] 3 model artifacts: `xgb_oop_p10`, `xgb_oop_p50`, `xgb_oop_p90`
- [x] Feature importances per quantile
- [x] `compare_models_local.py` updated with Stage 2 OOP section (separate from Stage 1 table)

### Status
**Code complete.** Run `python modeling/train_oop_local.py` (~3-4 GB RAM with default --sample 0.3).

---

## Phase 6 — Next.js Portfolio Web App ✅ COMPLETE

**Goal:** Patient-facing query interface combining both stages + LSTM forecast.

**Live URL:** https://web-three-omega-21.vercel.app

### Milestone 6.1 — Architecture & Data Layer (April 8, 2026)
- [x] Supabase project created (us-east-1, project `zdkoniqnvbklxtsviikl`)
- [x] 7 PostgreSQL tables with RLS (public SELECT) — 119K+ total rows
- [x] Pre-computed Stage 1 aggregations from 103M gold rows → 32,818 groups
- [x] Pre-computed Stage 2 OOP quantiles from 10.3M synthetic rows → 23,424 groups
- [x] LSTM forecasts uploaded: 62,703 rows (20,901 groups × 3 years)
- [x] Label encoders, state summary, model metrics, feature importances

### Milestone 6.2 — Next.js App (April 8, 2026)
- [x] Next.js 16 + Material UI + Recharts + Supabase JS client
- [x] 6 pages: Dashboard, Cost Estimator, Forecast Explorer, Model Comparison, Data Explorer, About
- [x] Two-stage cost estimation with fallback queries for missing combinations
- [x] Interactive LSTM forecast chart with P10-P90 confidence bands
- [x] Feature importance bar chart, model metrics table, methodology accordions
- [x] State summary table with sorting/filtering, EDA plot gallery
- [x] Responsive layout (mobile drawer nav, stacked forms)

### Milestone 6.3 — Deployment (April 8, 2026)
- [x] Deployed to Vercel (production)
- [x] Environment variables: NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY

### Tech Stack
- **Frontend:** Next.js 16.2.3 (App Router, Turbopack), Material UI v7, Recharts
- **Backend:** Supabase (PostgreSQL) — pre-computed predictions, no live model inference
- **Deployment:** Vercel
- **Data Bridge:** Python scripts aggregate local_pipeline parquets → Supabase REST API

---

## Changelog

### 2026-04-08
- **Phase 6 complete** — Next.js portfolio web app deployed to Vercel
- Created Supabase project with 7 tables (lookup_labels, stage1_allowed_amounts, stage2_oop_estimates, lstm_forecasts, state_summary, model_metrics, feature_importances)
- Pre-computed Stage 1 allowed amounts by (specialty × bucket × state × POS): 32,818 groups from 103M rows
- Pre-computed Stage 2 OOP quantiles by (specialty × bucket × region × dual × supplemental × age × income): 23,424 groups
- Built 6-page Next.js app: Dashboard, Cost Estimator, Forecast Explorer, Model Comparison, Data Explorer, About
- Cost Estimator: two-stage flow with Supabase lookups, auto-detected census region, P10/P50/P90 OOP
- Forecast Explorer: interactive Recharts chart + pre-generated LSTM specialty trend plots
- Model Comparison: metrics table, RF feature importance chart, methodology accordions
- **Phase 3 LSTM trained** — ran in WSL Ubuntu with PyTorch CUDA on RTX 5070 Ti
- LSTM: R²=0.886, MAE=$8.84 (val 2022-2023), early stop epoch 19, 213K params
- Fixed `train_lstm_local.py`: label_encoders list→dict handling, numpy array `.index()` → `np.where()`
- **Phase 5 OOP trained** — XGBoost quantile regression in WSL
- OOP: P50 R²=0.40, P50 coverage=50.0%, P90 coverage=90.0%

### 2026-04-07
- **Phase 4 code complete** — MCBS integration pipeline (download + Bronze + Silver + crosswalk)
- Created `pull_mcbs_data.py`: downloads Survey File (2015-2022) + Cost Supplement (2018-2023) ZIPs from CMS
- Created `06_mcbs_bronze_local.py`: handles CMS dir structure (SFPUF/CSPUF subdirs), merges 3 survey rounds on PUF_ID
- Created `07_mcbs_silver_local.py`: Cost Supplement cleaning with actual CMS column names (PAMTOOP, CSP_AGE, etc.)
- Created `08_mcbs_crosswalk_local.py`: national-level crosswalk (no region in PUF), join on year
- **Key PUF constraints discovered:** no Census region, Survey/Cost PUF_IDs don't overlap, Cost Supplement self-contained
- Derived features: age_band, oop_share, has_medicaid, has_private_ins
- **Dual-track architecture** — Track A (real PUF, national) + Track B (synthetic LDS, regional)
- Created `generate_synthetic_mcbs.py`: samples demographics from real MCBS, modulates OOP by dual/supplemental/chronic
- Updated `08_mcbs_crosswalk_local.py` with `--mode puf|synthetic` flag
- Synthetic data clearly labeled; drop-in replacement for real LDS data
- **Phase 5 code complete** — Stage 2 OOP quantile regression
- Created `modeling/train_oop_local.py`: 3 XGBoost quantile boosters (P10/P50/P90) on 12 features
- Avg_Mdcr_Alowd_Amt (Stage 1 target) becomes Stage 2 input feature — two-stage pipeline connected
- Metrics: MAE, RMSE, R2, coverage, pinball loss per quantile
- Updated `compare_models_local.py` with Stage 2 OOP section (separate table)
- **Phase 3 code complete** — LSTM time-series forecasting (training deferred)
- Created `modeling/train_lstm_local.py`: MedicareLSTM with static embeddings, temporal split, MC Dropout forecasting
- Architecture: 2-layer LSTM, embed_dim=8 for group keys, AdamW + ReduceLROnPlateau + early stopping
- Temporal split: train on years <= 2021, validate on 2022-2023 predictions
- Autoregressive forecast 2024-2026 with 50 MC Dropout samples for P10/P50/P90 confidence bounds
- 3 visualization charts: specialty trends with forecast bands, forecast distribution, top growth specialties
- MLflow integration: run_name=`lstm_local`, logs model + forecast parquet + plots
- Updated `compare_models_local.py`: LSTM added to MODEL_RUN_NAMES, excluded from paired t-test (sequence model)
- Updated CLAUDE.md with LSTM commands, torch dependency, file layout

### 2026-04-04
- **Phase 2 complete** — national scale training with 10-feature gold schema
- Swapped target: `Avg_Mdcr_Pymt_Amt` → `Avg_Mdcr_Alowd_Amt` (aligned with spec Stage 1)
- Added 5 new features: `hcpcs_bucket`, `place_of_srvc_flag`, `Bene_Avg_Risk_Scre`, `log_srvcs`, `log_benes`
- Created `pull_provider_data.py` for CMS "by Provider" risk scores
- Implemented regional batch training (XGBoost incremental, RF warm_start) by Census region
- Added `year` column through entire pipeline for LSTM readiness
- Per-state gold parquets replace monolithic gold.parquet
- Created `05_lstm_sequences_local.py` — 23,672 LSTM-ready sequence groups
- RF wins at R²=0.8843 on national data, XGBoost at 0.8331
- GLM diverged (needs hyperparameter tuning)
- Unified MLflow experiment: all models under `/Users/rvedire@iu.edu/medicare_models`

### 2026-03-30
- **Phase 1 baseline** — initial pipeline, 3 models, Databricks integration
- Data infrastructure: CMS API pull, partitioning, Parquet conversion
- Databricks notebooks for Bronze/Silver/Gold/EDA
- Local training variants with MLflow → Databricks logging
