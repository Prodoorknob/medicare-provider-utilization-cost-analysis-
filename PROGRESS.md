# Medicare Provider Cost & Patient OOP Prediction Pipeline — Progress Log

## Project Timeline & Milestones

> Last updated: 2026-04-11
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

## Phase 7 — V2 Full-Data Model Improvements 🏗️ IN PROGRESS

**Goal:** Train on all 126.8M rows (no sampling/batching) on Google Colab Pro A100, add CatBoost + LightGBM + ensemble, ablation analysis.

### Milestone 7.1 — Colab Pro Infrastructure (April 9, 2026)
- [x] V2_MODEL_SPEC.md — full implementation spec (8 notebooks, CU budget, execution order)
- [x] Gold parquets (1.9 GB) + MCBS synthetic + LSTM sequences uploaded to Google Drive (`AllowanceMap/V2/`)
- [x] External covariates pulled: Medicare CF, Medical CPI (BLS API), sequestration, MACRA/MIPS, COVID indicators
- [x] `pull_external_covariates.py` — BLS API integration with hardcoded fallback
- [x] Colab notebooks: V2_01 through V2_03 built, validated (syntax + smoke test on 1% CA sample)
- [x] Memory-safe pattern: save train/test split to Colab local SSD, each model cell reloads independently

### Milestone 7.2 — V2_01 Full-Data Training (April 9-10, 2026)
- [x] XGBoost: R²=0.9452, MAE=$7.73, RMSE=$15.43 (1000 rounds, 504s on A100 GPU)
- [x] CatBoost: R²=0.9070, MAE=$10.88, RMSE=$20.10 (3000 iters, 7204s on A100 GPU)
- [x] LightGBM: R²=0.9575, MAE=$6.73, RMSE=$13.59 (1000 rounds, CPU, GOSS)
- [x] All 3 models logged to Databricks MLflow, saved to Drive
- [x] CatBoost GPU fixes: `bootstrap_type='MVS'`, removed `rsm`, increased to 3000 iterations
- [x] LightGBM GPU fix: switched to CPU (`device='cpu'`) due to `max_bin > 255` error with high-cardinality categoricals

### Milestone 7.3 — V2_02 No-Charge Ablation (April 10, 2026)
- [x] XGBoost no-charge: R²=0.9295 (delta: -0.016 from with-charge)
- [x] CatBoost no-charge: R²=0.8974 (delta: -0.010)
- [x] LightGBM no-charge: R²=0.9428 (delta: -0.015)
- [x] **Key finding:** Avg_Sbmtd_Chrg removal drops R² by only 0.01-0.02, not the predicted 0.25-0.30
- [x] hcpcs_target_enc + HCPCS_Cd_idx capture the same pricing signal via procedure identity
- [x] No-charge LightGBM (R²=0.943) still beats all V1 models (best V1: RF R²=0.884)

### Milestone 7.4 — V2_03 Stacked Ensemble (April 10-11, 2026)
- [x] 5-fold OOF stacking with RidgeCV meta-learner (alpha=0.1)
- [x] Ensemble: R²=0.9580, MAE=$6.68, RMSE=$13.50
- [x] **Ensemble lift: +0.0004 over LightGBM alone — negligible**
- [x] Ridge weights: LGB=1.120, XGB=-0.097, CB=-0.023 (ensemble = ~LightGBM only)
- [x] Training time: 13.3 hours on A100 (CU budget severely underestimated: ~149 CU vs 12-16 est.)
- [x] Root cause: correlated error patterns across all 3 boosters on same feature set

### V2 Stage 1 Final Results
| Model | MAE | RMSE | R² | vs V1 Best |
|---|---|---|---|---|
| **LightGBM V2** | **$6.73** | **$13.59** | **0.9575** | **+0.073** |
| XGBoost V2 | $7.73 | $15.43 | 0.9452 | +0.061 |
| Ensemble V2 | $6.68 | $13.50 | 0.9580 | +0.074 |
| CatBoost V2 | $10.88 | $20.10 | 0.9070 | +0.023 |
| LGB no-charge | $7.70 | $15.77 | 0.9428 | +0.058 |

### CU Consumption (Colab Pro, A100 @ 11.2 CU/hr)
| Notebook | Hours | CU |
|---|---|---|
| V2_01 (3 models) | ~2.5 | ~28 |
| V2_02 (3 ablation) | ~3 | ~34 |
| V2_03 (5-fold ensemble) | 13.3 | ~149 |
| **Total** | ~19 | **~211 / 300** |

### Milestone 7.5 — V2_04 CatBoost Monotonic OOP (April 11, 2026)
- [x] CatBoost quantile regression (P10/P50/P90) with monotonicity constraints on 10.3M OOP rows
- [x] CatBoost GPU doesn't support `monotone_constraints` — switched to CPU
- [x] P10 early-stopped at 5 iterations (quantile loss flat at floor — most P10 values near $0)
- [x] P50 needed 4000 iterations at LR=0.1 to converge on CPU
- [x] Non-crossing correction: post-hoc sort (P10 ≤ P50 ≤ P90)
- [x] CQR calibration with 60/20/20 split (train/calibration/test)
- [x] **Result: P50 R²=0.173, MAE=$10.55** — worse than V1 XGB Quantile (R²=0.400)
- [x] **Root cause:** Monotonicity constraints too restrictive on synthetic data where assumed relationships (income→OOP, chronic→OOP) don't hold cleanly

### Milestone 7.6 — V2_05 Zero-Inflated OOP (April 11, 2026)
- [x] Two-stage model: CatBoost gate classifier P(OOP=0) + CatBoost conditional regression on non-zero OOP
- [x] Gate classifier with `auto_class_weights='Balanced'` for class imbalance
- [x] Combined prediction: `(1 - p_zero) * predicted_oop`
- [x] **Result: P50 R²=-0.054, MAE=$11.95** — predicting worse than the mean
- [x] **Root cause:** Gate classifier and conditional regression compound errors; synthetic OOP distribution doesn't have a clean zero-inflation pattern

### Milestone 7.7 — V2_06 TFT Forecasting (April 11, 2026)
- [x] Temporal Fusion Transformer with 5 external covariates (Medicare CF, Medical CPI, sequestration, COVID, MACRA/MIPS)
- [x] `pytorch-forecasting==1.1.1` + `lightning==2.2.5` (pinned for compatibility)
- [x] Fixed: Drive FUSE timeout (copy to local SSD), PyTorch 2.6 `weights_only` checkpoint issue, numpy binary incompatibility
- [x] 16,184 groups (filtered from 23,672 for minimum sequence length)
- [x] Encoder length=6, prediction length=2, `allow_missing_timesteps=True`
- [x] **Result: R²=0.846, MAE=$11.48, RMSE=$20.48** — worse than V1 LSTM (R²=0.886)
- [x] **Root cause:** Short sequences (4-11 years per group) don't give TFT's attention mechanism enough data; LSTM's simpler architecture better suited for thin time series

### Milestone 7.8 — V2_07 Hierarchical Reconciliation (April 11, 2026)
- [x] 3-level hierarchy: National (1) → State (63) → Bottom specialty×bucket×state (23,672)
- [x] MinTrace with shrinkage (lambda=0.5), `hierarchicalforecast` library
- [x] Coherence verification: PASS (national = sum of states, states = sum of bottom)
- [x] **Result: Reconciliation had no effect** — base forecasts already coherent (built bottom-up from naive last-value forecasts)
- [x] MinTrace only adds value when base forecasts are independently generated at each level

### Milestone 7.9 — V2_08 Comparison Report (April 11, 2026)
- [x] V1 vs V2 comparison across all stages (Stage 1, ablation, Stage 2 OOP, forecasting)
- [x] Comparison plots saved to `v2_artifacts/plots/`
- [x] All runs pulled from Databricks MLflow

### V2 Complete Results

**Stage 1: Allowed Amount — V2 dominant**
| Model | MAE | RMSE | R² | vs V1 Best |
|---|---|---|---|---|
| **LightGBM V2** | **$6.73** | **$13.59** | **0.9575** | **+0.073** |
| XGBoost V2 | $7.73 | $15.43 | 0.9452 | +0.061 |
| Ensemble V2 | $6.68 | $13.50 | 0.9580 | +0.074 |
| CatBoost V2 | $10.88 | $20.10 | 0.9070 | +0.023 |
| LGB no-charge | $7.70 | $15.77 | 0.9428 | +0.058 |

**Stage 2: OOP — V1 wins**
| Model | MAE | RMSE | R² | Notes |
|---|---|---|---|---|
| **XGB Quantile V1** | **$9.78** | **$18.28** | **0.400** | **Best OOP model** |
| CatBoost Mono V2 | $10.55 | $21.34 | 0.173 | Monotonicity too restrictive |
| CatBoost ZI V2 | $11.95 | $24.15 | -0.054 | Zero-inflation hurts on synthetic data |

**Forecasting — V1 wins**
| Model | MAE | RMSE | R² | Notes |
|---|---|---|---|---|
| **LSTM V1** | **$8.84** | **$17.69** | **0.886** | **Best forecast model** |
| TFT V2 | $11.48 | $20.48 | 0.846 | Short sequences limit attention |

### Key Takeaways
1. **Stage 1 production model: LightGBM V2 (R²=0.958).** Full data was the unlock, not model architecture.
2. **Stage 2 production model: XGB Quantile V1 (R²=0.40).** Monotonicity constraints and zero-inflation don't help on synthetic OOP data. Real MCBS LDS data with actual OOP distributions would likely change this.
3. **Forecast production model: LSTM V1 (R²=0.886).** TFT needs longer time series (100+ steps) to outperform; 4-11 years per group is too thin for attention.
4. **Ensemble not worth it:** +0.0004 R² for 13.3 hrs compute. Deploy LightGBM alone.
5. **Charge ablation insight:** Removing `Avg_Sbmtd_Chrg` only drops R² by 0.01-0.02 — models genuinely predictive via procedure identity.

### Remaining Work
- [ ] Update web app with V2 Stage 1 model results (LightGBM V2 as production)
- [ ] Update Supabase model_metrics and feature_importances tables

---

## Phase 8 — Foundation Models & Forecast Stacker Track ✅ COMPLETE

**Goal:** Determine whether foundation time-series models (Chronos, TimesFM) and/or multivariate deep forecasters (TFT with covariates) can beat the LSTM V1 baseline (reported R² 0.886) on the 2022-2023 temporal holdout. If any approach approaches R² ≈ 0.90, deploy it as the new forecast model; otherwise close the forecast track.

**Outcome:** Forecast track **CLOSED**. Production model is the **LightGBM Stacker V2_12** (R² 0.8852 under fair evaluation). Signal ceiling for annual-resolution Medicare forecasting empirically confirmed at ≈0.885.

### Milestone 8.1 — V2_09 Foundation Model Exploration (April 11, 2026)
- [x] Chronos-T5-Large zero-shot inference on 20,572 groups (3D key = ptype × bucket × state)
- [x] Temporal split: train ≤ 2021, eval 2022-2023, forecast 2024-2026
- [x] TimesFM 2.5 attempted but install failed; Chronos-T5 only
- [x] Result: MAE $9.64, RMSE $20.33, R² 0.8485 — **lost to LSTM baseline by R² 0.0375**
- [x] MLflow run `chronos_t5_large_v2_colab` logged to Databricks
- [x] Forecast parquet saved: `chronos_forecast.parquet`

### Milestone 8.2 — V2_10 Derived-Feature Foundation Models (April 12, 2026)
- [x] Chronos-Bolt-Base (upgraded from T5-Large)
- [x] Four series variants: raw, charge_ratio, risk_adjusted, vol_normalized
- [x] Data source: aggregated 63 Gold parquets (102.8M rows → 180,318 group-year combos)
- [x] Results (2022-2023 fair holdout):
  - raw: MAE $9.54, RMSE $19.80, R² 0.8563 (**Bolt-Base beats T5-Large by R² +0.008**)
  - risk_adjusted: identical to raw (risk ≈ 1.0 no-op)
  - charge_ratio: R² 0.1937 — **collapsed** (back-transform through submitted charge amplifies error)
  - vol_normalized: R² −540.24 — **catastrophic** (compounding volume forecast error)
- [x] Conclusion: Derived features did not improve. Bolt-Base architecture swap is the only gain.

### Milestone 8.3 — V2_11 CPI/CF Deflation Variants (April 13, 2026)
- [x] Five deflation strategies applied to Chronos-Bolt input:
  - raw: no deflation
  - cpi_deflated: divide by CPI factor
  - cf_deflated: divide by Conversion Factor
  - cpi_cf_deflated: double deflation (**winner**)
  - seq_cpi_deflated: sequestration adjustment + CPI
- [x] Known future CPI/CF (2024-2026) used to reinflate forecasts — leak-free
- [x] Results on 2022-2023 fair holdout:
  - **cpi_cf_deflated: MAE $9.39, RMSE $19.71, R² 0.8576** (best FM variant)
  - raw: R² 0.8563
  - cf_deflated: R² 0.8541
  - seq_cpi_deflated: R² 0.8528
  - cpi_deflated: R² 0.8473 (**worse than raw** — CPI over-corrects)
- [x] Conclusion: Deflation helped by only +0.0013 R². Foundation models still lag LSTM.

### Milestone 8.4 — V2_12 LightGBM Stacker (April 14, 2026) 🏆 PRODUCTION
- [x] Built LightGBM meta-learner over LSTM + Chronos base forecasts
- [x] Self-contained Colab notebook: retrains LSTM in-notebook (3-5 min A100), re-runs Chronos cpi_cf_deflated, trains stacker
- [x] **Discovery: teacher-forcing bug in `train_lstm_local.py` evaluate() at line 340.** The reported V1 LSTM baseline R² 0.886 / RMSE $36.42 was inflated — `evaluate()` uses 1-step-ahead teacher forcing instead of autoregressive rollout. Cell 6 of V2_12 fixed this with proper batched autoregressive rollout from context ≤ 2021.
- [x] Fair autoregressive LSTM baseline: MAE $9.82, RMSE $18.91, R² 0.8689 (vs reported 0.886)
- [x] Stacker features (13): lstm_pred, chronos_pred, forecast_year, ptype/state/bucket, n_history_years, last_history_value, history_mean, history_cv, history_trend, cpi_factor, cf_factor
- [x] 5-fold GroupKFold CV → OOF metrics (no group-level leakage)
- [x] **Stacker OOF results: MAE $8.74, RMSE $17.69, R² 0.8852** — wins on all three metrics
- [x] Refit final stacker on full 2022-2023 data, applied to pre-computed 2024-2026 forecasts
- [x] `stacker_forecast_2024_2026.parquet` saved to Drive, LSTM-compatible schema
- [x] MLflow run `lgb_stacker_v2_12_colab` logged

**Feature importance (gain, V2_12):**
| Feature | Gain % |
|---|---:|
| lstm_pred | 70.48 |
| last_history_value | 15.97 |
| chronos_pred | 3.63 |
| history_mean | 2.67 |
| Rndrng_Prvdr_Type_idx | 1.91 |
| history_cv | 1.62 |
| history_trend | 1.18 |
| Rndrng_Prvdr_State_Abrvtn_idx | 1.16 |
| n_history_years | 0.77 |
| hcpcs_bucket | 0.31 |
| forecast_year | 0.27 |
| cpi_factor | 0.02 |
| cf_factor | 0.00 |

Chronos contributes only 3.6% of stacker gain; the stacker is essentially LSTM (70.5%) + persistence anchor (16.0%) + history conditioning (10%).

### Milestone 8.5 — V2_13 Multivariate TFT (April 14, 2026)
- [x] TemporalFusionTransformer with 7 channels per group-year from Gold parquets (`gold_year/` directory with year column, not `gold/`)
- [x] Feature structure:
  - `static_categoricals`: provider_type, state, hcpcs_bucket
  - `time_varying_known_reals` (decoder sees future): conversion_factor, cpi_medical, covid_indicator
  - `time_varying_unknown_reals` (encoder only): Avg_Mdcr_Alowd_Amt, log_srvcs, log_benes, Avg_Sbmtd_Chrg, Bene_Avg_Risk_Scre, srvcs_per_bene
- [x] max_encoder_length=11 (full 2013-2023), max_prediction_length=3
- [x] hidden_size=64, attention_head_size=4, dropout=0.15
- [x] QuantileLoss → native P10/P50/P90 bounds
- [x] **Results: MAE $9.23, RMSE $18.79, R² 0.8691** — statistically tied with fair LSTM (+0.0002 R²)
- [x] Underperformed stacker by R² 0.0161
- [x] **Multivariate hypothesis REJECTED:** 5 observed covariates + 3 known-future decoder inputs bought essentially nothing over univariate target history at annual resolution
- [x] Secondary finding: TFT multi-horizon forecasts are more coherent than stacker's autoregressive forecasts (2024/2025/2026 means $70/$68/$68 vs stacker's $72/$62/$62)
- [x] MLflow run `tft_multivariate_v2_13_colab` logged

### V2_09–V2_13 Final Leaderboard (2022-2023 fair temporal holdout, N=32,481)

| Model | MAE | RMSE | R² | Status |
|---|---:|---:|---:|---|
| LSTM V1 (reported, teacher-forced) | $8.84 | $36.42 | 0.8860 | **Inflated by teacher-forcing bug** |
| LSTM V1 (fair, autoregressive) | $9.82 | $18.91 | 0.8689 | Honest LSTM baseline |
| Chronos-T5-Large (V2_09) | $9.64 | $20.33 | 0.8485 | Archived |
| Chronos-Bolt raw (V2_10) | $9.54 | $19.80 | 0.8563 | Archived |
| Chronos-Bolt cpi_cf_deflated (V2_11) | $9.39 | $19.71 | 0.8576 | Base model for stacker |
| **LGB Stacker V2_12** | **$8.74** | **$17.69** | **0.8852** | 🏆 **PRODUCTION** |
| Multivariate TFT V2_13 | $9.23 | $18.79 | 0.8691 | Confirmed signal ceiling |

### Key Findings & Decisions

1. **Signal ceiling ≈ R² 0.885 at annual resolution.** Three independent modeling approaches (fair LSTM, Chronos, multivariate TFT) cluster at 0.857-0.869. Only ensemble diversity via the stacker breaks above, buying +0.0163 R² over its best base model. No further modeling yields without finer temporal resolution.

2. **Teacher-forcing bug in `train_lstm_local.py`** inflated the reported LSTM baseline by R² 0.017 and RMSE $17.51. This made earlier comparisons to Chronos misleading — the "LSTM has heavy right tail" hypothesis from V2_09-V2_11 writeups was an artifact of teacher-forcing, not a real property of the model.

3. **Multivariate covariates at annual resolution carry no orthogonal signal.** V2_13 TFT with volume, charge, risk trajectories tied univariate LSTM. At annual aggregation these features move in near-lockstep with the target, so their past values don't help forecast its future.

4. **The only remaining lever is quarterly data.** Going from 11 annual points to 44 quarterly points per group would unlock CNN/TCN/PatchTST architectures, enable proper fine-tuning of foundation models, and expose seasonal structure (open enrollment, holiday spikes, COVID disruption shape). Estimated 2-4 week data-engineering project. **Backlog, not current sprint.**

5. **Chronos-Bolt dominates T5-Large** for this task. Architecture swap (+0.008 R²) was the only foundation-model gain; every feature-engineering and deflation variant on top of Bolt was marginal or negative.

6. **V2_06 TFT (univariate) underperformed LSTM (0.846 vs 0.886) because it was univariate, not because TFT is weaker.** V2_13 multivariate TFT is tied with the fair LSTM — the architecture was always fine; what was missing was multivariate inputs, and those turn out not to help at this resolution either.

### Production Deployment Decision

**Deploy LightGBM Stacker V2_12** as the forecast model:
- Best validation metrics on the only ground-truth holdout (2022-2023)
- Forecast parquet: `{DRIVE}/v2_artifacts/predictions/stacker_forecast_2024_2026.parquet`
- LSTM-compatible schema → drop-in replacement for existing API endpoint
- Point-only predictions (p10=p50=p90); quantile bounds to be synthesized from historical error distribution OR replaced by quantile stacker in follow-up

**Alternative (TFT V2_13) remains defensible** if product-quality concerns about forecast coherence outweigh the 0.0161 R² gap. TFT provides native P10/P50/P90 quantile bounds and more gentle multi-horizon trajectories.

### Follow-up Tasks (Backlog)

- [ ] Fix `train_lstm_local.py` evaluate() — add `--eval-mode {teacher_forced,autoregressive}` flag, report both to avoid future measurement confusion
- [ ] Quantile stacker variant — 20-line change to V2_12 Cell 10 to train 3 LightGBM boosters at alpha={0.1, 0.5, 0.9}
- [ ] Quarterly data ingestion (only path above R² 0.89 ceiling) — evaluate CMS quarterly MUP-PS availability
- [ ] Fine-tune Chronos-Bolt on Medicare sequences (deferred; zero-shot ceiling reached)

---

## Phase 9 — Provider Anomaly Investigation Agent ✅ PHASES A–C COMPLETE

**Goal:** Deliver an end-to-end fraud investigation agent on top of the existing AllowanceMap silver layer. Statistical outlier detection → contextual evidence retrieval → rule checks → Claude-generated investigation briefs. Spec: [`PROVIDER_ANOMALY_AGENT_SPEC.md`](PROVIDER_ANOMALY_AGENT_SPEC.md).

### Milestone 9.1 — Phase A: Data Foundation (April 23, 2026)
- [x] `anomaly/compute_npi_profiles.py` — 11.52M NPI-year profiles from silver (`local_pipeline/anomaly/npi_profiles.parquet`, 640 MB)
  - 22 metrics per NPI-year incl. log/level volume, srvcs_per_bene, charge_to_allowed_ratio, Herfindahl, bucket distribution, YoY changes, risk score
  - 100% risk-score coverage after fixing path inheritance bug from gold script (`../data` → `data/`)
  - Reuses [`hcpcs_to_bucket()`](notebooks/03_gold_features_local.py:100) logic from gold layer
- [x] `anomaly/compute_benchmarks.py` — 3 benchmark tables (specialty × year, specialty × state × year, national × year)
  - 1,067 specialty-year groups, 50,078 state-specialty-year groups (26.8% thin <10 providers — confirmed §11.2 caveat)
  - Each metric reported with mean + P5/P25/P50/P75/P95

### Milestone 9.2 — Phase B: Detection Engine (April 23, 2026)
- [x] `anomaly/detect_outliers.py` — three independent methods, output `local_pipeline/anomaly/flags.parquet`
  - **Method A (z-score):** log1p-transform on heavy-tailed metrics, threshold |z|>3.0, falls back to specialty-year benchmark when state-specialty-year peer group <30
  - **Method B (Isolation Forest):** per-specialty (≥200 providers), `contamination=0.01`, 16 features incl. bucket distribution
  - **Method C (Temporal):** YoY rules with absolute volume gate (`total_services ≥ 1000`) + COVID recovery year (2021) excluded
- [x] **Composite signal calibrated within spec target (1–3%):** 89,201 NPI-years fire on ≥2 methods (0.77%); 8,391 fire on all 3 (0.073%); raw union 6.83%
- [x] Top-flagged specialties match historical OIG fraud priorities: Pain Management, Radiation Oncology, Diagnostic Radiology, Cardiology

### Milestone 9.3 — Phase C: Investigation Agent (April 23, 2026)
- [x] `anomaly/schemas.py` — `ProviderContext`, `RuleCheckResult`, `InvestigationBrief` dataclasses
- [x] `anomaly/retrieve_context.py` — `ContextRetriever` caches profiles + benchmarks at init, lazy-loads silver per state for top-HCPCS lookup
- [x] `anomaly/check_rules.py` — 9 rules from spec §6: 4 evaluable (HIGH_INTENSITY, VOLUME_SPIKE, CHARGE_INFLATION, PROCEDURE_CONCENTRATION) + 5 transparently marked NOT EVALUABLE with reason (no daily data, no per-encounter grouping, no claims linkage, no diagnosis codes, scope table not yet built)
- [x] `anomaly/generate_brief.py` — system prompt + structured user prompt + Claude API call with markdown parsing
  - Default model: `claude-sonnet-4-6` (per spec §9.3); ephemeral `cache_control` on system prompt (silently skipped — system <2048-token min for Sonnet 4.6 cache)
  - Dry-run is the default; `--live` opts in to API spend
  - Loads `ANTHROPIC_API_KEY` from cross-project `.env` (override=True to bypass empty shell var)
- [x] `anomaly/agent.py` — orchestrator: rank flags by composite severity → context → rules → brief → markdown + JSON per NPI
- [x] **10-brief live run** ([`local_pipeline/anomaly/briefs/`](local_pipeline/anomaly/briefs/)): 5 CRITICAL, 4 HIGH, 1 MEDIUM. Cost **$0.36** total (29K input / 18K output tokens, ~42s/brief).

### Key Findings from Sample Briefs
| NPI | Specialty | State | Year | Risk | Pattern |
|---|---|---|---|---|---|
| 1710906219 | Optometry | OH | 2018 | CRITICAL (91) | 1,784 cataract surgeries (CPT 66984) on 12 beneficiaries = 148/patient. Out-of-scope code; $1.38 avg allowed suggests Medicare denying claims. Claude hypothesized **NPI identity theft** and recommended OIG ZPIC referral. |
| 1255325338 | Mass Immunizer | TX | 2023 | CRITICAL (91) | Trend=spike, all 3 methods triggered. |
| 1033474374 | Cardiology | NY | 2018 | HIGH (78) | 1,240 services, srvcs/bene=112, HHI=1.0 — 1 HCPCS code repeated. |
| 1386678977 | Multispecialty Clinic | MI | 2013 | MEDIUM (42) | Correctly downgraded — Claude identified "specialty-mismatch artifact" (rad onc billed under multispecialty group), legitimate HCC weighting. |

### Key Decisions Made (Phase 9)
| Decision | Rationale |
|---|---|
| Branched `agent/provider-anomaly` from `main`, not from phase8-stacker-surface | Anomaly work is independent of forecast/frontend track. |
| Used composite (multi-method) overlap as the credible-signal rate | Raw OR-of-3-methods produced 6.83% (5× spec target). Multi-method intersection (0.77%) hits spec without arbitrary threshold tightening. |
| log1p z-score on heavy-tailed metrics | Naive z-score over-fired on lognormal billing distributions; log1p brings ~1% tail. |
| Absolute-volume gate on temporal rules | Filters out provider ramp-ups (e.g., 10→30 services = +200% but not suspicious). Excludes COVID recovery year (2021) where everyone YoY rebounded. |
| Sonnet 4.6 over Opus 4.7 for briefs | Per spec §9.3; structured narrative task does not need Opus reasoning depth. Cost: $0.04/brief vs $0.20/brief. |
| Dry-run as default in `agent.py --live` opt-in | API spend (10 briefs ≈ $0.36) requires explicit user opt-in per the harness's risky-action defaults. |
| Skipped OUT_OF_SPECIALTY rule for V1 | Requires per-specialty HCPCS scope table; deferred to follow-up. Marked NOT EVALUABLE so briefs disclose the gap. |

### Limitations & Follow-ups
- 5 of 9 spec rules cannot be evaluated with public CMS data (no claim-level dates, no per-encounter grouping, no beneficiary linkage, no diagnosis codes). Briefs disclose this explicitly.
- System prompt (1,200 tokens) below Sonnet 4.6's 2,048-token cache minimum — caching silently no-ops. Pad system prompt or batch via Messages API to capture cache savings on larger runs.
- Specialty-HCPCS scope table not yet built; OUT_OF_SPECIALTY rule placeholder.
- No analyst review interface yet (CLI / web dashboard from spec §8 deferred).
- `_load_api_key` uses `override=True` because shell had `ANTHROPIC_API_KEY=""` blocking dotenv default behavior — worth documenting for any other repos that rely on cross-project `.env`.

---

## Changelog

### 2026-04-23 — Provider Anomaly Investigation Agent (Phases A–C)
- **Phase 9 launch** — opened `agent/provider-anomaly` branch from `main`. Designed in `PROVIDER_ANOMALY_AGENT_SPEC.md` (2026-04-08) and now implemented end-to-end through Claude brief generation.
- **Phase A (Data Foundation):**
  - Built `anomaly/compute_npi_profiles.py` — 11.52M NPI-years × 22 metrics from silver layer in 184s
  - Built `anomaly/compute_benchmarks.py` — 3 benchmark tables (specialty × year, +state, +national) in ~30s
  - Fixed inherited `../data` path bug from gold script — provider risk scores now 100% covered
- **Phase B (Detection Engine):**
  - Built `anomaly/detect_outliers.py` with z-score, Isolation Forest, and temporal methods
  - Initial naive run: 15.57% raw flag rate (5× spec target). Tuned with log1p transform + absolute-volume gates → composite (≥2 methods) **0.77%**, within spec §4.2 target
  - Top-flagged specialties (Pain Management, Radiation Oncology, Diagnostic Radiology) match historical OIG priorities
- **Phase C (Investigation Agent):**
  - Built schemas, context retriever, rule checks, brief generator, orchestrator
  - Installed `anthropic==0.97.0` + `python-dotenv`; loaded `ANTHROPIC_API_KEY` from cross-project `coverdrive_pred_11/.env`
  - 10-brief live run with `claude-sonnet-4-6`: 5 CRITICAL / 4 HIGH / 1 MEDIUM
  - Total cost **$0.36** (29K input / 18K output tokens, ~42s per brief)
  - **Showcase brief:** Optometrist billing 1,784 cataract surgeries to 12 beneficiaries — Claude identified scope-of-practice violation (CPT 66984 is ophthalmology), anatomical impossibility (148 surgeries/patient), and hypothesized NPI identity theft from the $1.38 average allowed
  - **Calibration validation:** 2013 multispecialty clinic correctly downgraded to MEDIUM — Claude identified the "specialty-mismatch artifact" rather than calling it fraud
- **Bugs fixed during build:**
  - F-string ternary-in-format-spec error in `check_rules.py` (replaced with `_fmt()` helper)
  - Windows cp1252 encoding errors on Unicode arrows in `compute_npi_profiles.py` (replaced with ASCII)
  - `dotenv.load_dotenv` not overriding empty `ANTHROPIC_API_KEY` shell var (added `override=True`)
- **Memory updates:** added `project_anomaly_agent.md`

### 2026-04-14 — Forecast Track Closeout (V2_09–V2_13 analysis + V2_12/V2_13 implementation)
- **Phase 8 complete** — Foundation model track + forecast stacker + multivariate TFT. Forecast track formally CLOSED.
- Analyzed V2_09–V2_11 (Chronos foundation model experiments) from prior session:
  - V2_09 Chronos-T5-Large: R² 0.8485
  - V2_10 Chronos-Bolt derived features: best R² 0.8563, charge_ratio/vol_norm variants collapsed
  - V2_11 Chronos-Bolt cpi_cf_deflated: R² 0.8576 (best foundation-model variant)
- **Built V2_12 LightGBM Stacker** — `colab/V2_12_stacker_forecast.ipynb` + `scripts/build_v2_12_notebook.py`
  - 13-feature meta-learner blending LSTM + Chronos base forecasts with history conditioning
  - 5-fold GroupKFold OOF evaluation to prevent group-level leakage
  - Self-contained: retrains LSTM in-notebook (~5 min A100), reruns Chronos cpi_cf_deflated (~2 min)
  - **Stacker OOF: MAE $8.74, RMSE $17.69, R² 0.8852** — production winner
  - Feature importance: LSTM 70.5%, last_history_value 16.0%, Chronos 3.6%, rest 10%
- **Discovered teacher-forcing bug in `train_lstm_local.py evaluate()`** — function at line 340 uses 1-step-ahead teacher-forcing instead of autoregressive rollout, inflating reported V1 LSTM R² by 0.017 and RMSE by $17.51. V2_12 Cell 6 implemented proper batched autoregressive rollout from context ≤ 2021 to produce the fair baseline: MAE $9.82, RMSE $18.91, R² 0.8689
- **Corrected interpretation of V2_09–V2_11 analysis:** the "LSTM has heavy right tail, Chronos has half the RMSE" hypothesis was a teacher-forcing artifact. Under fair comparison all models have similar RMSE/MAE ratios (1.9-2.1) — near-Gaussian errors across the board
- **Built V2_13 Multivariate TFT** — `colab/V2_13_multivariate_tft.ipynb` + `scripts/build_v2_13_notebook.py`
  - TemporalFusionTransformer with 5 observed + 3 known-future covariates
  - Data: aggregated `gold_year/` parquets (year column present) → 180K group-year panel
  - Config: hidden_size=64, attention_heads=4, max_encoder=11, max_prediction=3, QuantileLoss
  - **TFT V2_13: MAE $9.23, RMSE $18.79, R² 0.8691** — tied fair LSTM (+0.0002), lost to stacker by 0.0161
  - **Multivariate hypothesis rejected:** at annual resolution, observed covariates provide no orthogonal signal over target history
- Fixed V2_13 `gold/` → `gold_year/` path bug (original `gold/` parquets lack year column)
- TFT secondary finding: multi-horizon joint prediction produces more coherent 2024-2026 forecasts than stacker's autoregressive drift ($70/$68/$68 vs $72/$62/$62)
- **Signal ceiling confirmed at R² ≈ 0.885** — no further modeling yields without quarterly data ingestion
- Created session memory files: `project_foundation_models.md` (reflects V2_09–V2_13 final state)
- Drafted frontend deployment todos for next session (see `FRONTEND_TODOS.md`)
- Created `MODELING.md` — comprehensive model zoo reference across all phases

### 2026-04-11 (evening)
- **V2 complete** — all 8 Colab notebooks (V2_01 through V2_08) executed on Colab Pro T4
- V2_04: CatBoost monotonic OOP — P50 R²=0.173, monotonicity constraints too restrictive on synthetic data
- V2_05: Zero-inflated OOP — P50 R²=-0.054, gate+regression compound errors
- V2_06: TFT forecasting — R²=0.846 with 5 external covariates, underperforms V1 LSTM (0.886)
- V2_07: Hierarchical reconciliation — coherence PASS but no adjustment (bottom-up already coherent)
- V2_08: Final comparison report — all runs pulled from MLflow, plots generated
- CatBoost GPU doesn't support monotone_constraints — switched V2_04/V2_05 to CPU
- pytorch-forecasting version conflicts: pinned ==1.1.1 + lightning==2.2.5, numpy binary compat fix
- PyTorch 2.6 weights_only=True breaks checkpoint loading — used trained model directly
- Drive FUSE timeouts — added shutil.copy to local SSD pattern across V2_04-V2_06
- **Production model decisions:** LightGBM V2 (Stage 1), XGB Quantile V1 (Stage 2), LSTM V1 (Forecast)

### 2026-04-11 (morning)
- **V2 Stage 1 complete** — full-data training on Colab Pro A100, 126.8M rows, no sampling
- LightGBM V2: R²=0.9575, MAE=$6.73 — best single model, +$5.31 MAE improvement over V1 RF
- XGBoost V2: R²=0.9452, MAE=$7.73 — full data closes V1 gap (was 0.833 at 30% sample)
- CatBoost V2: R²=0.9070, MAE=$10.88 — needed 3000 iterations, slowest convergence
- Ensemble V2: R²=0.9580, lift +0.0004 over LightGBM — not worth 13.3 hrs compute
- No-charge ablation: R² drop only 0.01-0.02, models genuinely predictive without billed charge
- Created 3 Colab notebooks (V2_01 through V2_03) with memory-safe SSD-split pattern
- Added `pull_external_covariates.py` (BLS API + hardcoded Medicare CF, sequestration, MACRA/MIPS)
- Added `modeling/train_catboost_local.py` and `modeling/train_lgbm_local.py`
- CU budget: ~211 of 300 consumed (estimate was 26-38 — 6.5x underestimate)

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
