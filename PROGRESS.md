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

## Phase 3 — LSTM Time-Series Forecasting 🔜 NEXT

**Goal:** Build LSTM model on year-indexed sequences to forecast allowed amounts 2-3 years forward by specialty × HCPCS bucket × state.

### Planned Milestones
- [ ] 3.1 — PyTorch LSTM model architecture (sequence input → forecast output)
- [ ] 3.2 — Train on years 2013-2021, validate on 2022-2023
- [ ] 3.3 — Confidence-bounded forecast generation (2024-2026 projections)
- [ ] 3.4 — Specialty trend visualization charts
- [ ] 3.5 — MLflow logging alongside Stage 1 models
- [ ] 3.6 — Forecast output: `forecast_2024_2026.parquet`

### Input Data Ready
- `local_pipeline/lstm/sequences.parquet` — 23,672 groups
- 10,540 groups with all 11 years (2013-2023)
- 2,771 groups with < 3 years (will be filtered)

---

## Phase 4 — MCBS Integration (Planned)

**Goal:** Ingest Medicare Current Beneficiary Survey data, build parallel Bronze/Silver pipeline, create crosswalk.

### Planned Milestones
- [ ] 4.1 — MCBS PUF download script
- [ ] 4.2 — MCBS Bronze ingest notebook
- [ ] 4.3 — MCBS Silver cleaning notebook
- [ ] 4.4 — Region × specialty crosswalk table (Census regions map to our `CENSUS_REGIONS` dict)

---

## Phase 5 — Stage 2 OOP Model (Planned)

**Goal:** Train quantile regression on MCBS features to predict patient out-of-pocket costs (P10/P50/P90).

### Planned Milestones
- [ ] 5.1 — Stage 2 gold feature matrix (MCBS + Stage 1 predicted allowed amount)
- [ ] 5.2 — Quantile GBM training (P10, P50, P90 simultaneously)
- [ ] 5.3 — Evaluation on held-out MCBS year
- [ ] 5.4 — MLflow logging in unified experiment

---

## Phase 6 — Streamlit Portfolio App (Planned)

**Goal:** Patient-facing query interface combining both stages + LSTM forecast.

### Planned Milestones
- [ ] 6.1 — Streamlit app scaffolding (`app.py`, `model_loader.py`)
- [ ] 6.2 — Stage 1 + Stage 2 inference pipeline
- [ ] 6.3 — LSTM forecast display with uncertainty bands
- [ ] 6.4 — Deploy to Streamlit Community Cloud or HuggingFace Spaces

---

## Changelog

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
