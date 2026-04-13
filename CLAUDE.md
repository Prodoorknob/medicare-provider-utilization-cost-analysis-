# CLAUDE.md

## Project Overview

Medicare Provider Utilization & Cost Analysis — end-to-end data pipeline and ML project predicting average Medicare **allowed amount** per service (`Avg_Mdcr_Alowd_Amt`) using CMS Physician & Practitioners data (2013-2023). Stage 1 of a two-stage pipeline; Stage 2 (future) predicts patient out-of-pocket costs using MCBS data.

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

**Current status:** Phases 3-5 code complete. LSTM training pending (GPU). MCBS pipeline done. OOP model ready to train.

## Notes

- `.env` contains Databricks credentials — never commit or expose
- `local_pipeline/` and `data/` / `partitioned_data/` are gitignored
- Databricks notebooks use `# COMMAND ----------` cell separators
- XGBoost auto-detects CUDA; RF batch mode is sklearn-only (cuML lacks warm_start)
- `--sample` flag on training scripts controls fraction of data loaded (default 0.3)
- Gold parquets use float32 for RF (halves VRAM) and float64 for XGBoost/GLM
- Year column must be re-injected by re-running `partition_medicare_data.py` from raw CSVs if missing
- Provider risk scores (`Bene_Avg_Risk_Scre`) come from a separate CMS dataset; missing scores imputed with global median (~1.0)
