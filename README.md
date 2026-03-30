# Medicare Provider Utilization & Cost Analysis

End-to-end pipeline that ingests CMS Medicare Physician & Practitioners data (2013–2023),
builds a Medallion architecture on Databricks, and trains regression models to predict
average Medicare payment per service (`Avg_Mdcr_Pymt_Amt`).

---

## Pipeline Overview

```
CMS API
  └─► pull_medicare_data.py        # Download yearly CSVs → data/
        └─► partition_medicare_data.py  # Split by STATE/PROVIDER_TYPE → partitioned_data/
              └─► csv_to_parquet.py     # Convert partitions to Parquet (in-place)

Databricks (Medallion)
  notebooks/01_bronze_ingest.py    # S3 Parquet → Bronze Delta table (raw strings)
  notebooks/02_silver_clean.py     # Type cast, null-drop, outlier removal → Silver Delta
  notebooks/03_gold_features.py    # Feature engineering, encoding → Gold Delta + Parquet export
  notebooks/04_eda.py              # Distributions, heatmaps, provider summaries

Local Modeling (logs to Databricks MLflow)
  modeling/train_glm.py            # Tweedie GLM baseline
  modeling/train_rf.py             # Random Forest + RandomizedSearchCV
  modeling/train_xgb.py           # XGBoost + early stopping
  modeling/compare_models.py       # Metrics table + paired t-test
```

---

## Directory Structure

```
medicare-provider-utilization-cost-analysis/
├── data/                   # Raw CSVs from CMS API (gitignored, ~GB scale)
├── partitioned_data/       # STATE/PROVIDER_TYPE Parquets (gitignored)
├── notebooks/              # Databricks notebooks — commit these
├── modeling/               # Local training scripts — commit these
├── pull_medicare_data.py
├── partition_medicare_data.py
├── csv_to_parquet.py
└── Project Proposal.docx
```

---

## Quickstart

### 1 · Pull raw data
```bash
# Pull a single year for testing
python pull_medicare_data.py --year 2023 --limit 50000

# Pull all years (slow — ~10M rows/year)
python pull_medicare_data.py
```

### 2 · Partition & convert
```bash
python partition_medicare_data.py --input-dir data --output-dir partitioned_data
python csv_to_parquet.py --dir partitioned_data
```

### 3 · Upload to S3 & run Databricks notebooks
Upload `partitioned_data/` to your S3 bucket, then run notebooks in order:
`01_bronze_ingest` → `02_silver_clean` → `03_gold_features` → `04_eda`

### 4 · Train models locally
```bash
# Point at the Gold parquet export from notebook 03
python modeling/train_glm.py --data /path/to/gold/features.parquet
python modeling/train_rf.py  --data /path/to/gold/features.parquet
python modeling/train_xgb.py --data /path/to/gold/features.parquet

# Compare results across MLflow experiments
python modeling/compare_models.py
```

---

## Data Source

CMS Medicare Physician & Practitioners dataset (2013–2023) via the
[CMS Data API](https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners).

Key columns used:

| Column | Description |
|---|---|
| `Rndrng_NPI` | Provider NPI identifier |
| `Rndrng_Prvdr_Type` | Provider specialty |
| `Rndrng_Prvdr_State_Abrvtn` | State abbreviation |
| `HCPCS_Cd` | Procedure/service code |
| `Tot_Benes` | Total unique beneficiaries |
| `Tot_Srvcs` | Total services rendered |
| `Avg_Sbmtd_Chrg` | Avg submitted charge amount |
| `Avg_Mdcr_Allo_Amt` | Avg Medicare allowed amount |
| `Avg_Mdcr_Pymt_Amt` | **Target** — Avg Medicare payment |
| `Avg_Mdcr_Stdzd_Amt` | Geographically standardized payment |

---

## Requirements

```
pandas
pyarrow
requests
scikit-learn
xgboost
mlflow
pyspark       # for Databricks notebooks
matplotlib
seaborn
scipy
```
