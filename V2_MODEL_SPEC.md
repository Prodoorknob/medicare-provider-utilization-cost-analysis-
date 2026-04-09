# V2 Model Improvements Implementation Spec

## Context

Medicare Provider Utilization & Cost Analysis is at V1 — models trained on 30% sampled data with regional batching due to 24 GB WSL2 RAM constraint. V2 moves training to **Google Colab Pro (300 CU)** for full-data, no-compromise training. Databricks MLflow remains the experiment tracker.

**Problem:** V1 models underperform because (a) 70% of data is discarded, (b) regional batching fragments the learning, (c) `Avg_Sbmtd_Chrg` at 61.8% feature importance masks true predictive power, (d) OOP model lacks domain constraints, (e) LSTM forecasting has no external economic signals.

**Goal:** Train on all 126.8M rows, build an ensemble, add domain-informed OOP modeling, and upgrade forecasting with external covariates + TFT + hierarchical reconciliation.

---

## 1. Colab Notebook Structure

```
V2_01_stage1_full_training.ipynb         # XGB + CatBoost + LightGBM on 126.8M rows
V2_02_stage1_ablation.ipynb              # Same 3 models without Avg_Sbmtd_Chrg
V2_03_stage1_ensemble.ipynb              # 5-fold OOF stacking (Ridge meta-learner)
V2_04_stage2_catboost_monotonic.ipynb    # CatBoost OOP with monotonicity + non-crossing + CQR
V2_05_stage2_zero_inflated.ipynb         # Gate classifier + conditional regression
V2_06_forecast_tft.ipynb                 # Temporal Fusion Transformer with external covariates
V2_07_forecast_hierarchical.ipynb        # Hierarchical reconciliation (national→state→specialty)
V2_08_compare_v2.ipynb                   # V1 vs V2 comparison table
```

### Notebook Template (cell 1 of every notebook)

```python
import os, subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "mlflow>=2.12", "catboost>=1.2", "lightgbm>=4.3", "databricks-sdk>=0.20"])

from google.colab import drive, userdata
drive.mount('/content/drive')

DRIVE_ROOT = "/content/drive/MyDrive/AllowanceMap/V2"
GOLD_DIR   = f"{DRIVE_ROOT}/gold"
OOP_DATA   = f"{DRIVE_ROOT}/mcbs_synthetic/synthetic_oop.parquet"
SEQ_DATA   = f"{DRIVE_ROOT}/lstm/sequences.parquet"
ENCODERS   = f"{DRIVE_ROOT}/gold/label_encoders.json"
EXT_DIR    = f"{DRIVE_ROOT}/external"
ARTIFACTS  = f"{DRIVE_ROOT}/v2_artifacts"
os.makedirs(ARTIFACTS, exist_ok=True)

os.environ["DATABRICKS_HOST"]  = "https://dbc-d709cbb6-fe84.cloud.databricks.com"
os.environ["DATABRICKS_TOKEN"] = userdata.get("DATABRICKS_TOKEN")

import mlflow, requests
mlflow.set_tracking_uri("databricks")
resp = requests.get(
    f"{os.environ['DATABRICKS_HOST']}/api/2.0/preview/scim/v2/Me",
    headers={"Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}"},
    timeout=10,
)
resp.raise_for_status()
USER_HOME = f"/Users/{resp.json()['userName']}"
mlflow.set_experiment(f"{USER_HOME}/medicare_models")
print(f"MLflow: {USER_HOME}/medicare_models")
```

---

## 2. Data Pipeline — Local to Colab

### Google Drive Directory Structure

```
My Drive/AllowanceMap/V2/
├── gold/                            # Upload from local_pipeline/gold/ (1.9 GB, 63 parquets)
│   ├── CA.parquet ... WY.parquet
│   ├── label_encoders.json
│   └── hcpcs_target_enc.json
├── mcbs_synthetic/
│   └── synthetic_oop.parquet        # Upload from local_pipeline/mcbs_synthetic/ (10.3M rows)
├── lstm/
│   └── sequences.parquet            # Upload from local_pipeline/lstm/ (23,672 groups)
├── external/                        # User-sourced external covariates
│   ├── conversion_factors.csv       # Medicare CF by year (2013-2026)
│   ├── medical_cpi.csv              # BLS CUUR0000SAM annual averages
│   ├── sequestration_rates.csv      # CMS published rates by year
│   ├── macra_mips.csv               # QPP adjustment factors by year
│   └── covid_indicators.csv         # Binary flag + optional intensity
└── v2_artifacts/                    # Outputs from V2 notebooks
    ├── models/
    ├── predictions/
    └── plots/
```

### Upload Procedure

1. Zip `local_pipeline/gold/` → upload via Google Drive web UI
2. Upload `local_pipeline/mcbs_synthetic/synthetic_oop.parquet`
3. Upload `local_pipeline/lstm/sequences.parquet`
4. Create `external/` directory, populate with sourced CSV files (see Section 9)
5. Total upload: ~2.1 GB

### Full-Data Loading Function

```python
import glob, pandas as pd, numpy as np, pyarrow.parquet as pq

TARGET = "Avg_Mdcr_Alowd_Amt"
FEATURES = [
    "Rndrng_Prvdr_Type_idx", "Rndrng_Prvdr_State_Abrvtn_idx",
    "HCPCS_Cd_idx", "hcpcs_bucket", "place_of_srvc_flag",
    "Bene_Avg_Risk_Scre", "log_srvcs", "log_benes",
    "Avg_Sbmtd_Chrg", "srvcs_per_bene",
    "specialty_bucket", "pos_bucket", "hcpcs_target_enc",
]
CAT_FEATURE_NAMES = [
    "Rndrng_Prvdr_Type_idx", "Rndrng_Prvdr_State_Abrvtn_idx",
    "HCPCS_Cd_idx", "hcpcs_bucket", "place_of_srvc_flag",
    "specialty_bucket", "pos_bucket",
]

def load_all_gold(gold_dir, features, target=TARGET, sample=1.0):
    """Load all state parquets into a single array. No batching, no sampling by default."""
    load_cols = list(dict.fromkeys(features + [target, "year"]))
    parts = []
    for f in sorted(glob.glob(os.path.join(gold_dir, "*.parquet"))):
        avail = set(pq.read_schema(f).names)
        cols = [c for c in load_cols if c in avail]
        df = pd.read_parquet(f, columns=cols).dropna(subset=features + [target])
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    if sample < 1.0:
        df = df.sample(frac=sample, random_state=42)
    X    = df[features].astype("float64").values
    y    = np.log1p(df[target].astype("float64").values)
    year = df["year"].values if "year" in df.columns else None
    print(f"Loaded {len(df):,} rows, {len(features)} features")
    return X, y, year, df

def get_cat_indices(features):
    return [i for i, f in enumerate(features) if f in CAT_FEATURE_NAMES]
```

**Memory at full scale:** 126.8M × 14 cols × 8 bytes ≈ 14 GB. Colab High-RAM (83 GB) handles this easily.

---

## 3. Stage 1: Full-Data Training (V2_01)

### Runtime: T4 GPU + High-RAM | ~4-6 hrs | ~8-12 CU

Train XGBoost, CatBoost, LightGBM on all 126.8M rows. No sampling, no regional batching.

### Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### XGBoost

```python
import xgboost as xgb

XGB_PARAMS = {
    "objective":        "reg:squarederror",
    "learning_rate":    0.05,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "tree_method":      "hist",
    "device":           "cuda",
    "seed":             42,
    "max_bin":          256,
}

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURES)
dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=FEATURES)

booster = xgb.train(
    XGB_PARAMS, dtrain, num_boost_round=1000,
    evals=[(dtest, "val")], early_stopping_rounds=50, verbose_eval=100,
)

# MLflow
with mlflow.start_run(run_name="xgb_v2_full_colab"):
    mlflow.log_params({**XGB_PARAMS, "n_rounds": 1000, "early_stopping": 50,
        "source": "colab", "strategy": "full_dmatrix", "target_transform": "log1p",
        "sample_frac": 1.0, "split_strategy": "random", "n_features": 13,
        "ablation_avg_submitted_charge": False, "version": "v2"})
    y_pred = booster.predict(dtest)
    y_t, y_p = np.expm1(y_test), np.expm1(y_pred)
    mlflow.log_metrics({
        "test_mae": mean_absolute_error(y_t, y_p),
        "test_rmse": np.sqrt(mean_squared_error(y_t, y_p)),
        "test_r2": r2_score(y_t, y_p),
    })
    mlflow.log_dict(booster.get_score(importance_type="gain"), "feature_importances.json")
    mlflow.xgboost.log_model(booster, artifact_path="xgb_model")
```

### CatBoost

```python
from catboost import CatBoostRegressor, Pool

CB_PARAMS = {
    "loss_function": "RMSE", "learning_rate": 0.05, "depth": 6,
    "subsample": 0.8, "rsm": 0.8, "task_type": "GPU",
    "random_seed": 42, "verbose": 100, "allow_writing_files": False,
    "iterations": 1000,
}

cat_idx = get_cat_indices(FEATURES)
pool_train = Pool(X_train, label=y_train, cat_features=cat_idx, feature_names=FEATURES)
pool_test  = Pool(X_test,  label=y_test,  cat_features=cat_idx, feature_names=FEATURES)

model = CatBoostRegressor(**CB_PARAMS)
model.fit(pool_train, eval_set=pool_test, early_stopping_rounds=50, use_best_model=True)

# MLflow run name: "catboost_v2_full_colab"
# Same metrics pattern as XGBoost
```

### LightGBM

```python
import lightgbm as lgb

LGB_PARAMS = {
    "objective": "regression", "metric": "rmse", "learning_rate": 0.05,
    "num_leaves": 63, "max_depth": -1, "subsample": 0.8,
    "colsample_bytree": 0.8, "min_child_samples": 20,
    "device": "gpu", "seed": 42, "verbose": -1,
    "boosting_type": "gbdt", "data_sample_strategy": "goss",
}

cat_cols = [FEATURES[i] for i in get_cat_indices(FEATURES)]
ds_train = lgb.Dataset(X_train, label=y_train, feature_name=FEATURES, categorical_feature=cat_cols)
ds_test  = lgb.Dataset(X_test,  label=y_test,  feature_name=FEATURES, categorical_feature=cat_cols, reference=ds_train)

booster = lgb.train(LGB_PARAMS, ds_train, num_boost_round=1000,
    valid_sets=[ds_test], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])

# MLflow run name: "lgbm_v2_full_colab"
```

---

## 4. Stage 1: No-Charge Ablation (V2_02)

### Runtime: T4 GPU + High-RAM | ~3-5 hrs | ~6-10 CU

Same 3 models, same hyperparameters, but with `Avg_Sbmtd_Chrg` removed.

```python
FEATURES_NO_CHARGE = [f for f in FEATURES if f != "Avg_Sbmtd_Chrg"]
# 12 features instead of 13
```

**MLflow run names:**
- `xgb_v2_no_charge_colab`
- `catboost_v2_no_charge_colab`
- `lgbm_v2_no_charge_colab`

All runs log `"ablation_avg_submitted_charge": True`.

**Expected:** R² drops from ~0.88 → ~0.55-0.70. This quantifies honest predictive power without knowing the billed charge.

---

## 5. Stage 1: Ensemble Stacking (V2_03)

### Runtime: T4 GPU + High-RAM | ~6-8 hrs | ~12-16 CU

5-fold OOF stacking with Ridge meta-learner on 3 base models (XGBoost, CatBoost, LightGBM).

```python
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros((len(X_train_full), 3))  # columns: XGB, CatBoost, LightGBM
test_preds = np.zeros((len(X_test), 3))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_full)):
    X_tr, X_val = X_train_full[tr_idx], X_train_full[val_idx]
    y_tr, y_val = y_train_full[tr_idx], y_train_full[val_idx]

    # Train XGB fold → oof_preds[val_idx, 0], test_preds[:, 0] += pred / N_FOLDS
    # Train CatBoost fold → oof_preds[val_idx, 1], test_preds[:, 1] += pred / N_FOLDS
    # Train LightGBM fold → oof_preds[val_idx, 2], test_preds[:, 2] += pred / N_FOLDS

meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
meta.fit(oof_preds, y_train_full)
y_ensemble = meta.predict(test_preds)
```

**MLflow run name:** `ensemble_stack_v2_colab`

**Additional params:** `meta_alpha`, `meta_coefs`, `base_learners: "XGBoost+CatBoost+LightGBM"`, `n_folds: 5`

**Success criterion:** R² > best single model by ≥ 0.01

---

## 6. Stage 2: CatBoost Monotonic + Non-Crossing + CQR (V2_04)

### Runtime: T4 GPU | ~2-3 hrs | ~4-6 CU

All three OOP improvements in a single notebook.

### OOP Features and Target

```python
OOP_FEATURES = [
    "Avg_Mdcr_Alowd_Amt", "Bene_Avg_Risk_Scre", "Rndrng_Prvdr_Type_idx",
    "hcpcs_bucket", "place_of_srvc_flag", "census_region",
    "age", "sex", "income", "chronic_count", "dual_eligible", "has_supplemental",
]
OOP_TARGET = "per_service_oop"
OOP_CAT_IDX = [2, 3, 4, 5]  # specialty, bucket, pos, region
```

### 6A. CatBoost with Monotonicity Constraints

```python
MONOTONE = [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1]
#           ^allowed  ^risk ^spec ^bkt ^pos ^reg ^age ^sex ^inc ^chron ^dual ^supp
# +1: higher allowed amt → higher OOP (20% coinsurance)
# +1: higher income → higher OOP (less subsidy eligibility)
# +1: more chronic conditions → higher OOP (more services needed)
# -1: dual eligible → lower OOP (Medicaid covers cost-sharing)
# -1: has supplemental → lower OOP (Medigap pays deductibles/coinsurance)

for alpha, label in [(0.1, "p10"), (0.5, "p50"), (0.9, "p90")]:
    cb = CatBoostRegressor(
        loss_function=f"Quantile:alpha={alpha}",
        iterations=1000, learning_rate=0.05, depth=6,
        subsample=0.8, rsm=0.8, task_type="GPU",
        random_seed=42, verbose=0, allow_writing_files=False,
        monotone_constraints=MONOTONE,
    )
    cb.fit(pool_train, eval_set=pool_val, early_stopping_rounds=50, use_best_model=True)
```

### 6B. Non-Crossing Quantile Correction

```python
pred_p10_raw = cb_p10.predict(pool_test)
pred_p50     = cb_p50.predict(pool_test)
pred_p90_raw = cb_p90.predict(pool_test)

# Post-hoc sort (cheap, effective)
pred_p10 = np.minimum(pred_p10_raw, pred_p50)
pred_p90 = np.maximum(pred_p90_raw, pred_p50)

crossing_rate = float(np.mean(pred_p10_raw > pred_p50) + np.mean(pred_p90_raw < pred_p50))
mlflow.log_metric("crossing_rate_before_sort", crossing_rate)
```

### 6C. Conformalized Quantile Regression (CQR)

```python
# Split: 60% train / 20% calibration / 20% test
n = len(X_oop)
idx = np.random.RandomState(42).permutation(n)
n_tr, n_cal = int(n * 0.6), int(n * 0.2)
tr_idx, cal_idx, te_idx = idx[:n_tr], idx[n_tr:n_tr+n_cal], idx[n_tr+n_cal:]

# Train on train set (same monotonic CatBoost)
# Calibrate on calibration set
cal_scores = np.maximum(pred_p10_cal - y_cal, y_cal - pred_p90_cal)
q_hat = np.quantile(cal_scores, 0.80 * (1 + 1/len(cal_scores)))  # 80% target coverage

# Apply to test set
pred_lower = pred_p10_test - q_hat
pred_upper = pred_p90_test + q_hat
coverage = np.mean((y_test >= pred_lower) & (y_test <= pred_upper))
mlflow.log_metric("cqr_80pct_interval_coverage", coverage)  # should be ~0.80 ± 0.02
```

**MLflow run name:** `catboost_oop_monotonic_colab`

---

## 7. Stage 2: Zero-Inflated OOP Model (V2_05)

### Runtime: T4 GPU | ~1-2 hrs | ~2-4 CU

Two-stage model for the zero-heavy OOP distribution.

### Stage A: Gate Classifier — P(OOP = 0)

```python
from catboost import CatBoostClassifier

y_gate = (y_oop == 0).astype(int)
# Check zero fraction first:
zero_frac = y_gate.mean()
print(f"Zero-OOP fraction: {zero_frac:.1%}")  # expect ~30-40% for dual-eligible heavy data

gate = CatBoostClassifier(
    iterations=500, learning_rate=0.05, depth=6,
    task_type="GPU", random_seed=42, verbose=0,
    allow_writing_files=False, cat_features=OOP_CAT_IDX,
)
gate.fit(pool_train_gate, eval_set=pool_test_gate, early_stopping_rounds=30)
```

### Stage B: Conditional Regression — E[OOP | OOP > 0]

```python
nonzero_mask_train = y_oop_train > 0
nonzero_mask_test  = y_oop_test > 0

cb_nonzero = CatBoostRegressor(
    loss_function="Quantile:alpha=0.5",
    iterations=1000, learning_rate=0.05, depth=6,
    monotone_constraints=MONOTONE,
    task_type="GPU", random_seed=42, verbose=0,
    allow_writing_files=False, cat_features=OOP_CAT_IDX,
)
cb_nonzero.fit(Pool(X_train[nonzero_mask_train], y_oop_train[nonzero_mask_train], ...))
```

### Combined Prediction

```python
p_zero = gate.predict_proba(pool_test)[:, 1]
oop_if_pos = cb_nonzero.predict(pool_test)
y_pred_zi = (1 - p_zero) * np.maximum(oop_if_pos, 0)
```

**MLflow run name:** `catboost_oop_zero_inflated_colab`

**Metrics:** Same as V1 OOP (P10/P50/P90 MAE, coverage, pinball) + `zero_rate_train`, `zero_rate_test`, `gate_auc`, `gate_f1`

---

## 8. Forecasting: TFT with External Covariates (V2_06)

### Runtime: T4 GPU | ~2-3 hrs | ~4-6 CU

### Extra Installs

```python
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "pytorch-forecasting>=1.1", "pytorch-lightning>=2.2"])
```

### External Covariates Loading

User manually sources CSVs into `external/` directory. Expected schema:

```csv
# conversion_factors.csv
year,conversion_factor
2013,34.0230
...
2026,32.00

# medical_cpi.csv
year,cpi_medical
2013,425.1
...

# sequestration_rates.csv
year,sequestration_rate
2013,0.020
...

# covid_indicators.csv
year,covid_indicator
2013,0
...
2021,1
2022,0
...
```

### Data Preparation

Convert `sequences.parquet` (23,672 groups with year-ordered sequences) into flat TFT-compatible DataFrame:

```python
import pytorch_forecasting as ptf
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

# Load external covariates
cf_df  = pd.read_csv(f"{EXT_DIR}/conversion_factors.csv")
cpi_df = pd.read_csv(f"{EXT_DIR}/medical_cpi.csv")
seq_df = pd.read_csv(f"{EXT_DIR}/sequestration_rates.csv")
covid_df = pd.read_csv(f"{EXT_DIR}/covid_indicators.csv")
ext = cf_df.merge(cpi_df, on="year").merge(seq_df, on="year").merge(covid_df, on="year")

# Flatten sequences
ts_rows = []
for _, row in sequences_df.iterrows():
    gid = f"{int(row['Rndrng_Prvdr_Type_idx'])}_{int(row['hcpcs_bucket'])}_{int(row['Rndrng_Prvdr_State_Abrvtn_idx'])}"
    for yr, val in zip(row["years"], row["target_seq"]):
        ext_row = ext[ext["year"] == yr].iloc[0] if yr in ext["year"].values else {}
        ts_rows.append({
            "group_id": gid,
            "time_idx": yr - 2013,
            "year": yr,
            "target": val,
            "conversion_factor": ext_row.get("conversion_factor", 33.0),
            "cpi_medical": ext_row.get("cpi_medical", 500.0),
            "sequestration_rate": ext_row.get("sequestration_rate", 0.02),
            "covid_indicator": ext_row.get("covid_indicator", 0),
            "provider_type": str(int(row["Rndrng_Prvdr_Type_idx"])),
            "state": str(int(row["Rndrng_Prvdr_State_Abrvtn_idx"])),
            "hcpcs_bucket": str(int(row["hcpcs_bucket"])),
        })
ts_df = pd.DataFrame(ts_rows)
```

### TFT Configuration

```python
max_encoder_length = 8    # up to 8 years of history
max_prediction_length = 3  # forecast 2024-2026

training_cutoff = ts_df["time_idx"].max() - max_prediction_length  # train up to 2021

training = TimeSeriesDataSet(
    ts_df[ts_df["time_idx"] <= training_cutoff],
    time_idx="time_idx",
    target="target",
    group_ids=["group_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["provider_type", "state", "hcpcs_bucket"],
    time_varying_known_reals=["conversion_factor", "cpi_medical",
                               "sequestration_rate", "covid_indicator"],
    time_varying_unknown_reals=["target"],
    target_normalizer=GroupNormalizer(groups=["group_id"]),
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=1e-3,
    hidden_size=32,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=16,
    output_size=7,  # quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    loss=ptf.metrics.QuantileLoss(),
    reduce_on_plateau_patience=5,
)

trainer = pl.Trainer(
    max_epochs=50, accelerator="gpu", gradient_clip_val=0.1,
    callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min")],
)
trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
```

**MLflow run name:** `tft_v2_colab`

**Advantages over V1 LSTM:**
- Native quantile output (no MC Dropout needed)
- Variable selection networks auto-weight covariates
- Interpretable attention weights
- External covariates inform trend direction for 2024-2026

---

## 9. Forecasting: Hierarchical Reconciliation (V2_07)

### Runtime: High-RAM CPU | ~1-2 hrs | ~1-2 CU

### Extra Installs

```python
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "hierarchicalforecast>=0.4"])
```

### Hierarchy

```
Total (national, 1 series)
├── State_1 (CA, 1 series)
│   ├── Specialty_A in CA
│   └── Specialty_B in CA
├── State_2 (TX, 1 series)
│   └── ...
└── ...
```

Bottom-level = specialty × state (~23K series). State = sum of specialties. National = sum of states.

### Reconciliation Method

MinTrace with shrinkage (`mint_shrink`) — minimizes trace of reconciled forecast error covariance.

```python
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import MinTraceShrink

# S matrix: encodes hierarchy (national → state → specialty×state)
# Base forecasts: TFT predictions at all levels
reconciliation = HierarchicalReconciliation(reconcilers=[MinTraceShrink()])
reconciled = reconciliation.reconcile(Y_hat_df=base_forecasts, S=S_matrix, tags=tags)
```

**MLflow run name:** `hierarchical_reconciled_v2_colab`

**Success criterion:** National forecast = sum of state forecasts (coherence). Often also improves accuracy at all levels.

---

## 10. External Covariate Data Sources

User manually downloads these and places CSVs in `My Drive/AllowanceMap/V2/external/`.

| Covariate | Source URL | Format | Notes |
|---|---|---|---|
| **Medicare Conversion Factor** | https://www.cms.gov/medicare/payment/fee-schedules/physician/pfs-relative-value-files | Look in annual Federal Register final rules | 1 value per year (2013-2026) |
| **Medical CPI** | https://data.bls.gov/timeseries/CUUR0000SAM | BLS Data Finder, select "Annual" average | Series CUUR0000SAM |
| **Sequestration Rate** | https://www.cms.gov/medicare/payment/claims-based-sequestration | CMS fact sheets by year | 2% standard, 0% in 2020-2021 |
| **MACRA/MIPS Adjustments** | https://qpp.cms.gov/resources/all-resources | QPP final rule archives | Distribution of payment adjustments |
| **COVID Indicators** | https://healthdata.gov/ | Binary: 1 for 2020-2021, 0 otherwise | Optional: CDC excess utilization data |

**CSV schemas:** Each file needs `year` column (int, 2013-2026) + one value column. For 2024-2026 projections, use CMS final rule projections or CBO estimates.

---

## 11. CU Budget Summary

| Notebook | Runtime | Hours | CU/hr | Est. CU |
|---|---|---|---|---|
| V2_01 (3 full models) | T4 + High-RAM | 4-6 | 1.96 | 8-12 |
| V2_02 (3 ablation) | T4 + High-RAM | 3-5 | 1.96 | 6-10 |
| V2_03 (ensemble 5-fold) | T4 + High-RAM | 6-8 | 1.96 | 12-16 |
| V2_04 (monotonic + CQR) | T4 GPU | 2-3 | 1.96 | 4-6 |
| V2_05 (zero-inflated) | T4 GPU | 1-2 | 1.96 | 2-4 |
| V2_06 (TFT) | T4 GPU | 2-3 | 1.96 | 4-6 |
| V2_07 (hierarchical) | High-RAM CPU | 1-2 | 0.72 | 1-2 |
| V2_08 (comparison) | High-RAM CPU | 0.5 | 0.72 | <1 |
| **Debug/rerun buffer** | Mixed | ~5 | ~2 | ~10 |
| **TOTAL** | | | | **~48-67 CU** |

**Budget: 300 CU. Remaining after full pipeline: ~230+ CU** for hyperparameter sweeps, ablation experiments, or reruns.

---

## 12. Execution Order — Dependency Graph

```
[Upload data to Google Drive]  ← one-time prerequisite
         |
         v
      V2_01 (Stage 1: full-data XGB, CatBoost, LightGBM)
         |
    +----+----+
    |         |
    v         v
  V2_02     V2_03                V2_04, V2_05 (Stage 2 — independent of Stage 1)
(ablation) (ensemble)                |
                                     v
                              [Stage 2 complete]
         |
         v
   [Stage 1 complete]
         |
    +----+----+
    |         |
    v         v
  V2_06     V2_07 (depends on V2_06 forecasts)
  (TFT)   (hierarchical)
              |
              v
           V2_08 (comparison — run last)
```

**Parallelizable:** V2_02 + V2_03 after V2_01. V2_04 + V2_05 anytime (independent). V2_07 after V2_06.

---

## 13. MLflow Run Names — Complete Reference

| Run Name | Model | Stage | Notebook |
|---|---|---|---|
| `xgb_v2_full_colab` | XGBoost 126.8M rows | Stage 1 | V2_01 |
| `catboost_v2_full_colab` | CatBoost 126.8M rows | Stage 1 | V2_01 |
| `lgbm_v2_full_colab` | LightGBM 126.8M rows | Stage 1 | V2_01 |
| `xgb_v2_no_charge_colab` | XGBoost no Avg_Sbmtd_Chrg | Stage 1 | V2_02 |
| `catboost_v2_no_charge_colab` | CatBoost no Avg_Sbmtd_Chrg | Stage 1 | V2_02 |
| `lgbm_v2_no_charge_colab` | LightGBM no Avg_Sbmtd_Chrg | Stage 1 | V2_02 |
| `ensemble_stack_v2_colab` | Ridge stacking (3 boosters) | Stage 1 | V2_03 |
| `catboost_oop_monotonic_colab` | CatBoost monotonic + CQR | Stage 2 | V2_04 |
| `catboost_oop_zero_inflated_colab` | Zero-inflated gate + regression | Stage 2 | V2_05 |
| `tft_v2_colab` | Temporal Fusion Transformer | Forecast | V2_06 |
| `hierarchical_reconciled_v2_colab` | MinTrace reconciliation | Forecast | V2_07 |

---

## 14. Success Criteria

### Stage 1

| Variant | V1 Baseline | V2 Target |
|---|---|---|
| XGBoost full | R²=0.833 (0.3 sample, batch) | R² ≥ 0.88 |
| CatBoost full | Not trained | R² ≥ 0.88 |
| LightGBM full | Not trained | R² ≥ 0.88 |
| Any no-charge | N/A | R² ≥ 0.55 (quantifies honest power) |
| Ensemble | N/A | R² ≥ best_single + 0.01 |

### Stage 2

| Variant | V1 Baseline | V2 Target |
|---|---|---|
| CatBoost monotonic P50 | R²=0.40 | R² ≥ 0.42 |
| Non-crossing | Crossing possible | Crossing rate < 1% |
| CQR 80% interval | No guarantee | Coverage = 80% ± 2% |
| Zero-inflated | Not attempted | P50 R² ≥ 0.45 |

### Forecasting

| Variant | V1 Baseline | V2 Target |
|---|---|---|
| TFT + covariates | LSTM R²=0.886 | R² ≥ 0.89 (group-level) |
| Hierarchical | Not attempted | Sum-coherent across levels |

---

## 15. compare_models_local.py Updates

After V2 training, update `modeling/compare_models_local.py` to include V2 run names:

```python
MODEL_RUN_NAMES = {
    # V1
    "GLM":             "glm_sgd_local",
    "RF (V1)":         "rf_randomized_search_local",
    "XGB (V1)":        "xgb_extmem_local",
    "LSTM (V1)":       "lstm_local",
    # V2
    "XGB (V2)":        "xgb_v2_full_colab",
    "CatBoost (V2)":   "catboost_v2_full_colab",
    "LightGBM (V2)":   "lgbm_v2_full_colab",
    "Ensemble (V2)":   "ensemble_stack_v2_colab",
    "TFT (V2)":        "tft_v2_colab",
}

ABLATION_RUN_NAMES = {
    "XGB no-charge":     "xgb_v2_no_charge_colab",
    "CatBoost no-charge":"catboost_v2_no_charge_colab",
    "LightGBM no-charge":"lgbm_v2_no_charge_colab",
}

OOP_RUN_NAMES = {
    "XGB_Quantile (V1)":       "xgb_quantile_oop_local",
    "CatBoost_Mono (V2)":      "catboost_oop_monotonic_colab",
    "CatBoost_ZI (V2)":        "catboost_oop_zero_inflated_colab",
}
```

---

## 16. Pip Dependencies Summary

| Package | Version | Used In |
|---|---|---|
| `mlflow` | ≥ 2.12 | All notebooks |
| `databricks-sdk` | ≥ 0.20 | All notebooks |
| `catboost` | ≥ 1.2 | V2_01-05 |
| `lightgbm` | ≥ 4.3 | V2_01-03 |
| `xgboost` | (Colab default) | V2_01-03 |
| `pytorch-forecasting` | ≥ 1.1 | V2_06 |
| `pytorch-lightning` | ≥ 2.2 | V2_06 |
| `hierarchicalforecast` | ≥ 0.4 | V2_07 |
| `mapie` | ≥ 0.8 | V2_04 (optional CQR alt) |

---

## 17. Output Artifacts

### Models → Google Drive + MLflow

| Artifact | MLflow Path | Drive Path |
|---|---|---|
| XGB V2 full | `xgb_model` | `v2_artifacts/models/xgb_v2_full.json` |
| CatBoost V2 full | `catboost_model` | `v2_artifacts/models/catboost_v2_full.cbm` |
| LightGBM V2 full | `lgbm_model` | `v2_artifacts/models/lgbm_v2_full.txt` |
| Ensemble Ridge meta | `ridge_meta` | `v2_artifacts/models/ensemble_ridge.pkl` |
| OOP monotonic P10/P50/P90 | `catboost_oop_{q}` | `v2_artifacts/models/oop_mono_{q}.cbm` |
| OOP gate classifier | `gate_model` | `v2_artifacts/models/oop_gate.cbm` |
| OOP ZI regression | `regression_model` | `v2_artifacts/models/oop_zi_reg.cbm` |
| TFT checkpoint | `tft_model` | `v2_artifacts/models/tft_best.ckpt` |

### Predictions → Google Drive

| File | Path |
|---|---|
| Ensemble OOF predictions | `v2_artifacts/predictions/ensemble_oof.parquet` |
| CQR conformity scores | `v2_artifacts/predictions/cqr_scores.parquet` |
| TFT forecast 2024-2026 | `v2_artifacts/predictions/tft_forecast.parquet` |
| Hierarchical reconciled | `v2_artifacts/predictions/hierarchical_reconciled.parquet` |

### Plots → Google Drive + MLflow

| Plot | Path |
|---|---|
| V1 vs V2 comparison | `v2_artifacts/plots/v1_v2_comparison.png` |
| Charge ablation impact | `v2_artifacts/plots/charge_ablation.png` |
| Ensemble weights | `v2_artifacts/plots/ensemble_weights.png` |
| OOP monotonicity check | `v2_artifacts/plots/oop_monotonicity.png` |
| CQR coverage calibration | `v2_artifacts/plots/cqr_coverage.png` |
| TFT variable importance | `v2_artifacts/plots/tft_importance.png` |
| TFT attention patterns | `v2_artifacts/plots/tft_attention.png` |

---

## 18. Verification

1. **After V2_01:** Check MLflow for 3 runs with `version=v2`, compare test_r2 against V1 baselines
2. **After V2_02:** Compare no-charge R² vs with-charge R² — delta quantifies charge dependency
3. **After V2_03:** Ensemble R² must exceed best single model
4. **After V2_04:** Check `crossing_rate_before_sort < 0.01`, `cqr_80pct_interval_coverage ∈ [0.78, 0.82]`
5. **After V2_05:** Check `gate_auc > 0.80`, `zero_rate_test` matches expected distribution
6. **After V2_06:** Compare TFT val loss vs V1 LSTM val loss (group-level R²)
7. **After V2_07:** Verify sum-coherence: national forecast ≈ sum(state forecasts) within rounding
8. **After V2_08:** Run `compare_models_local.py` locally — all V2 runs should appear in comparison table
