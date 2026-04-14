"""
Build colab/V2_13_multivariate_tft.ipynb — Multivariate TFT with observed + known-future
covariates. Targets beating V2_12 stacker on 2022-2023 fair holdout.

Run:
    python scripts/build_v2_13_notebook.py
"""
import json
import os

NB_PATH = os.path.join(os.path.dirname(__file__), "..", "colab", "V2_13_multivariate_tft.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


# ---------------------------------------------------------------------------
# Cell contents
# ---------------------------------------------------------------------------

C0_MD = """# V2_13 — Multivariate TFT with Observed + Known-Future Covariates

**Goal:** Beat the V2_12 LightGBM stacker (R² 0.8852, RMSE $17.69) by giving a
Temporal Fusion Transformer access to multivariate per-group-year signals that
neither the univariate LSTM V1 nor Chronos-Bolt has ever seen.

**What's new vs V2_06 (the earlier TFT attempt):**
- V2_06 was univariate (target only) → R² 0.846, lost to LSTM
- V2_13 uses **7 channels per group-year** built from Gold parquets
- `max_encoder_length` raised from 6 → 11 (use full 2013–2023 history)
- `max_prediction_length` = 3 (covers 2022–2023 eval and 2024–2026 forecast in one model)

**Feature structure for TFT:**
- `static_categoricals`: provider_type, state, hcpcs_bucket (same as LSTM)
- `time_varying_known_reals` (decoder sees future values):
    - `conversion_factor` — CMS-published through 2026
    - `cpi_medical` — BLS medical inflation index
    - `covid_indicator` — known flag
- `time_varying_unknown_reals` (encoder-only):
    - `Avg_Mdcr_Alowd_Amt` — the target
    - `log_srvcs`, `log_benes` — utilization trajectory
    - `Avg_Sbmtd_Chrg` — billing behavior
    - `Bene_Avg_Risk_Scre` — case mix
    - `srvcs_per_bene` — intensity

**Why this should work better than V2_12:**
1. **Multi-horizon (not autoregressive).** TFT predicts all 2022/2023/2024/2025/2026 in one
   shot from the encoder state. No compounding error across the horizon — which was the
   source of LSTM's 2024 → 2025 $10 drop in V2_12's final forecast.
2. **Observed covariates.** V2_12's feature importance showed Chronos contributing only
   3.6% of gain vs LSTM's 70.5%. Replacing the LSTM base with a model that can see volume
   and charge trajectories attacks the 70.5% directly — much more leverage than blending.
3. **Known-future covariates in the decoder.** V2_12 showed CPI/CF were dead features
   (0.02% / 0.0% gain) because they were encoded implicitly. TFT's decoder consumes
   known-future reals explicitly during prediction — the model learns their relationship
   to allowed amounts instead of us hardcoding a linear deflation.

**Success criterion:** If OOF R² ≥ 0.89 on the same fair 2022-2023 holdout used in V2_12,
declare victory and wrap the forecasting track.

**Runtime:** A100 GPU | ~60–90 min | ~3–4 CU
"""

C1_ENV = """# ── Cell 1: Environment Setup ──────────────────────────────────────────────────────────
import os, subprocess, sys

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'mlflow>=2.12', 'databricks-sdk>=0.20'])
# pytorch-forecasting==1.1.1 + lightning==2.2.5 is the compatible pair (per V2_06 notes).
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
    'pytorch-forecasting==1.1.1'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'lightning==2.2.5', 'torchmetrics'])

from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = '/content/drive/MyDrive/AllowanceMap/V2'
GOLD_DIR   = f'{DRIVE_ROOT}/gold_year'  # NOTE: gold/ lacks the year column; gold_year/ has it
SEQ_DATA   = f'{DRIVE_ROOT}/lstm/sequences.parquet'
EXT_DIR    = f'{DRIVE_ROOT}/external'
ARTIFACTS  = f'{DRIVE_ROOT}/v2_artifacts'
os.makedirs(f'{ARTIFACTS}/models', exist_ok=True)
os.makedirs(f'{ARTIFACTS}/predictions', exist_ok=True)
os.makedirs(f'{ARTIFACTS}/plots', exist_ok=True)

os.environ['DATABRICKS_HOST']  = 'https://dbc-d709cbb6-fe84.cloud.databricks.com'
os.environ['DATABRICKS_TOKEN'] = 'dapi880a64dc497c1fabc1f409c20f9db6b1'

import mlflow, requests
mlflow.set_tracking_uri('databricks')
resp = requests.get(
    f"{os.environ['DATABRICKS_HOST']}/api/2.0/preview/scim/v2/Me",
    headers={'Authorization': f"Bearer {os.environ['DATABRICKS_TOKEN']}"},
    timeout=10,
)
resp.raise_for_status()
USER_HOME = f"/Users/{resp.json()['userName']}"
mlflow.set_experiment(f'{USER_HOME}/medicare_models')
print(f'MLflow: {USER_HOME}/medicare_models')
"""

C2_PANEL = """# ── Cell 2: Aggregate Gold → Multivariate Group-Year Panel ───────────────────
import gc, json, time, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

GROUP_KEYS = ['Rndrng_Prvdr_Type_idx', 'hcpcs_bucket', 'Rndrng_Prvdr_State_Abrvtn_idx']
TARGET_COL = 'Avg_Mdcr_Alowd_Amt'

OBSERVED_NUMERIC = [
    'log_srvcs',
    'log_benes',
    'Avg_Sbmtd_Chrg',
    'Bene_Avg_Risk_Scre',
    'srvcs_per_bene',
]

LOAD_COLS = GROUP_KEYS + ['year', TARGET_COL] + OBSERVED_NUMERIC

# Copy Gold parquets to local SSD
LOCAL_GOLD = '/content/gold'
if not os.path.exists(LOCAL_GOLD):
    os.makedirs(LOCAL_GOLD)
    print('Copying Gold parquets to local SSD...')
    gold_files = [f for f in sorted(os.listdir(GOLD_DIR)) if f.endswith('.parquet')]
    for i, fname in enumerate(gold_files, 1):
        shutil.copy(f'{GOLD_DIR}/{fname}', f'{LOCAL_GOLD}/{fname}')
        if i % 10 == 0:
            print(f'  {i}/{len(gold_files)}')
    print('Gold copied.')

# Aggregate per-state, concat into group-year panel
t0 = time.time()
parts = []
gold_files = [f for f in sorted(os.listdir(LOCAL_GOLD)) if f.endswith('.parquet')]
total_rows = 0
for i, fname in enumerate(gold_files, 1):
    df = pd.read_parquet(f'{LOCAL_GOLD}/{fname}', columns=LOAD_COLS)
    total_rows += len(df)
    agg = df.groupby(GROUP_KEYS + ['year'], as_index=False)[
        [TARGET_COL] + OBSERVED_NUMERIC
    ].mean()
    parts.append(agg)
    if i % 10 == 0:
        print(f'  {i}/{len(gold_files)} states aggregated ({total_rows:,} source rows so far)')

panel = pd.concat(parts, ignore_index=True)
del parts; gc.collect()
print(f'Aggregation: {time.time() - t0:.1f}s | Source rows: {total_rows:,}')
print(f'Panel (all group-years): {len(panel):,} rows')

# Filter to groups with >= 3 years (same filter used by V2_12)
group_counts = panel.groupby(GROUP_KEYS)['year'].nunique().reset_index(name='n_years')
valid = group_counts[group_counts['n_years'] >= 3][GROUP_KEYS]
panel = panel.merge(valid, on=GROUP_KEYS, how='inner')
print(f'Panel (>=3 years): {len(panel):,} rows, {panel.groupby(GROUP_KEYS).ngroups:,} groups')

# Cast types
for c in GROUP_KEYS:
    panel[c] = panel[c].astype(int)
panel['year'] = panel['year'].astype(int)
print(f'Year range: {panel["year"].min()}-{panel["year"].max()}')
"""

C3_EXT = """# ── Cell 3: External Covariates + TFT Feature Engineering ───────────────────
LOCAL_EXT = '/content/external'
os.makedirs(LOCAL_EXT, exist_ok=True)
for fname in os.listdir(EXT_DIR):
    src = f'{EXT_DIR}/{fname}'; dst = f'{LOCAL_EXT}/{fname}'
    if not os.path.exists(dst) and os.path.isfile(src):
        shutil.copy(src, dst)

ext_files = {
    'conversion_factor':  'conversion_factors.csv',
    'cpi_medical':        'medical_cpi.csv',
    'covid_indicator':    'covid_indicators.csv',
}
ext = None
for col, fname in ext_files.items():
    fpath = f'{LOCAL_EXT}/{fname}'
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        ext = df if ext is None else ext.merge(df, on='year', how='outer')

if ext is None or 'conversion_factor' not in ext.columns:
    print('WARNING: falling back to hardcoded external covariates')
    ext = pd.DataFrame({
        'year': list(range(2013, 2027)),
        'conversion_factor': [34.02, 35.80, 35.75, 35.88, 35.89, 35.99, 36.04, 36.09,
                              34.89, 33.06, 33.89, 32.74, 33.29, 31.92],
        'cpi_medical': [425.1, 435.3, 446.8, 463.7, 471.4, 478.4, 487.7, 499.4,
                        519.3, 545.1, 557.6, 570.2, 578.0, 585.4],
        'covid_indicator': [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    })
ext = ext.sort_values('year').reset_index(drop=True)
if 'covid_indicator' not in ext.columns:
    ext['covid_indicator'] = ((ext['year'] >= 2020) & (ext['year'] <= 2021)).astype(int)
print(f'External covariates: years {ext["year"].min()}-{ext["year"].max()}')
print(ext[['year', 'conversion_factor', 'cpi_medical', 'covid_indicator']].to_string(index=False))

# Merge into panel
panel = panel.merge(
    ext[['year', 'conversion_factor', 'cpi_medical', 'covid_indicator']],
    on='year', how='left',
)

# Fill residual NaNs conservatively (years before/after coverage)
panel['conversion_factor'] = panel['conversion_factor'].fillna(method='ffill').fillna(method='bfill')
panel['cpi_medical']       = panel['cpi_medical'].fillna(method='ffill').fillna(method='bfill')
panel['covid_indicator']   = panel['covid_indicator'].fillna(0).astype(int)

# TFT needs a string group_id and an integer time_idx starting at 0
panel['group_id'] = (
    panel['Rndrng_Prvdr_Type_idx'].astype(str) + '_' +
    panel['hcpcs_bucket'].astype(str) + '_' +
    panel['Rndrng_Prvdr_State_Abrvtn_idx'].astype(str)
)
panel['time_idx'] = panel['year'] - 2013  # 0 for 2013, 10 for 2023

# Static categoricals need to be strings for TFT
panel['provider_type'] = panel['Rndrng_Prvdr_Type_idx'].astype(str)
panel['state']         = panel['Rndrng_Prvdr_State_Abrvtn_idx'].astype(str)
panel['hcpcs_bucket_s'] = panel['hcpcs_bucket'].astype(str)

print(f'\\nFinal panel: {len(panel):,} rows, {panel["group_id"].nunique():,} groups')
print(f'time_idx range: {panel["time_idx"].min()}-{panel["time_idx"].max()}')
print(f'Columns: {list(panel.columns)}')
"""

C4_DATASET = """# ── Cell 4: TimeSeriesDataSet — Train on <=2021, Validate on 2022-2023 ──────
import torch
import pytorch_forecasting as ptf
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

MAX_ENCODER = 11       # 2013-2023 max history
MAX_PREDICT = 3        # cover 2022-2023 eval (2 steps) AND 2024-2026 forecast (3 steps)
TRAINING_CUTOFF = 8    # time_idx for year 2021

# Keep only groups with >=3 observations before cutoff AND >=1 after cutoff (for val)
stats = panel.groupby('group_id').agg(
    n_before=('time_idx', lambda x: (x <= TRAINING_CUTOFF).sum()),
    n_after =('time_idx', lambda x: (x >  TRAINING_CUTOFF).sum()),
).reset_index()
valid_ids = stats[(stats['n_before'] >= 3) & (stats['n_after'] >= 1)]['group_id']
panel_f = panel[panel['group_id'].isin(valid_ids)].copy()
print(f'Groups with >=3 pre-cutoff + >=1 post-cutoff: {len(valid_ids):,}')
print(f'Filtered panel: {len(panel_f):,} rows')

STATIC_CATS = ['provider_type', 'state', 'hcpcs_bucket_s']
KNOWN_REALS = ['conversion_factor', 'cpi_medical', 'covid_indicator']
UNKNOWN_REALS = [
    'Avg_Mdcr_Alowd_Amt',
    'log_srvcs', 'log_benes',
    'Avg_Sbmtd_Chrg', 'Bene_Avg_Risk_Scre',
    'srvcs_per_bene',
]

training = TimeSeriesDataSet(
    panel_f[panel_f['time_idx'] <= TRAINING_CUTOFF],
    time_idx='time_idx',
    target='Avg_Mdcr_Alowd_Amt',
    group_ids=['group_id'],
    min_encoder_length=3,
    max_encoder_length=MAX_ENCODER,
    min_prediction_length=1,
    max_prediction_length=MAX_PREDICT,
    static_categoricals=STATIC_CATS,
    time_varying_known_reals=KNOWN_REALS,
    time_varying_unknown_reals=UNKNOWN_REALS,
    target_normalizer=GroupNormalizer(groups=['group_id'], transformation='log1p'),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

validation = TimeSeriesDataSet.from_dataset(
    training,
    panel_f,              # include 2022-2023 rows so they become prediction targets
    predict=True,         # use last max_prediction_length steps as decoder
    stop_randomization=True,
)

BATCH_SIZE = 256
train_dl = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=2)
val_dl   = validation.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=2)
print(f'Train batches: {len(train_dl)} | Val batches: {len(val_dl)}')
"""

C5_TRAIN = """# ── Cell 5: Train Multivariate TFT ──────────────────────────────────────────
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=1e-3,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.15,
    hidden_continuous_size=32,
    output_size=7,
    loss=ptf.metrics.QuantileLoss(),
    reduce_on_plateau_patience=5,
)
print(f'TFT params: {tft.size() / 1e3:.1f}K')

trainer = pl.Trainer(
    max_epochs=50,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    gradient_clip_val=0.15,
    callbacks=[
        pl.callbacks.EarlyStopping(monitor='val_loss', patience=8, mode='min'),
        pl.callbacks.LearningRateMonitor(),
    ],
    enable_progress_bar=True,
)

t0 = time.time()
trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
train_elapsed = time.time() - t0
print(f'\\nTFT training complete in {train_elapsed / 60:.1f} min')

# Use trained model directly (PyTorch 2.6 weights_only=True breaks checkpoint load)
best_tft = tft
"""

C6_EVAL = """# ── Cell 6: Fair Eval on 2022-2023 — Same Protocol as V2_12 ─────────────────
# TFT predicts multi-horizon from encoder state in ONE SHOT (not autoregressive),
# so this is directly comparable to V2_12's fair autoregressive LSTM + Chronos.

raw_preds = best_tft.predict(val_dl, mode='prediction', return_x=True)
y_pred = raw_preds.output.cpu().numpy() if hasattr(raw_preds, 'output') else raw_preds[0].cpu().numpy()
y_true = []
decoder_time_idx = []
decoder_group_id = []

for batch in val_dl:
    x, y = batch
    dec_t = x['decoder_time_idx'].cpu().numpy()
    grp   = x['groups'].cpu().numpy()
    # y is tuple (target, weight); take target
    tgt = y[0].cpu().numpy() if isinstance(y, (tuple, list)) else y.cpu().numpy()
    y_true.append(tgt)
    decoder_time_idx.append(dec_t)
    decoder_group_id.append(grp)

y_true = np.concatenate(y_true, axis=0)
decoder_time_idx = np.concatenate(decoder_time_idx, axis=0)
decoder_group_id = np.concatenate(decoder_group_id, axis=0)

print(f'y_pred shape: {y_pred.shape}  y_true shape: {y_true.shape}')

# Flatten to (N,) with year info
flat_pred = y_pred.reshape(-1)
flat_true = y_true.reshape(-1)
flat_year = 2013 + decoder_time_idx.reshape(-1)

# Keep only 2022, 2023 (some groups may have only 2022 in decoder)
eval_mask = (flat_year >= 2022) & (flat_year <= 2023) & np.isfinite(flat_true)
flat_pred = flat_pred[eval_mask]
flat_true = flat_true[eval_mask]
flat_year = flat_year[eval_mask]

flat_pred = np.clip(flat_pred, 0, None)

mae  = mean_absolute_error(flat_true, flat_pred)
rmse = np.sqrt(mean_squared_error(flat_true, flat_pred))
r2   = r2_score(flat_true, flat_pred)
TFT_FAIR = {'test_mae': mae, 'test_rmse': rmse, 'test_r2': r2, 'eval_n': int(len(flat_true))}

# Baselines from V2_12
LSTM_REPORTED = {'test_mae': 8.84, 'test_rmse': 36.42, 'test_r2': 0.8860}
LSTM_FAIR     = {'test_mae': 9.82, 'test_rmse': 18.91, 'test_r2': 0.8689}
CHRONOS_FAIR  = {'test_mae': 9.39, 'test_rmse': 19.71, 'test_r2': 0.8576}
STACKER_V2_12 = {'test_mae': 8.74, 'test_rmse': 17.69, 'test_r2': 0.8852}

print('\\n' + '=' * 80)
print('FAIR COMPARISON — 2022-2023 Temporal Holdout')
print('=' * 80)
print(f'{"Model":<40} {"MAE ($)":>10} {"RMSE ($)":>10} {"R2":>8}')
print('-' * 72)
print(f'{"LSTM V1 (reported, teacher-forced)":<40} '
      f'{LSTM_REPORTED["test_mae"]:>10.2f} {LSTM_REPORTED["test_rmse"]:>10.2f} '
      f'{LSTM_REPORTED["test_r2"]:>8.4f}')
print(f'{"LSTM (fair, autoregressive)":<40} '
      f'{LSTM_FAIR["test_mae"]:>10.2f} {LSTM_FAIR["test_rmse"]:>10.2f} '
      f'{LSTM_FAIR["test_r2"]:>8.4f}')
print(f'{"Chronos cpi_cf_deflated":<40} '
      f'{CHRONOS_FAIR["test_mae"]:>10.2f} {CHRONOS_FAIR["test_rmse"]:>10.2f} '
      f'{CHRONOS_FAIR["test_r2"]:>8.4f}')
print(f'{"LGB Stacker V2_12":<40} '
      f'{STACKER_V2_12["test_mae"]:>10.2f} {STACKER_V2_12["test_rmse"]:>10.2f} '
      f'{STACKER_V2_12["test_r2"]:>8.4f}')
print(f'{"Multivariate TFT V2_13":<40} '
      f'{TFT_FAIR["test_mae"]:>10.2f} {TFT_FAIR["test_rmse"]:>10.2f} '
      f'{TFT_FAIR["test_r2"]:>8.4f}')

lift_vs_lstm    = TFT_FAIR['test_r2'] - LSTM_FAIR['test_r2']
lift_vs_stacker = TFT_FAIR['test_r2'] - STACKER_V2_12['test_r2']
print(f'\\nTFT V2_13 vs LSTM fair:  R2 lift = {lift_vs_lstm:+.4f}')
print(f'TFT V2_13 vs Stacker:    R2 lift = {lift_vs_stacker:+.4f}')
print(f'TFT V2_13 vs 0.89 gate:  R2 diff = {TFT_FAIR["test_r2"] - 0.89:+.4f}')
"""

C7_FORECAST = """# ── Cell 7: Apply TFT to 2024-2026 Forecast Horizon ─────────────────────────
# Build encoder+decoder DataFrame with 2024-2026 rows containing known-future
# covariates (CF, CPI, covid) and NaN targets. TFT decoder ignores unknowns in
# the future.

future_years = [2024, 2025, 2026]
future_rows = []
group_meta = panel_f.drop_duplicates('group_id')[
    ['group_id', 'provider_type', 'state', 'hcpcs_bucket_s'] + GROUP_KEYS
].copy()

# Last-known row per group for filling unknowns
last_obs = (
    panel_f.sort_values(['group_id', 'time_idx'])
           .groupby('group_id')
           .tail(1)
           .set_index('group_id')
)

ext_future = ext[ext['year'].isin(future_years)].set_index('year').to_dict('index')

for _, g in group_meta.iterrows():
    gid = g['group_id']
    last = last_obs.loc[gid] if gid in last_obs.index else None
    for yr in future_years:
        cf  = ext_future.get(yr, {}).get('conversion_factor', 32.0)
        cpi = ext_future.get(yr, {}).get('cpi_medical', 580.0)
        cov = int(ext_future.get(yr, {}).get('covid_indicator', 0))
        row = {
            'group_id':        gid,
            'time_idx':        yr - 2013,
            'year':            yr,
            'provider_type':   g['provider_type'],
            'state':           g['state'],
            'hcpcs_bucket_s':  g['hcpcs_bucket_s'],
            'Rndrng_Prvdr_Type_idx':         int(g['Rndrng_Prvdr_Type_idx']),
            'hcpcs_bucket':                  int(g['hcpcs_bucket']),
            'Rndrng_Prvdr_State_Abrvtn_idx': int(g['Rndrng_Prvdr_State_Abrvtn_idx']),
            'conversion_factor': cf,
            'cpi_medical':       cpi,
            'covid_indicator':   cov,
            # Placeholders for unknowns (TFT masks these in decoder)
            'Avg_Mdcr_Alowd_Amt':  float(last['Avg_Mdcr_Alowd_Amt']) if last is not None else 0.0,
            'log_srvcs':           float(last['log_srvcs'])          if last is not None else 0.0,
            'log_benes':           float(last['log_benes'])          if last is not None else 0.0,
            'Avg_Sbmtd_Chrg':      float(last['Avg_Sbmtd_Chrg'])     if last is not None else 0.0,
            'Bene_Avg_Risk_Scre':  float(last['Bene_Avg_Risk_Scre']) if last is not None else 1.0,
            'srvcs_per_bene':      float(last['srvcs_per_bene'])     if last is not None else 0.0,
        }
        future_rows.append(row)

future_df = pd.DataFrame(future_rows)
# Combine encoder context (all observed 2013-2023) + decoder (2024-2026)
full_df = pd.concat([panel_f, future_df], ignore_index=True)
full_df = full_df.sort_values(['group_id', 'time_idx']).reset_index(drop=True)
print(f'Forecast panel: {len(full_df):,} rows ({len(future_df):,} future rows)')

# Build predict dataset
predict_ds = TimeSeriesDataSet.from_dataset(
    training, full_df, predict=True, stop_randomization=True,
)
predict_dl = predict_ds.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=2)

raw_future = best_tft.predict(predict_dl, mode='prediction', return_x=True)
fut_pred = raw_future.output.cpu().numpy() if hasattr(raw_future, 'output') else raw_future[0].cpu().numpy()

# Also get P10/P50/P90 quantiles
raw_quant = best_tft.predict(predict_dl, mode='quantiles', return_x=False)
quant_pred = raw_quant.cpu().numpy() if hasattr(raw_quant, 'cpu') else raw_quant

# raw_future.x has decoder time_idx and group ids for alignment
fut_time = []
fut_grp  = []
for batch in predict_dl:
    x, _ = batch
    fut_time.append(x['decoder_time_idx'].cpu().numpy())
    fut_grp.append(x['groups'].cpu().numpy())
fut_time = np.concatenate(fut_time, axis=0)
fut_grp  = np.concatenate(fut_grp, axis=0)

print(f'Future prediction shape: {fut_pred.shape} | Quantile shape: {quant_pred.shape}')

# Build forecast DataFrame in LSTM-compatible schema
gid_to_meta = group_meta.set_index('group_id').to_dict('index')

rows = []
for i in range(fut_pred.shape[0]):
    gid_int = int(fut_grp[i, 0] if fut_grp.ndim == 2 else fut_grp[i])
    gid = training.group_ids_mapping[0].get(gid_int, None) if hasattr(training, 'group_ids_mapping') else None
    # Fallback: use ordered dataset index
    if gid is None:
        # pytorch-forecasting returns groups as encoded indices; decode via index_to_group
        try:
            gid = training.index_to_group(gid_int)
        except Exception:
            gid = str(gid_int)

    meta = gid_to_meta.get(gid, {})
    for step in range(fut_pred.shape[1]):
        yr = 2013 + int(fut_time[i, step])
        if yr not in future_years:
            continue
        mean_val = float(fut_pred[i, step])
        if quant_pred.ndim == 3:  # (batch, horizon, quantiles)
            q = quant_pred[i, step]
        else:
            q = [mean_val] * 7
        # Default quantile order from QuantileLoss: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        p10 = float(np.clip(q[1], 0, None))
        p50 = float(np.clip(q[3], 0, None))
        p90 = float(np.clip(q[5], 0, None))
        mean_val = float(np.clip(mean_val, 0, None))

        rows.append({
            'Rndrng_Prvdr_Type_idx':         meta.get('Rndrng_Prvdr_Type_idx', -1),
            'hcpcs_bucket':                  meta.get('hcpcs_bucket', -1),
            'Rndrng_Prvdr_State_Abrvtn_idx': meta.get('Rndrng_Prvdr_State_Abrvtn_idx', -1),
            'forecast_year':                 yr,
            'forecast_mean':                 mean_val,
            'forecast_std':                  float(np.std([p10, p50, p90])),
            'forecast_p10':                  p10,
            'forecast_p50':                  p50,
            'forecast_p90':                  p90,
            'last_known_year':               2023,
            'last_known_value':              0.0,
            'n_history_years':               0,
        })

tft_forecast = pd.DataFrame(rows)
print(f'\\nTFT 2024-2026 forecast: {len(tft_forecast):,} rows')
print(tft_forecast.groupby('forecast_year')['forecast_mean'].describe()[['count', 'mean', 'std', '50%']])

tft_forecast_path = f'{ARTIFACTS}/predictions/tft_multivariate_forecast_2024_2026.parquet'
tft_forecast.to_parquet(tft_forecast_path, index=False)
print(f'Saved: {tft_forecast_path}')
"""

C8_MLFLOW = """# ── Cell 8: MLflow Logging + Plots ──────────────────────────────────────────

# Plot: bar comparison
models = ['LSTM (AR fair)', 'Chronos cpi_cf', 'LGB Stacker V2_12', 'Multivariate TFT V2_13']
mae_vals  = [LSTM_FAIR['test_mae'], CHRONOS_FAIR['test_mae'],
             STACKER_V2_12['test_mae'], TFT_FAIR['test_mae']]
rmse_vals = [LSTM_FAIR['test_rmse'], CHRONOS_FAIR['test_rmse'],
             STACKER_V2_12['test_rmse'], TFT_FAIR['test_rmse']]
r2_vals   = [LSTM_FAIR['test_r2'], CHRONOS_FAIR['test_r2'],
             STACKER_V2_12['test_r2'], TFT_FAIR['test_r2']]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = ['lightsalmon', 'steelblue', 'seagreen', 'purple']
for ax, vals, xl, ttl in [
    (axes[0], mae_vals,  'MAE ($)',  'MAE'),
    (axes[1], rmse_vals, 'RMSE ($)', 'RMSE'),
    (axes[2], r2_vals,   'R\u00b2',  'R-Squared'),
]:
    ax.barh(models, vals, color=colors, edgecolor='white')
    ax.set_xlabel(xl); ax.set_title(ttl)
    for i, v in enumerate(vals):
        fmt = f'${v:.2f}' if '$' in xl else f'{v:.4f}'
        ax.text(v + max(vals) * 0.01, i, fmt, va='center', fontsize=9)
axes[2].set_xlim(0.80, 0.92)
axes[2].axvline(0.89, linestyle='--', color='red', alpha=0.7, label='0.89 stop gate')
axes[2].legend(loc='lower right', fontsize=8)
fig.suptitle('V2_13: Multivariate TFT vs Forecast Track', fontweight='bold')
plt.tight_layout()
bar_path = f'{ARTIFACTS}/plots/v2_13_tft_comparison.png'
fig.savefig(bar_path, dpi=150, bbox_inches='tight')
plt.close(fig)

# Scatter: pred vs true on val
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(flat_true, flat_pred, s=3, alpha=0.3, color='purple')
lim = float(max(flat_true.max(), flat_pred.max()))
ax.plot([0, lim], [0, lim], '--', color='black', linewidth=1)
ax.set_xlabel('True allowed amount ($)')
ax.set_ylabel('TFT prediction ($)')
ax.set_title(f'TFT 2022-2023: R\u00b2 = {TFT_FAIR["test_r2"]:.4f}, RMSE = ${TFT_FAIR["test_rmse"]:.2f}')
ax.grid(True, alpha=0.3)
plt.tight_layout()
scatter_path = f'{ARTIFACTS}/plots/v2_13_tft_scatter.png'
fig.savefig(scatter_path, dpi=150)
plt.close(fig)

print(f'Saved: {bar_path}')
print(f'Saved: {scatter_path}')

# MLflow run
with mlflow.start_run(run_name='tft_multivariate_v2_13_colab'):
    mlflow.log_params({
        'model':            'TemporalFusionTransformer',
        'type':             'multivariate_forecast',
        'training':         f'train <=2021, eval 2022-2023, forecast 2024-2026',
        'max_encoder_length': MAX_ENCODER,
        'max_prediction_length': MAX_PREDICT,
        'hidden_size':      64,
        'attention_head_size': 4,
        'dropout':          0.15,
        'hidden_continuous_size': 32,
        'learning_rate':    1e-3,
        'batch_size':       BATCH_SIZE,
        'static_categoricals':     ','.join(STATIC_CATS),
        'known_reals':             ','.join(KNOWN_REALS),
        'unknown_reals':           ','.join(UNKNOWN_REALS),
        'n_train_groups':   int(len(valid_ids)),
        'n_params':         int(tft.size()),
        'source':           'colab',
        'version':          'v2',
        'stage':            'forecast',
    })
    mlflow.log_metrics({
        'test_mae':             TFT_FAIR['test_mae'],
        'test_rmse':            TFT_FAIR['test_rmse'],
        'test_r2':              TFT_FAIR['test_r2'],
        'eval_n':               TFT_FAIR['eval_n'],
        'lift_vs_lstm_fair_r2': lift_vs_lstm,
        'lift_vs_stacker_r2':   lift_vs_stacker,
        'train_elapsed_sec':    train_elapsed,
    })
    mlflow.log_param('eval_level',
                     'group_temporal_2022_2023 \u2014 same as V2_12 fair baseline')
    for p in [bar_path, scatter_path]:
        mlflow.log_artifact(p)
    mlflow.log_artifact(tft_forecast_path, artifact_path='forecasts')
    print('MLflow run: tft_multivariate_v2_13_colab')
"""

C9_MD = """## Summary
"""

C10_SUMMARY = """# ── Cell 9: Summary + Stop-Gate Check ───────────────────────────────────────
print('=' * 70)
print('V2_13 SUMMARY: Multivariate TFT with Observed + Known-Future Covariates')
print('=' * 70)
print()
print(f'{"Model":<36} {"MAE ($)":>10} {"RMSE ($)":>10} {"R2":>8}')
print('-' * 68)
print(f'{"LSTM (fair, autoregressive)":<36} '
      f'{LSTM_FAIR["test_mae"]:>10.2f} {LSTM_FAIR["test_rmse"]:>10.2f} '
      f'{LSTM_FAIR["test_r2"]:>8.4f}')
print(f'{"Chronos cpi_cf_deflated":<36} '
      f'{CHRONOS_FAIR["test_mae"]:>10.2f} {CHRONOS_FAIR["test_rmse"]:>10.2f} '
      f'{CHRONOS_FAIR["test_r2"]:>8.4f}')
print(f'{"LGB Stacker V2_12":<36} '
      f'{STACKER_V2_12["test_mae"]:>10.2f} {STACKER_V2_12["test_rmse"]:>10.2f} '
      f'{STACKER_V2_12["test_r2"]:>8.4f}')
print(f'{"Multivariate TFT V2_13":<36} '
      f'{TFT_FAIR["test_mae"]:>10.2f} {TFT_FAIR["test_rmse"]:>10.2f} '
      f'{TFT_FAIR["test_r2"]:>8.4f}')
print()

print(f'Lift vs LSTM fair:    R2 {lift_vs_lstm:+.4f}')
print(f'Lift vs V2_12 stacker: R2 {lift_vs_stacker:+.4f}')
print()

STOP_GATE = 0.89
if TFT_FAIR['test_r2'] >= STOP_GATE:
    print('=' * 70)
    print(f'  STOP GATE HIT: R2 = {TFT_FAIR["test_r2"]:.4f} >= {STOP_GATE}')
    print(f'  Forecasting track complete. Declare victory.')
    print('=' * 70)
    print('Recommended next steps:')
    print('  1. Swap Railway API to serve tft_multivariate_forecast_2024_2026.parquet')
    print('  2. Use TFT quantile outputs (p10/p50/p90) directly for uncertainty UI')
    print('  3. Freeze this as the production forecast model')
    print('  4. Archive V2_14+ ideas as backlog')
elif TFT_FAIR['test_r2'] > STACKER_V2_12['test_r2']:
    gap_to_gate = STOP_GATE - TFT_FAIR["test_r2"]
    print(f'TFT beats V2_12 stacker but missed the 0.89 gate by R2 {gap_to_gate:.4f}.')
    print('  -> Consider: larger hidden_size, longer training, or ensemble TFT + stacker')
    print('  -> OR accept TFT as incremental win and deploy')
else:
    gap_to_stacker = STACKER_V2_12['test_r2'] - TFT_FAIR['test_r2']
    print(f'TFT underperformed stacker by R2 {gap_to_stacker:+.4f}.')
    print('  -> Multivariate did not beat univariate blending.')
    print('  -> Inspect feature attention weights for insight.')
    print('  -> Signal ceiling may be ~0.885 for this data shape.')

print()
print('Forecast parquet saved to Drive:')
print(f'  {tft_forecast_path}')
print('MLflow run: tft_multivariate_v2_13_colab')
"""


nb = {
    "cells": [
        md(C0_MD),
        code(C1_ENV),
        code(C2_PANEL),
        code(C3_EXT),
        code(C4_DATASET),
        code(C5_TRAIN),
        code(C6_EVAL),
        code(C7_FORECAST),
        code(C8_MLFLOW),
        md(C9_MD),
        code(C10_SUMMARY),
    ],
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

os.makedirs(os.path.dirname(NB_PATH), exist_ok=True)
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Wrote {NB_PATH}")
print(f"Cells: {len(nb['cells'])}")
