"""
Build colab/V2_12_stacker_forecast.ipynb — LightGBM stacker combining LSTM + Chronos forecasts.

This script just emits the notebook JSON. Run once:
    python scripts/build_v2_12_notebook.py
"""
import json
import os

NB_PATH = os.path.join(os.path.dirname(__file__), "..", "colab", "V2_12_stacker_forecast.ipynb")


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

C0_MD = """# V2_12 — LightGBM Stacker: LSTM + Chronos Forecast Ensemble

**Goal:** Combine LSTM V1 and Chronos-Bolt (cpi_cf_deflated) forecasts via a LightGBM
meta-learner to beat both base models on the 2022–2023 temporal holdout, then apply
the learned blend to the 2024–2026 forecast horizon.

**Hypothesis (from V2_09–V2_11 analysis):**
- LSTM V1: MAE $8.84, RMSE $36.42, R² 0.886 — strong on central tendency, **heavy right tail**
- Chronos cpi_cf_deflated: MAE $9.39, RMSE $19.71, R² 0.858 — weaker R² but **half the RMSE**

The RMSE/MAE ratio (4.12 LSTM vs 2.10 Chronos) shows complementary error distributions.
A non-linear blend should exploit this — LSTM's strength in the middle, Chronos's
tail control — and beat either standalone.

**Strategy:**
1. Re-train LSTM V1 inside this notebook (config matches `modeling/train_lstm_local.py`)
2. Run Chronos-Bolt-Base with CPI+CF deflation (best variant from V2_11)
3. Extract per-group **2022 and 2023 predictions** from both base models → stacker training set
4. Train LightGBM stacker with **GroupKFold OOF** to avoid group-level leakage
5. Refit on all 2022–2023 data, apply to pre-computed 2024–2026 base-model forecasts
6. Save blended forecast parquet in LSTM-compatible schema

**Stacker features:** `lstm_pred`, `chronos_pred`, `forecast_year`, `ptype_idx`,
`state_idx`, `bucket`, `n_history_years`, `last_history_value`, `history_cv`,
`history_trend`, `cpi_factor`, `cf_factor`.

**Note:** Chronos-Bolt's `.predict()` does **not** accept a `num_samples` argument
(it returns 9 fixed quantile predictions). Point estimates are taken as the median
across quantiles — same pattern as V2_11.

**Runtime:** A100 GPU | ~15–20 min | ~3–5 CU
"""

C1_ENV = """# ── Cell 1: Environment Setup ──────────────────────────────────────────────────────────
import os, subprocess, sys

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'mlflow>=2.12', 'databricks-sdk>=0.20'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'chronos-forecasting[gpu]'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'lightgbm>=4.3'])

from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = '/content/drive/MyDrive/AllowanceMap/V2'
SEQ_DATA   = f'{DRIVE_ROOT}/lstm/sequences.parquet'
ENCODERS   = f'{DRIVE_ROOT}/gold/label_encoders.json'
EXT_DIR    = f'{DRIVE_ROOT}/external'
ARTIFACTS  = f'{DRIVE_ROOT}/v2_artifacts'
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

C2_LOAD = """# ── Cell 2: Load Sequences + External Covariates ─────────────────────────────
import gc, json, time, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

GROUP_KEYS = ['Rndrng_Prvdr_Type_idx', 'hcpcs_bucket', 'Rndrng_Prvdr_State_Abrvtn_idx']
LSTM_BASELINE    = {'test_mae': 8.84, 'test_rmse': 36.42, 'test_r2': 0.886}
CHRONOS_BASELINE = {'test_mae': 9.39, 'test_rmse': 19.71, 'test_r2': 0.8576}

# Copy to local SSD
LOCAL_SEQ = '/content/sequences.parquet'
LOCAL_EXT = '/content/external'
if not os.path.exists(LOCAL_SEQ):
    shutil.copy(SEQ_DATA, LOCAL_SEQ)
os.makedirs(LOCAL_EXT, exist_ok=True)
for fname in os.listdir(EXT_DIR):
    src = f'{EXT_DIR}/{fname}'
    dst = f'{LOCAL_EXT}/{fname}'
    if not os.path.exists(dst) and os.path.isfile(src):
        shutil.copy(src, dst)
print('Data copied to local SSD.')

# Load sequences
seq_df = pd.read_parquet(LOCAL_SEQ)
seq_df = seq_df[seq_df['n_years'] >= 3].reset_index(drop=True)
for col in GROUP_KEYS:
    seq_df[col] = seq_df[col].astype(int)
print(f'Sequences: {len(seq_df):,} groups (>= 3 years)')

# Load CPI / CF
ext_files = {
    'conversion_factor':  'conversion_factors.csv',
    'cpi_medical':        'medical_cpi.csv',
}
ext = None
for col, fname in ext_files.items():
    fpath = f'{LOCAL_EXT}/{fname}'
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        ext = df if ext is None else ext.merge(df, on='year', how='outer')

if ext is None:
    # Fallback — hardcoded from V2_11
    ext = pd.DataFrame({
        'year': list(range(2013, 2027)),
        'conversion_factor': [
            34.02, 35.80, 35.75, 35.88, 35.89, 35.99, 36.04, 36.09,
            34.89, 33.06, 33.89, 32.74, 33.29, 31.92
        ],
        'cpi_medical': [
            425.1, 435.3, 446.8, 463.7, 471.4, 478.4, 487.7, 499.4,
            519.3, 545.1, 557.6, 570.2, 578.0, 585.4
        ],
    })
ext = ext.sort_values('year').reset_index(drop=True)
ext_lookup = ext.set_index('year').to_dict('index')

BASE_YEAR = 2013
CPI_BASE = ext_lookup[BASE_YEAR]['cpi_medical']
CF_BASE  = ext_lookup[BASE_YEAR]['conversion_factor']

def cpi_factor(year):
    return ext_lookup.get(year, {}).get('cpi_medical', CPI_BASE) / CPI_BASE

def cf_factor(year):
    return ext_lookup.get(year, {}).get('conversion_factor', CF_BASE) / CF_BASE

print(f'External covariates loaded: {ext["year"].min()}-{ext["year"].max()}')
print(f'Base year {BASE_YEAR}: CPI={CPI_BASE}, CF={CF_BASE}')
"""

C3_BASEFORECAST = """# ── Cell 3: Load Pre-computed 2024-2026 Forecasts from LSTM & Chronos ────────
# These are the outputs of V1 LSTM training and V2_11 Chronos run.
# Used only for the final 2024-2026 stacker application pass.

LSTM_FORECAST_PATH = f'{DRIVE_ROOT}/lstm/forecast_2024_2026.parquet'
CHRONOS_FORECAST_PATH = f'{ARTIFACTS}/predictions/chronos_cpi_cf_deflated_forecast.parquet'

assert os.path.exists(LSTM_FORECAST_PATH), f'Missing LSTM forecast: {LSTM_FORECAST_PATH}'
assert os.path.exists(CHRONOS_FORECAST_PATH), f'Missing Chronos forecast: {CHRONOS_FORECAST_PATH}'

lstm_future = pd.read_parquet(LSTM_FORECAST_PATH)
chronos_future = pd.read_parquet(CHRONOS_FORECAST_PATH)

print(f'LSTM 2024-2026 forecast: {len(lstm_future):,} rows | cols={list(lstm_future.columns)}')
print(f'Chronos 2024-2026 forecast: {len(chronos_future):,} rows')
print(f'LSTM years: {sorted(lstm_future["forecast_year"].unique())}')
print(f'Chronos years: {sorted(chronos_future["forecast_year"].unique())}')

# Cast group keys to int for merging
for df in [lstm_future, chronos_future]:
    for col in GROUP_KEYS:
        df[col] = df[col].astype(int)
"""

# Cell 4 — MedicareLSTM class definition (copied from train_lstm_local.py)
C4_LSTMDEF = """# ── Cell 4: MedicareLSTM Class + Dataset + Temporal Split ────────────────────
# Copied from modeling/train_lstm_local.py — V1 LSTM architecture (64 hidden, 2 layers).
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_


class MedicareLSTM(nn.Module):
    def __init__(self, n_provider_types, n_states, n_hcpcs_buckets=6,
                 embed_dim=8, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.provider_embed = nn.Embedding(n_provider_types, embed_dim)
        self.state_embed    = nn.Embedding(n_states, embed_dim)
        self.bucket_embed   = nn.Embedding(n_hcpcs_buckets, embed_dim)
        self.lstm = nn.LSTM(
            input_size=1 + 3 * embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, target_seq, provider_type, state, hcpcs_bucket):
        B, T = target_seq.shape
        x = target_seq.unsqueeze(-1)
        p = self.provider_embed(provider_type).unsqueeze(1).expand(-1, T, -1)
        s = self.state_embed(state).unsqueeze(1).expand(-1, T, -1)
        h = self.bucket_embed(hcpcs_bucket).unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([x, p, s, h], dim=-1)
        out, _ = self.lstm(x)
        out = self.head_dropout(out)
        return self.head(out).squeeze(-1)


class MedicareSequenceDataset(Dataset):
    def __init__(self, seqs, ptypes, states, buckets, year_lists=None):
        self.seqs, self.ptypes, self.states, self.buckets = seqs, ptypes, states, buckets
        self.year_lists = year_lists

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        item = {
            'input_seq':     torch.tensor(seq[:-1], dtype=torch.float32),
            'target_seq':    torch.tensor(seq[1:],  dtype=torch.float32),
            'provider_type': torch.tensor(self.ptypes[idx], dtype=torch.long),
            'state':         torch.tensor(self.states[idx], dtype=torch.long),
            'hcpcs_bucket':  torch.tensor(self.buckets[idx], dtype=torch.long),
            'seq_len':       len(seq) - 1,
        }
        if self.year_lists is not None:
            years = self.year_lists[idx]
            val_mask = [years[i + 1] > 2021 for i in range(len(seq) - 1)]
            item['val_mask'] = torch.tensor(val_mask, dtype=torch.bool)
        return item


def collate_sequences(batch):
    max_len = max(item['seq_len'] for item in batch)
    B = len(batch)
    has_val_mask = 'val_mask' in batch[0]
    input_seqs  = torch.zeros(B, max_len, dtype=torch.float32)
    target_seqs = torch.zeros(B, max_len, dtype=torch.float32)
    mask        = torch.zeros(B, max_len, dtype=torch.bool)
    val_masks   = torch.zeros(B, max_len, dtype=torch.bool) if has_val_mask else None
    ptypes  = torch.stack([item['provider_type'] for item in batch])
    states  = torch.stack([item['state']         for item in batch])
    buckets = torch.stack([item['hcpcs_bucket']  for item in batch])
    for i, item in enumerate(batch):
        L = item['seq_len']
        input_seqs[i, :L]  = item['input_seq']
        target_seqs[i, :L] = item['target_seq']
        mask[i, :L] = True
        if has_val_mask:
            val_masks[i, :L] = item['val_mask']
    out = {'input_seq': input_seqs, 'target_seq': target_seqs, 'mask': mask,
           'provider_type': ptypes, 'state': states, 'hcpcs_bucket': buckets}
    if has_val_mask:
        out['val_mask'] = val_masks
    return out


def masked_mse_loss(pred, target, mask):
    sq = (pred - target) ** 2
    return (sq * mask.float()).sum() / mask.float().sum().clamp(min=1)


def temporal_split(df, train_end_year=2021):
    train_seqs, train_pt, train_st, train_bk = [], [], [], []
    val_seqs, val_pt, val_st, val_bk, val_years, val_keys = [], [], [], [], [], []
    for _, row in df.iterrows():
        years  = np.array(row['years'])
        target = np.array(row['target_seq'])
        ptype  = int(row['Rndrng_Prvdr_Type_idx'])
        state  = int(row['Rndrng_Prvdr_State_Abrvtn_idx'])
        bucket = int(row['hcpcs_bucket'])

        t_mask = years <= train_end_year
        t_target = target[t_mask]
        if len(t_target) >= 2:
            train_seqs.append(np.log1p(t_target))
            train_pt.append(ptype); train_st.append(state); train_bk.append(bucket)

        if np.any(years > train_end_year) and len(target) >= 3:
            val_seqs.append(np.log1p(target))
            val_pt.append(ptype); val_st.append(state); val_bk.append(bucket)
            val_years.append(years.tolist())
            val_keys.append((ptype, bucket, state))

    return (
        (train_seqs, np.array(train_pt), np.array(train_st), np.array(train_bk)),
        (val_seqs, np.array(val_pt), np.array(val_st), np.array(val_bk), val_years, val_keys),
    )


# Determine vocab sizes
n_provider_types = int(seq_df['Rndrng_Prvdr_Type_idx'].max()) + 1
n_states         = int(seq_df['Rndrng_Prvdr_State_Abrvtn_idx'].max()) + 1
n_hcpcs_buckets  = int(seq_df['hcpcs_bucket'].max()) + 1
print(f'Vocab: {n_provider_types} ptypes, {n_states} states, {n_hcpcs_buckets} buckets')
"""

C5_TRAIN_LSTM = """# ── Cell 5: Train LSTM V1 In-Notebook ────────────────────────────────────────
# Reproduces the V1 LSTM configuration that achieved R² 0.886 on 2022-2023 holdout.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

train_data, val_data = temporal_split(seq_df, train_end_year=2021)
train_seqs, train_pt, train_st, train_bk = train_data
val_seqs, val_pt, val_st, val_bk, val_years, val_keys = val_data
print(f'Train groups: {len(train_seqs):,} | Val groups: {len(val_seqs):,}')

train_ds = MedicareSequenceDataset(train_seqs, train_pt, train_st, train_bk)
val_ds   = MedicareSequenceDataset(val_seqs, val_pt, val_st, val_bk, year_lists=val_years)
train_dl = DataLoader(train_ds, batch_size=256, shuffle=True,
                      collate_fn=collate_sequences, num_workers=2, pin_memory=True)
val_dl   = DataLoader(val_ds, batch_size=512, shuffle=False,
                      collate_fn=collate_sequences, num_workers=2, pin_memory=True)

model = MedicareLSTM(
    n_provider_types=n_provider_types,
    n_states=n_states,
    n_hcpcs_buckets=n_hcpcs_buckets,
    embed_dim=8, hidden_size=64, num_layers=2, dropout=0.2,
).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f'LSTM params: {n_params:,}')

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

best_val_loss, best_state, wait, patience = float('inf'), None, 0, 10
N_EPOCHS = 50

t_train = time.time()
for epoch in range(1, N_EPOCHS + 1):
    model.train()
    train_loss_sum, train_n = 0.0, 0
    for batch in train_dl:
        inp  = batch['input_seq'].to(device); tgt = batch['target_seq'].to(device)
        msk  = batch['mask'].to(device)
        pt   = batch['provider_type'].to(device)
        st   = batch['state'].to(device); bk = batch['hcpcs_bucket'].to(device)
        pred = model(inp, pt, st, bk)
        loss = masked_mse_loss(pred, tgt, msk)
        optimizer.zero_grad(); loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss_sum += loss.item() * msk.float().sum().item()
        train_n += msk.float().sum().item()
    train_loss = train_loss_sum / max(train_n, 1)

    model.eval()
    val_loss_sum, val_n = 0.0, 0
    with torch.no_grad():
        for batch in val_dl:
            inp  = batch['input_seq'].to(device); tgt = batch['target_seq'].to(device)
            msk  = batch['mask'].to(device); vmsk = batch['val_mask'].to(device)
            pt   = batch['provider_type'].to(device)
            st   = batch['state'].to(device); bk = batch['hcpcs_bucket'].to(device)
            pred = model(inp, pt, st, bk)
            eff_mask = msk & vmsk
            if eff_mask.sum() > 0:
                loss = masked_mse_loss(pred, tgt, eff_mask)
                val_loss_sum += loss.item() * eff_mask.float().sum().item()
                val_n += eff_mask.float().sum().item()
    val_loss = val_loss_sum / max(val_n, 1)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1

    if epoch % 5 == 0 or epoch == 1:
        print(f'  Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}')
    if wait >= patience:
        print(f'  Early stop at epoch {epoch}')
        break

model.load_state_dict(best_state)
print(f'\\nLSTM training complete in {time.time() - t_train:.1f}s')
"""

C6_LSTM_EVAL = """# ── Cell 6: LSTM AUTOREGRESSIVE Eval → 2022-2023 Per-Group Predictions ───────
# IMPORTANT: We rollout LSTM autoregressively from context <=2021, feeding
# its own prediction back as input. This matches inference-time behavior
# (the 2024-2026 forecast in forecast_2024_2026.parquet is also autoregressive).
# The reported V1 LSTM baseline R²=0.886 used teacher forcing (LSTM saw true
# 2022 when predicting 2023) — that's an inflated number. Expect the fair
# autoregressive R² here to be LOWER than 0.886 but more comparable to Chronos.

model.eval()
lstm_eval_rows = []

# Precompute per-group context-only log seq and metadata
ctx_bundles = []
for idx in range(len(val_seqs)):
    full_log = val_seqs[idx]      # log1p full sequence
    yrs      = val_years[idx]
    ptype, bucket, state = val_keys[idx]

    # Split on year <= 2021
    ctx_log   = np.array([fl for fl, y in zip(full_log, yrs) if y <= 2021], dtype=np.float32)
    eval_yrs  = [y for y in yrs if y > 2021]
    eval_vals = [float(np.expm1(fl)) for fl, y in zip(full_log, yrs) if y > 2021]
    if len(ctx_log) < 2 or not eval_yrs:
        continue

    hist_raw = [float(np.expm1(fl)) for fl, y in zip(full_log, yrs) if y <= 2021]
    hist_mean  = float(np.mean(hist_raw))
    hist_std   = float(np.std(hist_raw))
    hist_cv    = hist_std / (abs(hist_mean) + 1e-6)
    hist_trend = (hist_raw[-1] - hist_raw[0]) / max(len(hist_raw) - 1, 1)

    ctx_bundles.append({
        'ctx_log':        ctx_log,
        'eval_years':     eval_yrs,
        'eval_vals':      eval_vals,
        'ptype':          ptype,
        'state':          state,
        'bucket':         bucket,
        'n_hist':         len(hist_raw),
        'last_hist':      hist_raw[-1],
        'hist_mean':      hist_mean,
        'hist_cv':        hist_cv,
        'hist_trend':     hist_trend,
    })
print(f'Autoregressive rollout over {len(ctx_bundles):,} groups...')

# Batched rollout: pad contexts to common length, rollout all groups in parallel
BATCH = 1024
N_FORWARD = 2  # 2022, 2023

t0 = time.time()
with torch.no_grad():
    for start in range(0, len(ctx_bundles), BATCH):
        batch = ctx_bundles[start:start + BATCH]
        B = len(batch)
        lens = [len(b['ctx_log']) for b in batch]
        max_len = max(lens)

        padded = np.zeros((B, max_len), dtype=np.float32)
        for i, b in enumerate(batch):
            padded[i, :lens[i]] = b['ctx_log']
        cur = torch.from_numpy(padded).to(device)
        cur_lens = list(lens)

        pt = torch.tensor([b['ptype']  for b in batch], dtype=torch.long, device=device)
        st = torch.tensor([b['state']  for b in batch], dtype=torch.long, device=device)
        bk = torch.tensor([b['bucket'] for b in batch], dtype=torch.long, device=device)

        step_preds_log = [[] for _ in range(B)]  # list of per-step log1p predictions

        for step in range(N_FORWARD):
            out = model(cur, pt, st, bk)  # (B, T)
            next_log = np.zeros(B, dtype=np.float32)
            for i, L in enumerate(cur_lens):
                next_log[i] = out[i, L - 1].item()
                step_preds_log[i].append(next_log[i])

            # Grow cur by 1 column, writing next_log at the new position
            new_col = torch.zeros((B, 1), dtype=torch.float32, device=device)
            cur = torch.cat([cur, new_col], dim=1)
            for i, L in enumerate(cur_lens):
                cur[i, L] = float(next_log[i])
                cur_lens[i] = L + 1

        for i, b in enumerate(batch):
            n_eval = len(b['eval_years'])
            preds_log = step_preds_log[i][:n_eval]
            preds_nom = [max(float(np.expm1(p)), 0.0) for p in preds_log]
            for k in range(n_eval):
                lstm_eval_rows.append({
                    'Rndrng_Prvdr_Type_idx':         b['ptype'],
                    'hcpcs_bucket':                  b['bucket'],
                    'Rndrng_Prvdr_State_Abrvtn_idx': b['state'],
                    'forecast_year':                 int(b['eval_years'][k]),
                    'lstm_pred':                     preds_nom[k],
                    'true_value':                    b['eval_vals'][k],
                    'n_history_years':               b['n_hist'],
                    'last_history_value':            b['last_hist'],
                    'history_mean':                  b['hist_mean'],
                    'history_cv':                    b['hist_cv'],
                    'history_trend':                 b['hist_trend'],
                })

lstm_eval_df = pd.DataFrame(lstm_eval_rows)
print(f'Autoregressive rollout done in {time.time() - t0:.1f}s')
print(f'LSTM eval rows: {len(lstm_eval_df):,}')
print(f'Years: {sorted(lstm_eval_df["forecast_year"].unique())}')

# Fair (autoregressive) LSTM standalone — expected lower than 0.886
mae  = mean_absolute_error(lstm_eval_df['true_value'], lstm_eval_df['lstm_pred'])
rmse = np.sqrt(mean_squared_error(lstm_eval_df['true_value'], lstm_eval_df['lstm_pred']))
r2   = r2_score(lstm_eval_df['true_value'], lstm_eval_df['lstm_pred'])
LSTM_AR_FAIR = {'test_mae': mae, 'test_rmse': rmse, 'test_r2': r2}
print(f'LSTM (autoregressive, fair): MAE=${mae:.2f} RMSE=${rmse:.2f} R2={r2:.4f}')
print(f'  vs reported V1 (teacher-forced):        R2={LSTM_BASELINE["test_r2"]:.4f}')
"""

C7_CHRONOS_EVAL = """# ── Cell 7: Chronos cpi_cf_deflated Eval Pass ────────────────────────────────
from chronos import BaseChronosPipeline

print('Loading Chronos-Bolt-Base...')
t_load = time.time()
pipeline = BaseChronosPipeline.from_pretrained(
    'autogluon/chronos-bolt-base',
    device_map='cuda',
    torch_dtype=torch.float32,
)
print(f'Loaded in {time.time() - t_load:.1f}s')

# Build deflated context / eval records (same logic as V2_11)
def cpi_cf_deflate(val, yr):
    return val / (cpi_factor(yr) * cf_factor(yr))

def cpi_cf_reinflate(val, yr):
    return val * (cpi_factor(yr) * cf_factor(yr))


chronos_records = []
for _, row in seq_df.iterrows():
    years  = np.array(row['years'])
    values = np.array(row['target_seq'], dtype=np.float64)
    deflated = np.array([cpi_cf_deflate(v, int(y)) for v, y in zip(values, years)])

    t_mask = years <= 2021
    e_mask = years > 2021
    context = deflated[t_mask]
    if len(context) < 2 or e_mask.sum() == 0:
        continue
    chronos_records.append({
        'context':       context.astype(np.float32),
        'eval_years':    years[e_mask],
        'eval_nominal':  values[e_mask],  # ground truth
        'n_eval':        int(e_mask.sum()),
        'ptype':         int(row['Rndrng_Prvdr_Type_idx']),
        'state':         int(row['Rndrng_Prvdr_State_Abrvtn_idx']),
        'bucket':        int(row['hcpcs_bucket']),
    })
print(f'Chronos eval records: {len(chronos_records):,}')

# Batch inference — note: Chronos-Bolt's .predict() does NOT accept num_samples.
# It returns a tensor of shape (batch, num_quantiles=9, horizon).
BATCH_SIZE = 512
chronos_eval_rows = []
t0 = time.time()

for start in range(0, len(chronos_records), BATCH_SIZE):
    batch = chronos_records[start:start + BATCH_SIZE]
    contexts = [torch.tensor(r['context'], dtype=torch.float32) for r in batch]
    max_h = max(r['n_eval'] for r in batch)
    samples = pipeline.predict(contexts, prediction_length=max_h)  # no num_samples!
    for i, r in enumerate(batch):
        h = r['n_eval']
        s = samples[i, :, :h].cpu().numpy()        # (9, h)
        median_deflated = np.median(s, axis=0)     # (h,) — central quantile
        for k in range(h):
            yr = int(r['eval_years'][k])
            pred_nominal = cpi_cf_reinflate(median_deflated[k], yr)
            pred_nominal = max(pred_nominal, 0.0)
            chronos_eval_rows.append({
                'Rndrng_Prvdr_Type_idx':         r['ptype'],
                'hcpcs_bucket':                  r['bucket'],
                'Rndrng_Prvdr_State_Abrvtn_idx': r['state'],
                'forecast_year':                 yr,
                'chronos_pred':                  float(pred_nominal),
            })

chronos_eval_df = pd.DataFrame(chronos_eval_rows)
print(f'Chronos inference: {time.time() - t0:.1f}s | {len(chronos_eval_df):,} rows')

# Sanity check
merged_check = lstm_eval_df.merge(chronos_eval_df, on=GROUP_KEYS + ['forecast_year'], how='inner')
mae  = mean_absolute_error(merged_check['true_value'], merged_check['chronos_pred'])
rmse = np.sqrt(mean_squared_error(merged_check['true_value'], merged_check['chronos_pred']))
r2   = r2_score(merged_check['true_value'], merged_check['chronos_pred'])
print(f'Chronos standalone on overlap: MAE=${mae:.2f} RMSE=${rmse:.2f} R2={r2:.4f}')

del pipeline
torch.cuda.empty_cache()
gc.collect()
"""

C8_STACKER_DATA = """# ── Cell 8: Build Stacker Training Dataset ───────────────────────────────────

# Merge LSTM and Chronos per-group-year predictions
stacker_df = lstm_eval_df.merge(
    chronos_eval_df,
    on=GROUP_KEYS + ['forecast_year'],
    how='inner',
)
print(f'Stacker training rows: {len(stacker_df):,}')

# Add CPI/CF factors as features
stacker_df['cpi_factor'] = stacker_df['forecast_year'].apply(cpi_factor)
stacker_df['cf_factor']  = stacker_df['forecast_year'].apply(cf_factor)

# Sanity check distributions
print()
print('Feature summary:')
for col in ['lstm_pred', 'chronos_pred', 'true_value', 'last_history_value', 'history_cv']:
    print(f'  {col:20s}: mean={stacker_df[col].mean():10.2f}  std={stacker_df[col].std():10.2f}')
print(f'  forecast_year breakdown: {stacker_df["forecast_year"].value_counts().to_dict()}')

# Correlations
print()
print('Base model correlations with truth:')
print(f'  LSTM    <-> true: {stacker_df[["lstm_pred", "true_value"]].corr().iloc[0, 1]:.4f}')
print(f'  Chronos <-> true: {stacker_df[["chronos_pred", "true_value"]].corr().iloc[0, 1]:.4f}')
print(f'  LSTM    <-> Chronos: {stacker_df[["lstm_pred", "chronos_pred"]].corr().iloc[0, 1]:.4f}')
"""

C9_STACKER_CV = """# ── Cell 9: GroupKFold CV → Stacker OOF Metrics ──────────────────────────────
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

FEATURES = [
    'lstm_pred',
    'chronos_pred',
    'forecast_year',
    'Rndrng_Prvdr_Type_idx',
    'Rndrng_Prvdr_State_Abrvtn_idx',
    'hcpcs_bucket',
    'n_history_years',
    'last_history_value',
    'history_mean',
    'history_cv',
    'history_trend',
    'cpi_factor',
    'cf_factor',
]
TARGET = 'true_value'
CAT_FEATURES = ['Rndrng_Prvdr_Type_idx', 'Rndrng_Prvdr_State_Abrvtn_idx', 'hcpcs_bucket']

# Group ID = (ptype, bucket, state) tuple hash — keeps both years of a group together
stacker_df['group_id'] = (
    stacker_df['Rndrng_Prvdr_Type_idx'].astype(str) + '_' +
    stacker_df['hcpcs_bucket'].astype(str) + '_' +
    stacker_df['Rndrng_Prvdr_State_Abrvtn_idx'].astype(str)
)

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'num_threads': -1,
}
N_FOLDS = 5
NUM_BOOST_ROUND = 1000
EARLY_STOP = 50

gkf = GroupKFold(n_splits=N_FOLDS)
oof_pred = np.zeros(len(stacker_df))
fold_metrics = []

t0 = time.time()
for fold, (tr_idx, va_idx) in enumerate(gkf.split(stacker_df, groups=stacker_df['group_id']), 1):
    X_tr = stacker_df.iloc[tr_idx][FEATURES]
    y_tr = stacker_df.iloc[tr_idx][TARGET]
    X_va = stacker_df.iloc[va_idx][FEATURES]
    y_va = stacker_df.iloc[va_idx][TARGET]

    tr_ds = lgb.Dataset(X_tr, y_tr, categorical_feature=CAT_FEATURES)
    va_ds = lgb.Dataset(X_va, y_va, categorical_feature=CAT_FEATURES, reference=tr_ds)

    booster = lgb.train(
        LGB_PARAMS, tr_ds,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[va_ds],
        callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(0)],
    )
    oof_pred[va_idx] = booster.predict(X_va, num_iteration=booster.best_iteration)
    oof_pred[va_idx] = np.clip(oof_pred[va_idx], 0, None)

    fmae  = mean_absolute_error(y_va, oof_pred[va_idx])
    frmse = np.sqrt(mean_squared_error(y_va, oof_pred[va_idx]))
    fr2   = r2_score(y_va, oof_pred[va_idx])
    fold_metrics.append((fmae, frmse, fr2))
    print(f'  Fold {fold}: MAE=${fmae:.2f}  RMSE=${frmse:.2f}  R2={fr2:.4f}  '
          f'(best_iter={booster.best_iteration})')

print(f'\\nCV total time: {time.time() - t0:.1f}s')

# Aggregate OOF metrics
stacker_mae  = mean_absolute_error(stacker_df[TARGET], oof_pred)
stacker_rmse = np.sqrt(mean_squared_error(stacker_df[TARGET], oof_pred))
stacker_r2   = r2_score(stacker_df[TARGET], oof_pred)

print('\\n' + '=' * 75)
print('STACKER OOF RESULTS vs BASE MODELS (2022-2023 holdout)')
print('=' * 75)
print(f'{"Model":<36} {"MAE ($)":>10} {"RMSE ($)":>10} {"R2":>8}')
print('-' * 68)
print(f'{"LSTM V1 (reported, teacher-forced)":<36} {LSTM_BASELINE["test_mae"]:>10.2f} '
      f'{LSTM_BASELINE["test_rmse"]:>10.2f} {LSTM_BASELINE["test_r2"]:>8.4f}')
print(f'{"LSTM (fair, autoregressive)":<36} {LSTM_AR_FAIR["test_mae"]:>10.2f} '
      f'{LSTM_AR_FAIR["test_rmse"]:>10.2f} {LSTM_AR_FAIR["test_r2"]:>8.4f}')
print(f'{"Chronos cpi_cf_deflated":<36} {CHRONOS_BASELINE["test_mae"]:>10.2f} '
      f'{CHRONOS_BASELINE["test_rmse"]:>10.2f} {CHRONOS_BASELINE["test_r2"]:>8.4f}')
print(f'{"LGB Stacker (OOF)":<36} {stacker_mae:>10.2f} {stacker_rmse:>10.2f} {stacker_r2:>8.4f}')

lift_vs_lstm_reported = stacker_r2 - LSTM_BASELINE['test_r2']
lift_vs_lstm_fair     = stacker_r2 - LSTM_AR_FAIR['test_r2']
lift_vs_chronos       = stacker_r2 - CHRONOS_BASELINE['test_r2']
print(f'\\nStacker vs LSTM reported  (teacher-forced): R2 lift = {lift_vs_lstm_reported:+.4f}')
print(f'Stacker vs LSTM fair      (autoregressive): R2 lift = {lift_vs_lstm_fair:+.4f}')
print(f'Stacker vs Chronos cpi_cf                 : R2 lift = {lift_vs_chronos:+.4f}')
"""

C10_FINAL_FORECAST = """# ── Cell 10: Train Final Stacker on All Eval Data → Apply to 2024-2026 Forecasts

# Refit on the full 2022-2023 training set (no held-out fold)
X_full = stacker_df[FEATURES]
y_full = stacker_df[TARGET]

# Mean best_iteration from CV as num_boost_round (no validation set here)
mean_best_iter = int(np.mean([len(fold_metrics)]) * 200)  # placeholder fallback
# Better: use last fold's booster.best_iteration — but safer to fit with a fixed budget.
final_rounds = 400  # conservative

full_ds = lgb.Dataset(X_full, y_full, categorical_feature=CAT_FEATURES)
final_booster = lgb.train(
    LGB_PARAMS, full_ds,
    num_boost_round=final_rounds,
    callbacks=[lgb.log_evaluation(0)],
)
print(f'Final stacker trained on {len(stacker_df):,} rows, {final_rounds} boost rounds')

# Feature importance
imp = pd.DataFrame({
    'feature': FEATURES,
    'importance': final_booster.feature_importance(importance_type='gain'),
}).sort_values('importance', ascending=False)
print('\\nFeature importance (gain):')
for _, r in imp.iterrows():
    print(f'  {r["feature"]:30s}: {r["importance"]:>10.0f}')

# -------- Apply to 2024-2026 horizon --------
# Need: LSTM mean + Chronos mean forecasts per (group, year), plus same static feats
lstm_future_slim = lstm_future[
    GROUP_KEYS + ['forecast_year', 'forecast_mean', 'last_known_value', 'n_history_years']
].rename(columns={
    'forecast_mean': 'lstm_pred',
    'last_known_value': 'last_history_value',
})

chronos_future_slim = chronos_future[
    GROUP_KEYS + ['forecast_year', 'forecast_mean']
].rename(columns={'forecast_mean': 'chronos_pred'})

future_df = lstm_future_slim.merge(
    chronos_future_slim,
    on=GROUP_KEYS + ['forecast_year'],
    how='inner',
)
print(f'\\n2024-2026 merged horizon: {len(future_df):,} rows')

# Rebuild history stats per group from sequences
hist_stats = []
for _, row in seq_df.iterrows():
    yrs = np.array(row['years'])
    vals = np.array(row['target_seq'], dtype=np.float64)
    # Use ALL history (years ≤ 2023) as the "context" for the 2024-2026 forecast
    hist_stats.append({
        'Rndrng_Prvdr_Type_idx':         int(row['Rndrng_Prvdr_Type_idx']),
        'hcpcs_bucket':                  int(row['hcpcs_bucket']),
        'Rndrng_Prvdr_State_Abrvtn_idx': int(row['Rndrng_Prvdr_State_Abrvtn_idx']),
        'history_mean':  float(np.mean(vals)),
        'history_cv':    float(np.std(vals) / (abs(np.mean(vals)) + 1e-6)),
        'history_trend': float((vals[-1] - vals[0]) / max(len(vals) - 1, 1)),
    })
hist_df = pd.DataFrame(hist_stats)
future_df = future_df.merge(hist_df, on=GROUP_KEYS, how='left')

# Factors
future_df['cpi_factor'] = future_df['forecast_year'].apply(cpi_factor)
future_df['cf_factor']  = future_df['forecast_year'].apply(cf_factor)

# Ensure all feature columns present in correct order
for f in FEATURES:
    if f not in future_df.columns:
        print(f'WARN: missing feature {f}')

future_df['stacker_pred'] = final_booster.predict(future_df[FEATURES])
future_df['stacker_pred'] = np.clip(future_df['stacker_pred'], 0, None)

print('\\n2024-2026 stacker forecast summary:')
print(future_df.groupby('forecast_year')['stacker_pred'].describe()[['count', 'mean', 'std', '50%']])

# Save in LSTM-compatible schema
out = future_df[GROUP_KEYS + ['forecast_year']].copy()
out['forecast_mean'] = future_df['stacker_pred']
out['forecast_std']  = 0.0  # stacker is point only; leave 0 or compute from CV later
out['forecast_p10']  = future_df['stacker_pred']
out['forecast_p50']  = future_df['stacker_pred']
out['forecast_p90']  = future_df['stacker_pred']
out['last_known_year']  = 2023
out['last_known_value'] = future_df['last_history_value']
out['n_history_years']  = future_df['n_history_years']

stacker_forecast_path = f'{ARTIFACTS}/predictions/stacker_forecast_2024_2026.parquet'
out.to_parquet(stacker_forecast_path, index=False)
print(f'\\nSaved: {stacker_forecast_path}  ({len(out):,} rows)')
"""

C11_MLFLOW = """# ── Cell 11: MLflow Logging + Plots ──────────────────────────────────────────

# Plot 1 — bar comparison
models = ['LSTM V1 (TF)', 'LSTM (AR, fair)', 'Chronos cpi_cf', 'LGB Stacker (OOF)']
mae_vals  = [LSTM_BASELINE['test_mae'], LSTM_AR_FAIR['test_mae'],
             CHRONOS_BASELINE['test_mae'], stacker_mae]
rmse_vals = [LSTM_BASELINE['test_rmse'], LSTM_AR_FAIR['test_rmse'],
             CHRONOS_BASELINE['test_rmse'], stacker_rmse]
r2_vals   = [LSTM_BASELINE['test_r2'], LSTM_AR_FAIR['test_r2'],
             CHRONOS_BASELINE['test_r2'], stacker_r2]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = ['coral', 'lightsalmon', 'steelblue', 'seagreen']
for ax, vals, xl, ttl in [
    (axes[0], mae_vals,  'MAE ($)',  'Mean Absolute Error'),
    (axes[1], rmse_vals, 'RMSE ($)', 'Root Mean Squared Error'),
    (axes[2], r2_vals,   'R\u00b2',  'R-Squared'),
]:
    ax.barh(models, vals, color=colors, edgecolor='white')
    ax.set_xlabel(xl); ax.set_title(ttl)
    for i, v in enumerate(vals):
        fmt = f'${v:.2f}' if '$' in xl else f'{v:.4f}'
        ax.text(v + max(vals) * 0.01, i, fmt, va='center', fontsize=9)
axes[2].set_xlim(0, 1.0)
fig.suptitle('V2_12: Stacker vs Base Models (2022-2023)', fontweight='bold')
plt.tight_layout()
bar_path = f'{ARTIFACTS}/plots/v2_12_stacker_comparison.png'
fig.savefig(bar_path, dpi=150, bbox_inches='tight')
plt.close(fig)

# Plot 2 — predicted vs true scatter for stacker OOF
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(stacker_df[TARGET], oof_pred, s=3, alpha=0.3, color='seagreen')
lim = max(stacker_df[TARGET].max(), oof_pred.max())
ax.plot([0, lim], [0, lim], '--', color='black', linewidth=1)
ax.set_xlabel('True allowed amount ($)')
ax.set_ylabel('Stacker OOF prediction ($)')
ax.set_title(f'Stacker OOF: R\u00b2 = {stacker_r2:.4f}, RMSE = ${stacker_rmse:.2f}')
ax.grid(True, alpha=0.3)
plt.tight_layout()
scatter_path = f'{ARTIFACTS}/plots/v2_12_stacker_scatter.png'
fig.savefig(scatter_path, dpi=150)
plt.close(fig)

# Plot 3 — feature importance
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(imp['feature'][::-1], imp['importance'][::-1], color='steelblue')
ax.set_xlabel('Gain')
ax.set_title('Stacker feature importance')
plt.tight_layout()
imp_path = f'{ARTIFACTS}/plots/v2_12_feature_importance.png'
fig.savefig(imp_path, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f'Saved: {bar_path}')
print(f'Saved: {scatter_path}')
print(f'Saved: {imp_path}')

# -------- MLflow --------
with mlflow.start_run(run_name='lgb_stacker_v2_12_colab'):
    mlflow.log_params({
        'model':           'LightGBM stacker',
        'type':            'meta_learner',
        'base_models':     'LSTM_V1 + Chronos_Bolt_cpi_cf_deflated',
        'training':        '5-fold GroupKFold CV',
        'n_folds':         N_FOLDS,
        'num_boost_round_cv': NUM_BOOST_ROUND,
        'early_stop':      EARLY_STOP,
        'final_rounds':    final_rounds,
        'n_features':      len(FEATURES),
        'n_train_rows':    len(stacker_df),
        'train_end_year':  2021,
        'eval_years':      '2022_2023',
        'source':          'colab',
        'version':         'v2',
        'stage':           'forecast',
        **{f'lgb_{k}': v for k, v in LGB_PARAMS.items() if k != 'num_threads'},
    })
    mlflow.log_metrics({
        'test_mae':                 stacker_mae,
        'test_rmse':                stacker_rmse,
        'test_r2':                  stacker_r2,
        'eval_n_groups':            len(stacker_df),
        'lstm_ar_fair_r2':          LSTM_AR_FAIR['test_r2'],
        'lstm_ar_fair_rmse':        LSTM_AR_FAIR['test_rmse'],
        'lstm_ar_fair_mae':         LSTM_AR_FAIR['test_mae'],
        'lift_vs_lstm_fair_r2':     lift_vs_lstm_fair,
        'lift_vs_lstm_reported_r2': lift_vs_lstm_reported,
        'lift_vs_chronos_r2':       lift_vs_chronos,
    })
    mlflow.log_param('eval_level',
                     'group_temporal_2022_2023 \u2014 same as LSTM baseline')
    for p in [bar_path, scatter_path, imp_path]:
        mlflow.log_artifact(p)
    mlflow.log_artifact(stacker_forecast_path, artifact_path='forecasts')
    print('MLflow run: lgb_stacker_v2_12_colab')
"""

C12_MD = """## Summary
"""

C13_SUMMARY = """# ── Cell 12: Summary ─────────────────────────────────────────────────────────
print('=' * 70)
print('V2_12 SUMMARY: LightGBM Stacker — LSTM + Chronos')
print('=' * 70)
print()
print('Strategy: Blend LSTM V1 and Chronos-Bolt (cpi_cf_deflated) forecasts')
print('          via a LightGBM meta-learner. Exploit the complementary error')
print('          profiles observed in V2_09-V2_11 (LSTM heavy right tail,')
print('          Chronos lower RMSE, higher MAE).')
print()
print(f'{"Model":<36} {"MAE ($)":>10} {"RMSE ($)":>10} {"R2":>8}')
print('-' * 68)
print(f'{"LSTM V1 (reported, teacher-forced)":<36} {LSTM_BASELINE["test_mae"]:>10.2f} '
      f'{LSTM_BASELINE["test_rmse"]:>10.2f} {LSTM_BASELINE["test_r2"]:>8.4f}')
print(f'{"LSTM (fair, autoregressive)":<36} {LSTM_AR_FAIR["test_mae"]:>10.2f} '
      f'{LSTM_AR_FAIR["test_rmse"]:>10.2f} {LSTM_AR_FAIR["test_r2"]:>8.4f}')
print(f'{"Chronos cpi_cf_deflated":<36} {CHRONOS_BASELINE["test_mae"]:>10.2f} '
      f'{CHRONOS_BASELINE["test_rmse"]:>10.2f} {CHRONOS_BASELINE["test_r2"]:>8.4f}')
print(f'{"LGB Stacker (OOF)":<36} {stacker_mae:>10.2f} {stacker_rmse:>10.2f} {stacker_r2:>8.4f}')
print()

if lift_vs_lstm_fair > 0:
    print(f'RESULT: Stacker beats fair LSTM baseline by R2 {lift_vs_lstm_fair:+.4f}')
    print(f'  -> Deploy stacker as the forecast model for 2024-2026 predictions.')
    print(f'  -> Note: comparison is against autoregressive LSTM, matching')
    print(f'     inference-time conditions. Reported 0.886 was teacher-forced.')
else:
    print(f'RESULT: Stacker did not beat fair LSTM baseline (R2 {lift_vs_lstm_fair:+.4f})')
    print(f'  -> Inspect feature importance and base-model correlation.')
    print(f'  -> Consider: (a) richer features, (b) fine-tuning Chronos,')
    print(f'     (c) rolling-origin folds for more training data.')

print()
print('All runs logged to MLflow. Stacker forecast parquet saved to Drive.')
print(f'  {stacker_forecast_path}')
"""


# ---------------------------------------------------------------------------
# Assemble notebook
# ---------------------------------------------------------------------------
nb = {
    "cells": [
        md(C0_MD),
        code(C1_ENV),
        code(C2_LOAD),
        code(C3_BASEFORECAST),
        code(C4_LSTMDEF),
        code(C5_TRAIN_LSTM),
        code(C6_LSTM_EVAL),
        code(C7_CHRONOS_EVAL),
        code(C8_STACKER_DATA),
        code(C9_STACKER_CV),
        code(C10_FINAL_FORECAST),
        code(C11_MLFLOW),
        md(C12_MD),
        code(C13_SUMMARY),
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
