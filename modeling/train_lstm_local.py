"""
train_lstm_local.py — PyTorch LSTM time-series forecasting, logs to Databricks MLflow

Predicts Medicare allowed amounts (Avg_Mdcr_Alowd_Amt) 2-3 years forward
by specialty x HCPCS bucket x state using year-ordered sequences from
05_lstm_sequences_local.py.

Temporal split: train on years <= 2021, validate on 2022-2023.
Forecast: autoregressive rollout 2024-2026 with MC Dropout confidence bounds.

Required env vars:
    DATABRICKS_HOST   e.g. https://<workspace>.azuredatabricks.net
    DATABRICKS_TOKEN  Personal Access Token or service principal secret

Usage:
    python modeling/train_lstm_local.py
    python modeling/train_lstm_local.py --epochs 100 --hidden-size 128
"""

import os
import sys
import json
import glob
import copy
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SEQ       = os.path.join(_PROJECT_ROOT, "local_pipeline", "lstm", "sequences.parquet")
DEFAULT_ENCODERS  = os.path.join(_PROJECT_ROOT, "local_pipeline", "gold", "label_encoders.json")
DEFAULT_OUTPUT    = os.path.join(_PROJECT_ROOT, "local_pipeline", "lstm")

TARGET = "Avg_Mdcr_Alowd_Amt"
GROUP_KEYS = [
    "Rndrng_Prvdr_Type_idx",
    "hcpcs_bucket",
    "Rndrng_Prvdr_State_Abrvtn_idx",
]


# ---------------------------------------------------------------------------
# MLflow config (matches train_xgb_local.py)
# ---------------------------------------------------------------------------
def configure_databricks_mlflow() -> str:
    """Configure MLflow and return the current user's workspace home path."""
    import requests
    host  = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")
    if not host or not token:
        raise EnvironmentError(
            "DATABRICKS_HOST and DATABRICKS_TOKEN must be set.\n"
            "  export DATABRICKS_HOST=https://<workspace>.azuredatabricks.net\n"
            "  export DATABRICKS_TOKEN=<your-pat>"
        )
    mlflow.set_tracking_uri("databricks")
    resp = requests.get(
        f"{host}/api/2.0/preview/scim/v2/Me",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    resp.raise_for_status()
    username = resp.json().get("userName", "unknown")
    print(f"MLflow tracking URI -> Databricks: {host}  (user: {username})")
    return f"/Users/{username}"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MedicareLSTM(nn.Module):
    """LSTM with static-feature embeddings for Medicare cost forecasting."""

    def __init__(
        self,
        n_provider_types: int,
        n_states: int,
        n_hcpcs_buckets: int = 6,
        embed_dim: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
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
        """
        Args:
            target_seq:    (B, T) float32 — log1p-transformed values
            provider_type: (B,)   int64
            state:         (B,)   int64
            hcpcs_bucket:  (B,)   int64
        Returns:
            (B, T) float32 — predicted next-step values
        """
        B, T = target_seq.shape
        x = target_seq.unsqueeze(-1)                             # (B, T, 1)
        p = self.provider_embed(provider_type).unsqueeze(1).expand(-1, T, -1)
        s = self.state_embed(state).unsqueeze(1).expand(-1, T, -1)
        h = self.bucket_embed(hcpcs_bucket).unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([x, p, s, h], dim=-1)                     # (B, T, 1+3*E)
        out, _ = self.lstm(x)                                    # (B, T, H)
        out = self.head_dropout(out)
        return self.head(out).squeeze(-1)                        # (B, T)


# ---------------------------------------------------------------------------
# Dataset & collation
# ---------------------------------------------------------------------------
class MedicareSequenceDataset(Dataset):
    """Variable-length sequence dataset for teacher-forcing training."""

    def __init__(self, seqs, provider_types, states, buckets, year_lists=None):
        self.seqs           = seqs            # list of 1D np arrays (log1p)
        self.provider_types = provider_types  # 1D int array
        self.states         = states
        self.buckets        = buckets
        self.year_lists     = year_lists      # optional, for val_mask

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        item = {
            "input_seq":     torch.tensor(seq[:-1], dtype=torch.float32),
            "target_seq":    torch.tensor(seq[1:],  dtype=torch.float32),
            "provider_type": torch.tensor(self.provider_types[idx], dtype=torch.long),
            "state":         torch.tensor(self.states[idx],         dtype=torch.long),
            "hcpcs_bucket":  torch.tensor(self.buckets[idx],        dtype=torch.long),
            "seq_len":       len(seq) - 1,
        }
        if self.year_lists is not None:
            years = self.year_lists[idx]
            val_mask = [years[i + 1] > 2021 for i in range(len(seq) - 1)]
            item["val_mask"] = torch.tensor(val_mask, dtype=torch.bool)
        return item


def collate_sequences(batch):
    """Pad variable-length sequences and build masks."""
    max_len = max(item["seq_len"] for item in batch)
    B = len(batch)
    has_val_mask = "val_mask" in batch[0]

    input_seqs  = torch.zeros(B, max_len, dtype=torch.float32)
    target_seqs = torch.zeros(B, max_len, dtype=torch.float32)
    mask        = torch.zeros(B, max_len, dtype=torch.bool)
    val_masks   = torch.zeros(B, max_len, dtype=torch.bool) if has_val_mask else None

    ptypes  = torch.stack([item["provider_type"] for item in batch])
    states  = torch.stack([item["state"]         for item in batch])
    buckets = torch.stack([item["hcpcs_bucket"]  for item in batch])

    for i, item in enumerate(batch):
        L = item["seq_len"]
        input_seqs[i, :L]  = item["input_seq"]
        target_seqs[i, :L] = item["target_seq"]
        mask[i, :L]        = True
        if has_val_mask:
            val_masks[i, :L] = item["val_mask"]

    out = {
        "input_seq":     input_seqs,
        "target_seq":    target_seqs,
        "mask":          mask,
        "provider_type": ptypes,
        "state":         states,
        "hcpcs_bucket":  buckets,
    }
    if has_val_mask:
        out["val_mask"] = val_masks
    return out


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def masked_mse_loss(pred, target, mask):
    sq_diff = (pred - target) ** 2
    return (sq_diff * mask.float()).sum() / mask.float().sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Data loading & temporal split
# ---------------------------------------------------------------------------
def load_sequences(seq_path: str, min_years: int = 3) -> pd.DataFrame:
    df = pd.read_parquet(seq_path)
    before = len(df)
    df = df[df["n_years"] >= min_years].reset_index(drop=True)
    print(f"Loaded {before} groups, kept {len(df)} with >= {min_years} years")
    for col in GROUP_KEYS:
        df[col] = df[col].astype(int)
    return df


def temporal_split(df: pd.DataFrame, train_end_year: int = 2021):
    """Split sequences into train (years<=2021) and val (full seq, mask marks 2022+)."""
    train_seqs, train_ptypes, train_states, train_buckets = [], [], [], []
    val_seqs, val_ptypes, val_states, val_buckets, val_years = [], [], [], [], []

    for _, row in df.iterrows():
        years      = np.array(row["years"])
        target_seq = np.array(row["target_seq"])
        ptype  = row["Rndrng_Prvdr_Type_idx"]
        state  = row["Rndrng_Prvdr_State_Abrvtn_idx"]
        bucket = row["hcpcs_bucket"]

        # Training: only years <= train_end_year
        train_mask = years <= train_end_year
        t_years  = years[train_mask]
        t_target = target_seq[train_mask]
        if len(t_target) >= 2:  # need at least 1 teacher-forcing pair
            train_seqs.append(np.log1p(t_target))
            train_ptypes.append(ptype)
            train_states.append(state)
            train_buckets.append(bucket)

        # Validation: full sequence for groups with any year > train_end_year
        has_val = np.any(years > train_end_year)
        if has_val and len(target_seq) >= 3:
            val_seqs.append(np.log1p(target_seq))
            val_ptypes.append(ptype)
            val_states.append(state)
            val_buckets.append(bucket)
            val_years.append(years.tolist())

    print(f"Temporal split: {len(train_seqs)} train groups, {len(val_seqs)} val groups")
    train_data = (train_seqs, np.array(train_ptypes), np.array(train_states), np.array(train_buckets))
    val_data   = (val_seqs, np.array(val_ptypes), np.array(val_states), np.array(val_buckets), val_years)
    return train_data, val_data


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(model, train_dl, val_dl, device, n_epochs=50, lr=1e-3, patience=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    best_val_loss = float("inf")
    best_state    = None
    wait          = 0
    history       = {"train_loss": [], "val_loss": []}

    for epoch in range(1, n_epochs + 1):
        # --- Train ---
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for batch in train_dl:
            inp  = batch["input_seq"].to(device)
            tgt  = batch["target_seq"].to(device)
            msk  = batch["mask"].to(device)
            pt   = batch["provider_type"].to(device)
            st   = batch["state"].to(device)
            bk   = batch["hcpcs_bucket"].to(device)

            pred = model(inp, pt, st, bk)
            loss = masked_mse_loss(pred, tgt, msk)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * msk.float().sum().item()
            train_n += msk.float().sum().item()

        avg_train = train_loss_sum / max(train_n, 1)
        history["train_loss"].append(avg_train)

        # --- Validate ---
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                inp = batch["input_seq"].to(device)
                tgt = batch["target_seq"].to(device)
                msk = batch["mask"].to(device)
                vmsk = batch["val_mask"].to(device)
                pt  = batch["provider_type"].to(device)
                st  = batch["state"].to(device)
                bk  = batch["hcpcs_bucket"].to(device)

                pred = model(inp, pt, st, bk)
                eff_mask = msk & vmsk
                loss = masked_mse_loss(pred, tgt, eff_mask)
                val_loss_sum += loss.item() * eff_mask.float().sum().item()
                val_n += eff_mask.float().sum().item()

        avg_val = val_loss_sum / max(val_n, 1)
        history["val_loss"].append(avg_val)
        scheduler.step(avg_val)
        cur_lr = optimizer.param_groups[0]["lr"]

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train_loss={avg_train:.6f} | val_loss={avg_val:.6f} | lr={cur_lr:.2e}")

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    return history, best_state, epoch


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, val_dl, device):
    """Compute dollar-scale metrics on validation set (2022-2023 predictions)."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in val_dl:
            inp  = batch["input_seq"].to(device)
            tgt  = batch["target_seq"].to(device)
            msk  = batch["mask"].to(device)
            vmsk = batch["val_mask"].to(device)
            pt   = batch["provider_type"].to(device)
            st   = batch["state"].to(device)
            bk   = batch["hcpcs_bucket"].to(device)

            pred = model(inp, pt, st, bk)
            eff_mask = msk & vmsk
            all_preds.append(pred[eff_mask].cpu().numpy())
            all_targets.append(tgt[eff_mask].cpu().numpy())

    preds   = np.expm1(np.concatenate(all_preds))
    targets = np.expm1(np.concatenate(all_targets))
    preds   = np.clip(preds, 0, None)  # allowed amount can't be negative

    mae  = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2   = r2_score(targets, preds)
    print(f"\n  Validation (2022-2023):")
    print(f"    MAE  = ${mae:,.2f}")
    print(f"    RMSE = ${rmse:,.2f}")
    print(f"    R2   = {r2:.4f}")
    return {"test_mae": mae, "test_rmse": rmse, "test_r2": r2}


# ---------------------------------------------------------------------------
# Forecast 2024-2026 with MC Dropout
# ---------------------------------------------------------------------------
def generate_forecasts(model, df, device, n_forward=3, n_mc_samples=50, batch_size=256):
    """Autoregressive forecast with MC Dropout for confidence bounds."""
    print(f"\nGenerating {n_forward}-year forecasts ({n_mc_samples} MC samples)...")
    model.train()  # keep dropout active for MC sampling

    all_seqs    = [np.log1p(np.array(row["target_seq"])) for _, row in df.iterrows()]
    all_ptypes  = df["Rndrng_Prvdr_Type_idx"].values
    all_states  = df["Rndrng_Prvdr_State_Abrvtn_idx"].values
    all_buckets = df["hcpcs_bucket"].values
    all_years   = [row["years"] for _, row in df.iterrows()]

    n_groups = len(all_seqs)
    # mc_samples: (n_groups, n_mc_samples, n_forward)
    mc_samples = np.zeros((n_groups, n_mc_samples, n_forward), dtype=np.float64)

    with torch.no_grad():
        for mc in range(n_mc_samples):
            for start in range(0, n_groups, batch_size):
                end = min(start + batch_size, n_groups)
                batch_seqs = all_seqs[start:end]
                pt = torch.tensor(all_ptypes[start:end], dtype=torch.long, device=device)
                st = torch.tensor(all_states[start:end], dtype=torch.long, device=device)
                bk = torch.tensor(all_buckets[start:end], dtype=torch.long, device=device)

                # Pad sequences
                lengths = [len(s) for s in batch_seqs]
                max_len = max(lengths)
                padded = np.zeros((end - start, max_len), dtype=np.float32)
                for i, s in enumerate(batch_seqs):
                    padded[i, :len(s)] = s

                cur_seqs = torch.tensor(padded, dtype=torch.float32, device=device)
                cur_lens = list(lengths)

                for step in range(n_forward):
                    pred = model(cur_seqs, pt, st, bk)  # (B, T)
                    # Extract prediction at last valid position for each group
                    next_vals = []
                    for i, L in enumerate(cur_lens):
                        next_vals.append(pred[i, L - 1].item())
                    next_vals = torch.tensor(next_vals, dtype=torch.float32, device=device)

                    for i in range(end - start):
                        mc_samples[start + i, mc, step] = next_vals[i].item()

                    # Extend sequences for next autoregressive step
                    new_col = next_vals.unsqueeze(-1)
                    cur_seqs = torch.cat([cur_seqs, new_col], dim=1)
                    cur_lens = [L + 1 for L in cur_lens]

            if (mc + 1) % 10 == 0:
                print(f"  MC sample {mc + 1}/{n_mc_samples}")

    # Invert log1p and build output DataFrame
    mc_dollars = np.expm1(mc_samples)
    mc_dollars = np.clip(mc_dollars, 0, None)

    rows = []
    for i in range(n_groups):
        last_year  = max(all_years[i])
        last_value = np.expm1(all_seqs[i][-1])
        for step in range(n_forward):
            samples = mc_dollars[i, :, step]
            rows.append({
                "Rndrng_Prvdr_Type_idx":          float(all_ptypes[i]),
                "hcpcs_bucket":                   float(all_buckets[i]),
                "Rndrng_Prvdr_State_Abrvtn_idx":  float(all_states[i]),
                "forecast_year":                  2024 + step,
                "forecast_mean":                  float(np.mean(samples)),
                "forecast_std":                   float(np.std(samples)),
                "forecast_p10":                   float(np.percentile(samples, 10)),
                "forecast_p50":                   float(np.median(samples)),
                "forecast_p90":                   float(np.percentile(samples, 90)),
                "last_known_year":                int(last_year),
                "last_known_value":               float(last_value),
                "n_history_years":                len(all_years[i]),
            })

    forecast_df = pd.DataFrame(rows)
    print(f"  Forecast rows: {len(forecast_df):,} ({n_groups} groups x {n_forward} years)")
    return forecast_df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_specialty_trends(df, forecast_df, label_encoders, output_dir, top_n=12):
    """Plot historical + forecast trends by specialty."""
    os.makedirs(output_dir, exist_ok=True)
    ptype_names = label_encoders.get("Rndrng_Prvdr_Type", [])
    if isinstance(ptype_names, list):
        inv_ptype = {i: name for i, name in enumerate(ptype_names)}
    else:
        inv_ptype = {int(v): k for k, v in ptype_names.items()}

    # Count groups per specialty for top_n selection
    specialty_counts = df["Rndrng_Prvdr_Type_idx"].value_counts().head(top_n)
    top_specialties = specialty_counts.index.tolist()

    # --- Plot 1: Specialty trend grid ---
    n_cols, n_rows = 4, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))
    axes = axes.flatten()

    for ax_idx, ptype_idx in enumerate(top_specialties):
        ax = axes[ax_idx]
        specialty_name = inv_ptype.get(ptype_idx, f"Type {ptype_idx}")

        # Historical: aggregate across states/buckets
        sub = df[df["Rndrng_Prvdr_Type_idx"] == ptype_idx]
        hist_data = {}
        for _, row in sub.iterrows():
            for yr, val in zip(row["years"], row["target_seq"]):
                hist_data.setdefault(yr, []).append(val)
        hist_years   = sorted(hist_data.keys())
        hist_means   = [np.mean(hist_data[y]) for y in hist_years]

        # Forecast: aggregate
        fsub = forecast_df[forecast_df["Rndrng_Prvdr_Type_idx"] == float(ptype_idx)]
        if not fsub.empty:
            f_years = sorted(fsub["forecast_year"].unique())
            f_means = [fsub[fsub["forecast_year"] == y]["forecast_mean"].mean() for y in f_years]
            f_p10   = [fsub[fsub["forecast_year"] == y]["forecast_p10"].mean() for y in f_years]
            f_p90   = [fsub[fsub["forecast_year"] == y]["forecast_p90"].mean() for y in f_years]

            ax.plot(f_years, f_means, "r--", linewidth=2, label="Forecast")
            ax.fill_between(f_years, f_p10, f_p90, alpha=0.2, color="red", label="P10-P90")

        ax.plot(hist_years, hist_means, "b-o", markersize=3, linewidth=1.5, label="Historical")
        ax.set_title(specialty_name, fontsize=10)
        ax.set_xlabel("Year", fontsize=8)
        ax.set_ylabel("Mean Allowed Amt ($)", fontsize=8)
        ax.tick_params(labelsize=7)
        if ax_idx == 0:
            ax.legend(fontsize=7)

    for ax_idx in range(len(top_specialties), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle("Medicare Allowed Amount Trends by Specialty (Top 12)", fontsize=14)
    plt.tight_layout()
    path1 = os.path.join(output_dir, "specialty_trends.png")
    fig.savefig(path1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path1}")

    # --- Plot 2: Forecast distribution (2026) ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    f2026 = forecast_df[forecast_df["forecast_year"] == 2026]["forecast_mean"]
    ax2.hist(f2026.clip(upper=f2026.quantile(0.99)), bins=80, color="steelblue", edgecolor="white")
    ax2.set_xlabel("Forecast Mean Allowed Amount ($)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Distribution of 2026 Forecast Means Across Groups", fontsize=13)
    path2 = os.path.join(output_dir, "forecast_distribution.png")
    fig2.savefig(path2, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {path2}")

    # --- Plot 3: Top growth specialties ---
    growth = []
    for ptype_idx in df["Rndrng_Prvdr_Type_idx"].unique():
        specialty_name = inv_ptype.get(ptype_idx, f"Type {ptype_idx}")
        sub = df[df["Rndrng_Prvdr_Type_idx"] == ptype_idx]
        last_vals = []
        for _, row in sub.iterrows():
            years_arr = np.asarray(row["years"])
            if 2023 in years_arr:
                idx_2023 = int(np.where(years_arr == 2023)[0][0])
                last_vals.append(row["target_seq"][idx_2023])
        if not last_vals:
            continue
        hist_2023 = np.mean(last_vals)

        fsub = forecast_df[
            (forecast_df["Rndrng_Prvdr_Type_idx"] == float(ptype_idx))
            & (forecast_df["forecast_year"] == 2026)
        ]
        if fsub.empty or hist_2023 < 1:
            continue
        f2026_mean = fsub["forecast_mean"].mean()
        pct = (f2026_mean - hist_2023) / hist_2023 * 100
        growth.append({"specialty": specialty_name, "growth_pct": pct})

    if growth:
        gdf = pd.DataFrame(growth).sort_values("growth_pct", ascending=False).head(15)
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        ax3.barh(gdf["specialty"], gdf["growth_pct"], color="coral", edgecolor="white")
        ax3.set_xlabel("Projected Growth 2023-2026 (%)", fontsize=11)
        ax3.set_title("Top 15 Specialties by Projected Cost Growth", fontsize=13)
        ax3.invert_yaxis()
        path3 = os.path.join(output_dir, "top_growth_specialties.png")
        fig3.savefig(path3, dpi=120, bbox_inches="tight")
        plt.close(fig3)
        print(f"  Saved: {path3}")

    return [p for p in [path1, path2, os.path.join(output_dir, "top_growth_specialties.png")]
            if os.path.exists(p)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load sequences
    df = load_sequences(args.data, min_years=args.min_years)

    # Load label encoders for embedding dims + visualization
    with open(args.label_encoders) as f:
        label_encoders = json.load(f)
    n_provider_types = len(label_encoders["Rndrng_Prvdr_Type"])
    n_states         = len(label_encoders["Rndrng_Prvdr_State_Abrvtn"])

    # Temporal split
    train_data, val_data = temporal_split(df, train_end_year=2021)
    train_seqs, train_pt, train_st, train_bk = train_data
    val_seqs, val_pt, val_st, val_bk, val_yrs = val_data

    # Datasets & dataloaders
    train_ds = MedicareSequenceDataset(train_seqs, train_pt, train_st, train_bk)
    val_ds   = MedicareSequenceDataset(val_seqs, val_pt, val_st, val_bk, year_lists=val_yrs)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_sequences, num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                          collate_fn=collate_sequences, num_workers=0, pin_memory=True)

    # Model
    model = MedicareLSTM(
        n_provider_types=n_provider_types,
        n_states=n_states,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Train
    print("\n=== Training ===")
    history, best_state, final_epoch = train_model(
        model, train_dl, val_dl, device,
        n_epochs=args.epochs, lr=args.lr, patience=args.patience,
    )
    model.load_state_dict(best_state)

    # Evaluate
    print("\n=== Evaluation ===")
    metrics = evaluate(model, val_dl, device)

    # Forecast
    print("\n=== Forecasting 2024-2026 ===")
    forecast_df = generate_forecasts(
        model, df, device,
        n_forward=3, n_mc_samples=args.n_mc_samples, batch_size=args.batch_size,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    forecast_path = os.path.join(args.output_dir, "forecast_2024_2026.parquet")
    forecast_df.to_parquet(forecast_path, index=False)
    print(f"Forecast saved -> {forecast_path}")

    # Visualize
    print("\n=== Visualization ===")
    plot_dir = os.path.join(args.output_dir, "plots")
    plot_paths = plot_specialty_trends(df, forecast_df, label_encoders, plot_dir)

    # MLflow
    print("\n=== MLflow Logging ===")
    user_home = configure_databricks_mlflow()
    mlflow.set_experiment(f"{user_home}/medicare_models")

    with mlflow.start_run(run_name="lstm_local"):
        mlflow.log_params({
            "model":           "LSTM",
            "hidden_size":     args.hidden_size,
            "num_layers":      args.num_layers,
            "embed_dim":       args.embed_dim,
            "dropout":         args.dropout,
            "n_epochs_trained": final_epoch,
            "batch_size":      args.batch_size,
            "learning_rate":   args.lr,
            "weight_decay":    1e-4,
            "optimizer":       "AdamW",
            "scheduler":       "ReduceLROnPlateau",
            "patience":        args.patience,
            "target_transform": "log1p",
            "min_seq_length":  args.min_years,
            "train_end_year":  2021,
            "n_train_groups":  len(train_seqs),
            "n_val_groups":    len(val_seqs),
            "n_mc_samples":    args.n_mc_samples,
            "n_params":        n_params,
            "device":          str(device),
            "source":          "local",
        })
        mlflow.log_metrics({
            "test_mae":  metrics["test_mae"],
            "test_rmse": metrics["test_rmse"],
            "test_r2":   metrics["test_r2"],
        })
        mlflow.pytorch.log_model(model, artifact_path="lstm_model")
        mlflow.log_artifact(forecast_path, artifact_path="forecasts")
        for p in plot_paths:
            mlflow.log_artifact(p, artifact_path="plots")
        mlflow.log_dict(history, "training_history.json")
        print("  MLflow run logged.")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LSTM time-series forecasting for Medicare allowed amounts"
    )
    parser.add_argument("--data", default=DEFAULT_SEQ,
                        help="Path to sequences.parquet")
    parser.add_argument("--label-encoders", default=DEFAULT_ENCODERS,
                        help="Path to label_encoders.json")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT,
                        help="Output directory for forecasts and plots")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--hidden-size", type=int, default=64,
                        help="LSTM hidden dimension (default: 64)")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of LSTM layers (default: 2)")
    parser.add_argument("--embed-dim", type=int, default=8,
                        help="Embedding dimension for categorical features (default: 8)")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate (default: 0.2)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (default: 10)")
    parser.add_argument("--n-mc-samples", type=int, default=50,
                        help="MC Dropout samples for forecast confidence (default: 50)")
    parser.add_argument("--min-years", type=int, default=3,
                        help="Minimum sequence length to include (default: 3)")
    args = parser.parse_args()
    main(args)
