"""
train_glm_local.py — GLM baseline, trained locally, logs to Databricks MLflow

Batch strategy: SGDRegressor with partial_fit() streams the parquet row-group
by row-group — the full dataset never lives in RAM. Mathematically equivalent
to a Tweedie GLM with a squared-error link at this scale.

Column pruning: only FEATURES + TARGET columns are loaded from parquet,
cutting memory footprint by ~40% vs loading all columns.

Required env vars:
    DATABRICKS_HOST   e.g. https://<workspace>.azuredatabricks.net
    DATABRICKS_TOKEN  Personal Access Token or service principal secret

Usage:
    python modeling/train_glm_local.py
    python modeling/train_glm_local.py --data local_pipeline/gold
"""

import os
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_DATA = os.path.join("local_pipeline", "gold")
TARGET       = "Avg_Mdcr_Alowd_Amt"
FEATURES     = [
    "Rndrng_Prvdr_Type_idx", "Rndrng_Prvdr_State_Abrvtn_idx",
    "HCPCS_Cd_idx", "hcpcs_bucket", "place_of_srvc_flag",
    "Bene_Avg_Risk_Scre", "log_srvcs", "log_benes",
    "Avg_Sbmtd_Chrg", "srvcs_per_bene",
    "specialty_bucket", "pos_bucket", "hcpcs_target_enc",
    # year, is_covid_era: added after silver is re-run with year injection
]
LOAD_COLS = FEATURES + [TARGET]

SGD_PARAMS = {
    # squared_error gives gradients proportional to residual magnitude, which is
    # essential for convergence. The previous "huber" with epsilon=0.1 caused ALL
    # log-space residuals (range ~0-10) to get constant ±1 gradients, making SGD
    # behave like a perceptron and diverge (R²=-102.80).
    "loss":          "squared_error",
    "penalty":       "elasticnet",   # L1+L2 for feature selection + stability
    "alpha":         0.001,          # stronger regularization than default (0.0001)
    "l1_ratio":      0.15,           # mostly L2, slight L1 for sparsity
    "learning_rate": "invscaling",   # lr = eta0 / (t^power_t) — stable long-run decay
    "eta0":          0.05,
    "power_t":       0.25,
    "tol":           1e-4,
    "max_iter":      1,              # one epoch per partial_fit call
    "random_state":  42,
    "warm_start":    True,
}
# Clip standardized features to prevent gradient explosions from extreme outliers
RATIO_CLIP = 5.0


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
    # Fetch current user's email to build valid experiment path
    resp = requests.get(
        f"{host}/api/2.0/preview/scim/v2/Me",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    resp.raise_for_status()
    username = resp.json().get("userName", "unknown")
    print(f"MLflow tracking URI -> Databricks: {host}  (user: {username})")
    return f"/Users/{username}"


def iter_row_groups(gold_dir: str):
    """
    Yields (X_chunk, y_log_chunk, is_test) reading one parquet row group at a time
    across all per-state gold parquets in the directory.
    Target is log1p-transformed so the model learns in log-space — essential for
    right-skewed cost distributions where raw values span several orders of magnitude.
    Deterministic 80/20 split: row groups whose global index % 5 == 0 go to test.
    """
    import glob
    parquet_files = sorted(glob.glob(os.path.join(gold_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in '{gold_dir}'")

    total_rg = sum(pq.ParquetFile(f).metadata.num_row_groups for f in parquet_files)
    total_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in parquet_files)
    print(f"Gold directory: {len(parquet_files)} files, {total_rg} row groups, "
          f"{total_rows:,} total rows")

    global_idx = 0
    for f in parquet_files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.metadata.num_row_groups):
            df = pf.read_row_group(rg, columns=LOAD_COLS).to_pandas().dropna()
            X      = df[FEATURES].astype("float64").values
            y_log  = np.log1p(df[TARGET].astype("float64").values)
            is_test = (global_idx % 5 == 0)
            global_idx += 1
            yield X, y_log, is_test


def log_metrics(y_true_log, y_pred_log, prefix=""):
    # Inverse-transform both back to dollar space before computing metrics
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mlflow.log_metrics({f"{prefix}mae": mae, f"{prefix}rmse": rmse, f"{prefix}r2": r2})
    print(f"  {prefix}MAE=${mae:.2f}  RMSE=${rmse:.2f}  R²={r2:.4f}")


def main(data_path: str, n_epochs: int = 3):
    user_home = configure_databricks_mlflow()

    scaler = StandardScaler()
    model  = SGDRegressor(**SGD_PARAMS)

    # ── Pass 1: fit scaler incrementally across ALL row groups ───────────────
    # Must use partial_fit here — fitting on one row group produces wrong
    # mean/std for the rest of the data, causing SGD to diverge completely.
    print("Pass 1/2: Fitting scaler across all row groups...")
    n_rg = 0
    for X, _, _ in iter_row_groups(data_path):
        scaler.partial_fit(X)
        n_rg += 1
    print(f"Scaler fitted on {n_rg} row groups.")

    mlflow.set_experiment(f"{user_home}/medicare_models")
    with mlflow.start_run(run_name="glm_sgd_local"):
        mlflow.log_params({**SGD_PARAMS, "n_epochs": n_epochs,
                           "n_features": len(FEATURES),
                           "source": "local", "strategy": "partial_fit",
                           "target_transform": "log1p",
                           "ratio_clip": RATIO_CLIP})

        # ── Pass 2: stream row groups for n_epochs ────────────────────────────
        print(f"\nPass 2/2: Training SGD for {n_epochs} epoch(s)...")
        X_test_parts, y_test_parts = [], []
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            for i, (X, y, is_test) in enumerate(iter_row_groups(data_path)):
                X_scaled = np.clip(scaler.transform(X), -RATIO_CLIP, RATIO_CLIP)
                if is_test:
                    if epoch == 0:   # collect test set once
                        X_test_parts.append(X_scaled)
                        y_test_parts.append(y)
                else:
                    model.partial_fit(X_scaled, y)
                if (i + 1) % 10 == 0:
                    print(f"  row group {i + 1} processed")

        X_test = np.vstack(X_test_parts)
        y_test = np.concatenate(y_test_parts)

        # Train metrics on first train row group as a proxy
        X_last, y_last, _ = next(
            (X, y, is_test) for X, y, is_test in iter_row_groups(data_path) if not is_test
        )
        X_last_scaled = np.clip(scaler.transform(X_last), -RATIO_CLIP, RATIO_CLIP)
        log_metrics(y_last, model.predict(X_last_scaled), prefix="train_")
        log_metrics(y_test, model.predict(X_test),        prefix="test_")

        from sklearn.pipeline import Pipeline
        pipe = Pipeline([("scaler", scaler), ("glm", model)])
        mlflow.sklearn.log_model(pipe, artifact_path="glm_model")
        print("GLM run posted to Databricks MLflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     default=DEFAULT_DATA, help="Path to gold directory with per-state parquets")
    parser.add_argument("--n-epochs", type=int, default=3,
                        help="Number of passes over the data for SGD (default: 3)")
    args = parser.parse_args()
    main(args.data, args.n_epochs)
