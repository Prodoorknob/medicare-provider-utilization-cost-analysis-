"""
train_xgb_local.py — XGBoost, trained locally, logs to Databricks MLflow

Two training modes:
  --mode batch  (default): Incremental training by Census region. Trains on
                4 regions sequentially, passing the booster forward via
                xgb.train(xgb_model=prev_booster). Keeps VRAM low.
  --mode full:  Loads all gold parquets into a single DMatrix and trains
                in one shot (original behavior, needs more RAM).

GPU support: auto-detects CUDA via a probe DMatrix.

Required env vars:
    DATABRICKS_HOST   e.g. https://<workspace>.azuredatabricks.net
    DATABRICKS_TOKEN  Personal Access Token or service principal secret

Usage:
    python modeling/train_xgb_local.py --mode batch
    python modeling/train_xgb_local.py --mode full --sample 0.5
"""

import os
import glob
import argparse
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_DATA = os.path.join("local_pipeline", "gold")
TARGET       = "Avg_Mdcr_Alowd_Amt"
FEATURES     = [
    "Rndrng_Prvdr_Type_idx", "Rndrng_Prvdr_State_Abrvtn_idx",
    "HCPCS_Cd_idx", "hcpcs_bucket", "place_of_srvc_flag",
    "Bene_Avg_Risk_Scre", "log_srvcs", "log_benes",
    "Avg_Sbmtd_Chrg", "srvcs_per_bene",
]
LOAD_COLS = FEATURES + [TARGET]

CENSUS_REGIONS = {
    "NORTHEAST":   ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
    "SOUTH":       ["DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV",
                    "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"],
    "MIDWEST":     ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO",
                    "NE", "ND", "SD"],
    "WEST":        ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY",
                    "AK", "CA", "HI", "OR", "WA"],
    "TERRITORIES": ["AA", "AE", "AP", "AS", "FM", "GU", "MP", "PR", "PW", "VI"],
}
ROUNDS_PER_REGION = 125   # ~500 total across 4 regions + territories


def _detect_device() -> str:
    """Use GPU if XGBoost was built with CUDA support, else CPU."""
    try:
        _d = xgb.DMatrix(np.zeros((1, 1)))
        xgb.train({"device": "cuda", "verbosity": 0}, _d, num_boost_round=1)
        return "cuda"
    except Exception:
        return "cpu"

XGB_PARAMS = {
    "objective":        "reg:squarederror",
    "learning_rate":    0.05,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "tree_method":      "hist",
    "device":           _detect_device(),
    "seed":             42,
}


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
    print(f"MLflow tracking URI → Databricks: {host}  (user: {username})")
    return f"/Users/{username}"


def _list_gold_files(gold_dir: str) -> dict[str, str]:
    """Return {STATE: filepath} mapping from gold directory."""
    files = {}
    for f in sorted(glob.glob(os.path.join(gold_dir, "*.parquet"))):
        state = os.path.splitext(os.path.basename(f))[0]
        files[state] = f
    return files


def _load_region(gold_files: dict[str, str], states: list[str], sample: float = 1.0):
    """Load and concat gold parquets for a list of states, column-pruned."""
    parts = []
    for st in states:
        if st in gold_files:
            df = pd.read_parquet(gold_files[st], columns=LOAD_COLS).dropna()
            parts.append(df)
    if not parts:
        return None, None
    df = pd.concat(parts, ignore_index=True)
    if sample < 1.0:
        df = df.sample(frac=sample, random_state=42)
    X = df[FEATURES].astype("float64").values
    y = np.log1p(df[TARGET].astype("float64").values)
    return X, y


def log_metrics(y_true_log, y_pred_log, prefix=""):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mlflow.log_metrics({f"{prefix}mae": mae, f"{prefix}rmse": rmse, f"{prefix}r2": r2})
    print(f"  {prefix}MAE=${mae:.2f}  RMSE=${rmse:.2f}  R²={r2:.4f}")


# ── Batch mode: incremental training by Census region ────────────────────────

def train_batch(gold_dir: str, sample: float):
    user_home  = configure_databricks_mlflow()
    gold_files = _list_gold_files(gold_dir)
    print(f"Found {len(gold_files)} gold state parquets in '{gold_dir}'")
    print(f"Device: {XGB_PARAMS['device']}")

    # Accumulate validation data across all regions
    X_val_parts, y_val_parts = [], []
    booster = None
    total_rounds = 0
    regions_trained = 0

    mlflow.set_experiment(f"{user_home}/medicare_models")
    with mlflow.start_run(run_name="xgb_extmem_local"):
        mlflow.log_params({
            **XGB_PARAMS,
            "rounds_per_region": ROUNDS_PER_REGION,
            "n_regions": len(CENSUS_REGIONS),
            "source": "local",
            "strategy": "incremental_region_batch",
            "target_transform": "log1p",
            "sample_frac": sample,
        })

        for region_name, states in CENSUS_REGIONS.items():
            X, y = _load_region(gold_files, states, sample)
            if X is None:
                print(f"\n  [{region_name}] No data found — skipping")
                continue

            # 80/20 split within region
            n_val  = max(1, int(len(y) * 0.2))
            idx    = np.random.RandomState(42).permutation(len(y))
            val_idx, train_idx = idx[:n_val], idx[n_val:]

            X_val_parts.append(X[val_idx])
            y_val_parts.append(y[val_idx])

            dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx], feature_names=FEATURES)
            dval   = xgb.DMatrix(X[val_idx],   label=y[val_idx],   feature_names=FEATURES)

            booster = xgb.train(
                XGB_PARAMS,
                dtrain,
                num_boost_round=ROUNDS_PER_REGION,
                evals=[(dval, "val")],
                xgb_model=booster,
                verbose_eval=50,
            )
            total_rounds += ROUNDS_PER_REGION
            regions_trained += 1
            print(f"  [{region_name}] {len(train_idx):,} train / {n_val:,} val — "
                  f"cumulative rounds: {total_rounds}")

        # Final evaluation on aggregated validation set
        if not X_val_parts:
            print("No validation data collected — aborting.")
            return

        X_val_all = np.vstack(X_val_parts)
        y_val_all = np.concatenate(y_val_parts)
        dval_all  = xgb.DMatrix(X_val_all, label=y_val_all, feature_names=FEATURES)

        mlflow.log_params({"total_rounds": total_rounds, "regions_trained": regions_trained})
        y_pred = booster.predict(dval_all)
        log_metrics(y_val_all, y_pred, prefix="test_")

        importances = booster.get_score(importance_type="gain")
        mlflow.log_dict(importances, "feature_importances.json")
        print("\nTop feature importances (gain):")
        for feat, score in sorted(importances.items(), key=lambda x: -x[1])[:10]:
            print(f"  {feat}: {score:.2f}")

        mlflow.xgboost.log_model(booster, artifact_path="xgb_model")
        print(f"XGBoost batch run complete — {regions_trained} regions, "
              f"{total_rounds} total rounds. Model logged to MLflow.")


# ── Full mode: load everything into one DMatrix ──────────────────────────────

def train_full(gold_dir: str, sample: float):
    user_home  = configure_databricks_mlflow()
    gold_files = _list_gold_files(gold_dir)
    print(f"Found {len(gold_files)} gold state parquets in '{gold_dir}'")
    print(f"Device: {XGB_PARAMS['device']}")

    X, y = _load_region(gold_files, list(gold_files.keys()), sample)
    if X is None:
        raise RuntimeError("No gold data found.")

    # 80/20 split
    n_val  = int(len(y) * 0.2)
    idx    = np.random.RandomState(42).permutation(len(y))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx], feature_names=FEATURES)
    dval   = xgb.DMatrix(X[val_idx],   label=y[val_idx],   feature_names=FEATURES)

    print(f"Train: {len(train_idx):,} rows  |  Val: {n_val:,} rows")

    mlflow.set_experiment(f"{user_home}/medicare_models")
    with mlflow.start_run(run_name="xgb_extmem_local"):
        n_rounds = 500
        mlflow.log_params({
            **XGB_PARAMS,
            "n_rounds": n_rounds,
            "early_stopping": 30,
            "source": "local",
            "strategy": "full_dmatrix",
            "target_transform": "log1p",
            "sample_frac": sample,
        })

        evals_result = {}
        booster = xgb.train(
            XGB_PARAMS, dtrain,
            num_boost_round=n_rounds,
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            evals_result=evals_result,
            verbose_eval=50,
        )

        mlflow.log_param("best_iteration", booster.best_iteration)
        y_pred = booster.predict(dval)
        log_metrics(y[val_idx], y_pred, prefix="test_")

        importances = booster.get_score(importance_type="gain")
        mlflow.log_dict(importances, "feature_importances.json")
        mlflow.xgboost.log_model(booster, artifact_path="xgb_model")
        print("XGBoost full run complete. Model logged to MLflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="Path to gold directory with per-state parquets")
    parser.add_argument("--mode", choices=["batch", "full"], default="batch",
                        help="Training mode: 'batch' (incremental by region) or 'full'")
    parser.add_argument("--sample", type=float, default=0.3,
                        help="Fraction of rows to load per region/globally (default: 0.3)")
    args = parser.parse_args()

    if args.mode == "batch":
        train_batch(args.data, args.sample)
    else:
        train_full(args.data, args.sample)
