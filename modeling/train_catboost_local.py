"""
train_catboost_local.py -- CatBoost, trained locally, logs to Databricks MLflow

Key advantage over XGBoost: native categorical feature handling via ordered
target statistics — no label encoding needed for high-cardinality features
like HCPCS codes (~6K unique). CatBoost builds per-category statistics
using a random permutation ordering that prevents target leakage.

Two training modes:
  --mode batch  (default): Incremental training by Census region. Trains on
                4 regions sequentially, passing the model forward via
                init_model parameter. Keeps RAM low.
  --mode full:  Loads all gold parquets into a single Pool and trains
                in one shot (needs more RAM).

GPU support: auto-detects CUDA via catboost.utils.get_gpu_device_count().

Required env vars:
    DATABRICKS_HOST   e.g. https://<workspace>.azuredatabricks.net
    DATABRICKS_TOKEN  Personal Access Token or service principal secret

Usage:
    python modeling/train_catboost_local.py --mode batch
    python modeling/train_catboost_local.py --mode full --sample 0.5
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Fix Windows console encoding for MLflow emoji output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import mlflow
import mlflow.catboost
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_DATA = os.path.join("local_pipeline", "gold")
TARGET       = "Avg_Mdcr_Alowd_Amt"

# Features — same as XGB/RF but CatBoost handles categoricals natively
FEATURES = [
    "Rndrng_Prvdr_Type_idx", "Rndrng_Prvdr_State_Abrvtn_idx",
    "HCPCS_Cd_idx", "hcpcs_bucket", "place_of_srvc_flag",
    "Bene_Avg_Risk_Scre", "log_srvcs", "log_benes",
    "Avg_Sbmtd_Chrg", "srvcs_per_bene",
    "specialty_bucket", "pos_bucket", "hcpcs_target_enc",
    # year, is_covid_era: added after silver is re-run with year injection
]

# Indices of categorical features in the FEATURES list.
# CatBoost will compute ordered target statistics for these instead of
# treating them as continuous values (which XGB/RF do).
CAT_FEATURE_NAMES = [
    "Rndrng_Prvdr_Type_idx",        # ~100 specialties
    "Rndrng_Prvdr_State_Abrvtn_idx", # ~63 states/territories
    "HCPCS_Cd_idx",                  # ~6K procedure codes — biggest win
    "hcpcs_bucket",                  # 6 clinical categories
    "place_of_srvc_flag",            # binary
    "specialty_bucket",              # interaction feature
    "pos_bucket",                    # interaction feature
]

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
ITERS_PER_REGION = 125   # ~500-625 total across regions


def _detect_device() -> str:
    """Use GPU if CUDA devices are available, else CPU."""
    try:
        from catboost.utils import get_gpu_device_count
        if get_gpu_device_count() > 0:
            return "GPU"
    except Exception:
        pass
    return "CPU"


DEVICE = _detect_device()

CB_PARAMS = {
    "loss_function":     "RMSE",
    "learning_rate":     0.05,
    "depth":             6,
    "subsample":         0.8,
    "rsm":               0.8,           # colsample_bytree equivalent
    "task_type":         DEVICE,
    "random_seed":       42,
    "verbose":           50,
    "allow_writing_files": False,        # no tmp files on disk
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
    print(f"MLflow tracking URI -> Databricks: {host}  (user: {username})")
    return f"/Users/{username}"


def _list_gold_files(gold_dir: str) -> dict[str, str]:
    """Return {STATE: filepath} mapping from gold directory."""
    files = {}
    for f in sorted(glob.glob(os.path.join(gold_dir, "*.parquet"))):
        state = os.path.splitext(os.path.basename(f))[0]
        files[state] = f
    return files


def _load_region(
    gold_files: dict[str, str],
    states: list[str],
    active_features: list[str],
    sample: float = 1.0,
):
    """
    Load and concat gold parquets for a list of states, column-pruned.
    Returns (df, year_array) where df has active_features + TARGET columns.
    CatBoost needs the raw DataFrame (not numpy) for categorical handling.
    """
    load_cols = list(dict.fromkeys(active_features + [TARGET, "year"]))
    parts = []
    for st in states:
        if st in gold_files:
            avail = set(pq.read_schema(gold_files[st]).names)
            cols  = [c for c in load_cols if c in avail]
            df = pd.read_parquet(gold_files[st], columns=cols).dropna(
                subset=[c for c in active_features + [TARGET] if c in cols]
            )
            parts.append(df)
    if not parts:
        return None, None
    df = pd.concat(parts, ignore_index=True)
    if sample < 1.0:
        df = df.sample(frac=sample, random_state=42)
    year = df["year"].values if "year" in df.columns else None
    return df, year


def _temporal_split(df, year):
    """Train on years <= 2021, test on years >= 2022."""
    if year is None:
        raise ValueError("year column not available for temporal split")
    train_mask = year <= 2021
    test_mask  = year >= 2022
    n_train, n_test = train_mask.sum(), test_mask.sum()
    if n_test == 0:
        raise ValueError("No test records with year >= 2022. Check gold data contains 2022-2023.")
    print(f"  Temporal split: {n_train:,} train (<=2021) / {n_test:,} test (>=2022)")
    return df[train_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


def _make_pool(df, active_features, cat_indices):
    """Create a CatBoost Pool with proper categorical feature declarations."""
    X = df[active_features].copy()
    y = np.log1p(df[TARGET].astype("float64").values)
    # Cast categorical columns to int for CatBoost
    for idx in cat_indices:
        col = active_features[idx]
        X[col] = X[col].astype(int)
    return Pool(X, label=y, cat_features=cat_indices, feature_names=active_features)


def log_metrics(y_true_log, y_pred_log, prefix=""):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mlflow.log_metrics({f"{prefix}mae": mae, f"{prefix}rmse": rmse, f"{prefix}r2": r2})
    print(f"  {prefix}MAE=${mae:.2f}  RMSE=${rmse:.2f}  R2={r2:.4f}")


def _get_cat_indices(active_features: list[str]) -> list[int]:
    """Get indices of categorical features that exist in active_features."""
    return [i for i, f in enumerate(active_features) if f in CAT_FEATURE_NAMES]


# -- Batch mode: incremental training by Census region -------------------------

def train_batch(gold_dir: str, sample: float, active_features: list[str], split: str):
    user_home  = configure_databricks_mlflow()
    gold_files = _list_gold_files(gold_dir)
    cat_indices = _get_cat_indices(active_features)
    print(f"Found {len(gold_files)} gold state parquets in '{gold_dir}'")
    print(f"Device: {DEVICE}")
    print(f"Features ({len(active_features)}): {active_features}")
    print(f"Categorical features ({len(cat_indices)}): {[active_features[i] for i in cat_indices]}")

    val_y_parts = []
    val_pred_parts = []
    model = None
    total_iters = 0
    regions_trained = 0

    ablation = "Avg_Sbmtd_Chrg" not in active_features
    run_name = "catboost_no_charge_local" if ablation else "catboost_local"

    mlflow.set_experiment(f"{user_home}/medicare_models")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            **{k: v for k, v in CB_PARAMS.items() if k != "verbose"},
            "iters_per_region": ITERS_PER_REGION,
            "n_regions": len(CENSUS_REGIONS),
            "source": "local",
            "strategy": "incremental_region_batch",
            "target_transform": "log1p",
            "sample_frac": sample,
            "split_strategy": split,
            "n_features": len(active_features),
            "n_cat_features": len(cat_indices),
            "ablation_avg_submitted_charge": ablation,
        })

        for region_name, states in CENSUS_REGIONS.items():
            df, year = _load_region(gold_files, states, active_features, sample)
            if df is None:
                print(f"\n  [{region_name}] No data found -- skipping")
                continue

            if split == "temporal":
                df_train, df_val = _temporal_split(df, year)
            else:
                n_val  = max(1, int(len(df) * 0.2))
                idx    = np.random.RandomState(42).permutation(len(df))
                val_idx, train_idx = idx[:n_val], idx[n_val:]
                df_train = df.iloc[train_idx].reset_index(drop=True)
                df_val   = df.iloc[val_idx].reset_index(drop=True)

            if len(df_train) == 0:
                print(f"  [{region_name}] No training rows after split -- skipping")
                continue

            pool_train = _make_pool(df_train, active_features, cat_indices)
            pool_val   = _make_pool(df_val, active_features, cat_indices)

            cb = CatBoostRegressor(
                iterations=ITERS_PER_REGION,
                **CB_PARAMS,
            )
            cb.fit(
                pool_train,
                eval_set=pool_val,
                init_model=model,
                use_best_model=False,   # use all iterations (incremental)
            )
            model = cb

            # Collect validation predictions
            y_val_log = np.log1p(df_val[TARGET].astype("float64").values)
            y_pred_log = cb.predict(pool_val)
            val_y_parts.append(y_val_log)
            val_pred_parts.append(y_pred_log)

            total_iters += ITERS_PER_REGION
            regions_trained += 1
            print(f"  [{region_name}] {len(df_train):,} train / {len(df_val):,} val -- "
                  f"cumulative iters: {total_iters}")

        if not val_y_parts:
            print("No validation data collected -- aborting.")
            return

        y_val_all  = np.concatenate(val_y_parts)
        y_pred_all = np.concatenate(val_pred_parts)

        mlflow.log_params({"total_iterations": total_iters, "regions_trained": regions_trained})
        log_metrics(y_val_all, y_pred_all, prefix="test_")

        importances = dict(zip(active_features, model.get_feature_importance()))
        mlflow.log_dict(importances, "feature_importances.json")
        print("\nTop feature importances:")
        for feat, score in sorted(importances.items(), key=lambda x: -x[1])[:10]:
            print(f"  {feat}: {score:.2f}")

        mlflow.catboost.log_model(model, artifact_path="catboost_model")
        print(f"CatBoost batch run complete -- {regions_trained} regions, "
              f"{total_iters} total iterations. Model logged to MLflow.")


# -- Full mode: load everything into one Pool ----------------------------------

def train_full(gold_dir: str, sample: float, active_features: list[str], split: str):
    user_home  = configure_databricks_mlflow()
    gold_files = _list_gold_files(gold_dir)
    cat_indices = _get_cat_indices(active_features)
    print(f"Found {len(gold_files)} gold state parquets in '{gold_dir}'")
    print(f"Device: {DEVICE}")
    print(f"Features ({len(active_features)}): {active_features}")
    print(f"Categorical features ({len(cat_indices)}): {[active_features[i] for i in cat_indices]}")

    df, year = _load_region(gold_files, list(gold_files.keys()), active_features, sample)
    if df is None:
        raise RuntimeError("No gold data found.")

    if split == "temporal":
        df_train, df_test = _temporal_split(df, year)
    else:
        n_val  = int(len(df) * 0.2)
        idx    = np.random.RandomState(42).permutation(len(df))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test  = df.iloc[val_idx].reset_index(drop=True)

    print(f"Train: {len(df_train):,} rows  |  Val: {len(df_test):,} rows")

    pool_train = _make_pool(df_train, active_features, cat_indices)
    pool_val   = _make_pool(df_test, active_features, cat_indices)

    ablation = "Avg_Sbmtd_Chrg" not in active_features
    run_name = "catboost_no_charge_local" if ablation else "catboost_local"

    mlflow.set_experiment(f"{user_home}/medicare_models")
    with mlflow.start_run(run_name=run_name):
        n_iters = 500
        mlflow.log_params({
            **{k: v for k, v in CB_PARAMS.items() if k != "verbose"},
            "iterations": n_iters,
            "early_stopping_rounds": 30,
            "source": "local",
            "strategy": "full_pool",
            "target_transform": "log1p",
            "sample_frac": sample,
            "split_strategy": split,
            "n_features": len(active_features),
            "n_cat_features": len(cat_indices),
            "ablation_avg_submitted_charge": ablation,
        })

        cb = CatBoostRegressor(
            iterations=n_iters,
            **CB_PARAMS,
        )
        cb.fit(
            pool_train,
            eval_set=pool_val,
            early_stopping_rounds=30,
            use_best_model=True,
        )

        mlflow.log_param("best_iteration", cb.get_best_iteration())
        y_test_log = np.log1p(df_test[TARGET].astype("float64").values)
        y_pred_log = cb.predict(pool_val)
        log_metrics(y_test_log, y_pred_log, prefix="test_")

        importances = dict(zip(active_features, cb.get_feature_importance()))
        mlflow.log_dict(importances, "feature_importances.json")
        mlflow.catboost.log_model(cb, artifact_path="catboost_model")
        print("CatBoost full run complete. Model logged to MLflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="Path to gold directory with per-state parquets")
    parser.add_argument("--mode", choices=["batch", "full"], default="batch",
                        help="Training mode: 'batch' (incremental by region) or 'full'")
    parser.add_argument("--sample", type=float, default=0.3,
                        help="Fraction of rows to load per region/globally (default: 0.3)")
    parser.add_argument("--no-charge", action="store_true",
                        help="Ablation: exclude Avg_Sbmtd_Chrg from features")
    parser.add_argument("--split", choices=["random", "temporal"], default="random",
                        help="Split strategy: 'random' (80/20) or 'temporal' (train<=2021, test>=2022)")
    args = parser.parse_args()

    active_features = [f for f in FEATURES if not (args.no_charge and f == "Avg_Sbmtd_Chrg")]

    if args.mode == "batch":
        train_batch(args.data, args.sample, active_features, args.split)
    else:
        train_full(args.data, args.sample, active_features, args.split)
