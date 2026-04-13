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
    "specialty_bucket", "pos_bucket", "hcpcs_target_enc",
    # year, is_covid_era: added after silver is re-run with year injection
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
    active_features controls which columns become X (allows --no-charge ablation).
    year is always loaded for temporal split.
    """
    load_cols = list(dict.fromkeys(active_features + [TARGET, "year"]))
    parts = []
    for st in states:
        if st in gold_files:
            avail = set(pq.read_schema(gold_files[st]).names)
            cols  = [c for c in load_cols if c in avail]
            df = pd.read_parquet(gold_files[st], columns=cols).dropna(
                subset=active_features + [TARGET]
            )
            parts.append(df)
    if not parts:
        return None, None, None
    df = pd.concat(parts, ignore_index=True)
    if sample < 1.0:
        df = df.sample(frac=sample, random_state=42)
    X    = df[active_features].astype("float64").values
    y    = np.log1p(df[TARGET].astype("float64").values)
    year = df["year"].values if "year" in df.columns else None
    return X, y, year


def _temporal_split(X, y, year):
    """Train on years <= 2021, test on years >= 2022."""
    if year is None:
        raise ValueError("year column not available for temporal split")
    train_mask = year <= 2021
    test_mask  = year >= 2022
    n_train, n_test = train_mask.sum(), test_mask.sum()
    if n_test == 0:
        raise ValueError("No test records with year >= 2022. Check gold data contains 2022-2023.")
    print(f"  Temporal split: {n_train:,} train (≤2021) / {n_test:,} test (≥2022)")
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


def log_metrics(y_true_log, y_pred_log, prefix=""):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mlflow.log_metrics({f"{prefix}mae": mae, f"{prefix}rmse": rmse, f"{prefix}r2": r2})
    print(f"  {prefix}MAE=${mae:.2f}  RMSE=${rmse:.2f}  R²={r2:.4f}")


# ── Batch mode: incremental training by Census region ────────────────────────

def train_batch(gold_dir: str, sample: float, active_features: list[str], split: str):
    user_home  = configure_databricks_mlflow()
    gold_files = _list_gold_files(gold_dir)
    print(f"Found {len(gold_files)} gold state parquets in '{gold_dir}'")
    print(f"Device: {XGB_PARAMS['device']}")
    print(f"Features ({len(active_features)}): {active_features}")

    X_val_parts, y_val_parts = [], []
    booster = None
    total_rounds = 0
    regions_trained = 0

    ablation = "Avg_Sbmtd_Chrg" not in active_features
    run_name = "xgb_no_charge_local" if ablation else "xgb_extmem_local"

    mlflow.set_experiment(f"{user_home}/medicare_models")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            **XGB_PARAMS,
            "rounds_per_region": ROUNDS_PER_REGION,
            "n_regions": len(CENSUS_REGIONS),
            "source": "local",
            "strategy": "incremental_region_batch",
            "target_transform": "log1p",
            "sample_frac": sample,
            "split_strategy": split,
            "n_features": len(active_features),
            "ablation_avg_submitted_charge": ablation,
        })

        for region_name, states in CENSUS_REGIONS.items():
            X, y, year = _load_region(gold_files, states, active_features, sample)
            if X is None:
                print(f"\n  [{region_name}] No data found — skipping")
                continue

            if split == "temporal":
                X_train, X_val, y_train, y_val = _temporal_split(X, y, year)
            else:
                n_val  = max(1, int(len(y) * 0.2))
                idx    = np.random.RandomState(42).permutation(len(y))
                val_idx, train_idx = idx[:n_val], idx[n_val:]
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

            if len(X_train) == 0:
                print(f"  [{region_name}] No training rows after split — skipping")
                continue

            X_val_parts.append(X_val)
            y_val_parts.append(y_val)

            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=active_features)
            dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=active_features)

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
            print(f"  [{region_name}] {len(X_train):,} train / {len(X_val):,} val — "
                  f"cumulative rounds: {total_rounds}")

        if not X_val_parts:
            print("No validation data collected — aborting.")
            return

        X_val_all = np.vstack(X_val_parts)
        y_val_all = np.concatenate(y_val_parts)
        dval_all  = xgb.DMatrix(X_val_all, label=y_val_all, feature_names=active_features)

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

def train_full(gold_dir: str, sample: float, active_features: list[str], split: str):
    user_home  = configure_databricks_mlflow()
    gold_files = _list_gold_files(gold_dir)
    print(f"Found {len(gold_files)} gold state parquets in '{gold_dir}'")
    print(f"Device: {XGB_PARAMS['device']}")
    print(f"Features ({len(active_features)}): {active_features}")

    X, y, year = _load_region(gold_files, list(gold_files.keys()), active_features, sample)
    if X is None:
        raise RuntimeError("No gold data found.")

    if split == "temporal":
        X_train, X_test, y_train, y_test = _temporal_split(X, y, year)
    else:
        n_val  = int(len(y) * 0.2)
        idx    = np.random.RandomState(42).permutation(len(y))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        X_train, X_test = X[train_idx], X[val_idx]
        y_train, y_test = y[train_idx], y[val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=active_features)
    dval   = xgb.DMatrix(X_test,  label=y_test,  feature_names=active_features)
    print(f"Train: {len(y_train):,} rows  |  Val: {len(y_test):,} rows")

    ablation = "Avg_Sbmtd_Chrg" not in active_features
    run_name = "xgb_no_charge_local" if ablation else "xgb_extmem_local"

    mlflow.set_experiment(f"{user_home}/medicare_models")
    with mlflow.start_run(run_name=run_name):
        n_rounds = 500
        mlflow.log_params({
            **XGB_PARAMS,
            "n_rounds": n_rounds,
            "early_stopping": 30,
            "source": "local",
            "strategy": "full_dmatrix",
            "target_transform": "log1p",
            "sample_frac": sample,
            "split_strategy": split,
            "n_features": len(active_features),
            "ablation_avg_submitted_charge": ablation,
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
        log_metrics(y_test, y_pred, prefix="test_")

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
    parser.add_argument("--no-charge", action="store_true",
                        help="Ablation: exclude Avg_Sbmtd_Chrg from features")
    parser.add_argument("--split", choices=["random", "temporal"], default="random",
                        help="Split strategy: 'random' (80/20) or 'temporal' (train≤2021, test≥2022)")
    args = parser.parse_args()

    active_features = [f for f in FEATURES if not (args.no_charge and f == "Avg_Sbmtd_Chrg")]

    if args.mode == "batch":
        train_batch(args.data, args.sample, active_features, args.split)
    else:
        train_full(args.data, args.sample, active_features, args.split)
