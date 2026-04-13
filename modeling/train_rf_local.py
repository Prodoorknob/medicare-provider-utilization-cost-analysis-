"""
train_rf_local.py — Random Forest, trained locally, logs to Databricks MLflow

Two training modes:
  --mode batch  (default): Warm-start training by Census region. Adds 125
                trees per region (500 total across 4 regions + territories).
                Uses sklearn only — cuML RF lacks warm_start support.
  --mode full:  Loads all gold parquets with sampling, runs RandomizedSearchCV.
                Auto-detects cuML (GPU) vs sklearn (CPU).

Required env vars:
    DATABRICKS_HOST   e.g. https://<workspace>.azuredatabricks.net
    DATABRICKS_TOKEN  Personal Access Token or service principal secret

Usage:
    python modeling/train_rf_local.py --mode batch
    python modeling/train_rf_local.py --mode full --sample 0.5
"""

import os
import glob
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
# year must be in LOAD_COLS even when not in active features (needed for temporal split)
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
TREES_PER_REGION = 125   # 5 regions × 125 = 625 total trees

# Fixed hyperparams for batch mode (provisioned for future search)
BATCH_RF_PARAMS = {
    "max_depth":         20,
    "min_samples_split": 5,
    "min_samples_leaf":  2,
    "max_features":      "sqrt",
    "random_state":      42,
    "n_jobs":            -1,
}

# Search grid for full mode
PARAM_DIST = {
    "n_estimators":      [100, 200, 400],
    "max_depth":         [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2"],
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
    year is always loaded for temporal split even if not in active_features.
    """
    load_cols = list(dict.fromkeys(active_features + [TARGET, "year"]))
    parts = []
    for st in states:
        if st in gold_files:
            avail = set(pd.read_parquet(gold_files[st], columns=[]).columns
                        if False else
                        pq.read_schema(gold_files[st]).names)
            cols = [c for c in load_cols if c in avail]
            df = pd.read_parquet(gold_files[st], columns=cols).dropna(subset=active_features + [TARGET])
            parts.append(df)
    if not parts:
        return None, None, None
    df = pd.concat(parts, ignore_index=True)
    if sample < 1.0:
        df = df.sample(frac=sample, random_state=42)
    X    = df[active_features].astype("float32").values
    y    = np.log1p(df[TARGET].astype("float32").values)
    year = df["year"].values if "year" in df.columns else None
    return X, y, year


def _temporal_split(X, y, year):
    """Train on years <= 2021, test on years >= 2022."""
    if year is None:
        raise ValueError("year column not available for temporal split")
    train_mask = year <= 2021
    test_mask  = year >= 2022
    n_train = train_mask.sum()
    n_test  = test_mask.sum()
    if n_test == 0:
        raise ValueError("No test records with year >= 2022. Check gold data contains 2022-2023.")
    print(f"  Temporal split: {n_train:,} train (≤2021) / {n_test:,} test (≥2022)")
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


def log_metrics(y_true_log, y_pred_log, prefix=""):
    y_true = np.expm1(np.asarray(y_true_log, dtype="float64"))
    y_pred = np.expm1(np.asarray(y_pred_log, dtype="float64"))
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mlflow.log_metrics({f"{prefix}mae": mae, f"{prefix}rmse": rmse, f"{prefix}r2": r2})
    print(f"  {prefix}MAE=${mae:.2f}  RMSE=${rmse:.2f}  R²={r2:.4f}")


# ── Batch mode: warm_start by Census region (sklearn only) ───────────────────

def train_batch(gold_dir: str, sample: float, active_features: list[str], split: str):
    user_home  = configure_databricks_mlflow()
    gold_files = _list_gold_files(gold_dir)
    print(f"Found {len(gold_files)} gold state parquets")
    print(f"Backend: scikit-learn (CPU) — cuML RF does not support warm_start")
    print(f"Features ({len(active_features)}): {active_features}")

    X_val_parts, y_val_parts = [], []
    cumulative_trees = 0

    rf = RandomForestRegressor(
        n_estimators=TREES_PER_REGION,
        warm_start=True,
        **BATCH_RF_PARAMS,
    )

    ablation = "Avg_Sbmtd_Chrg" not in active_features
    run_name = "rf_no_charge_local" if ablation else "rf_randomized_search_local"

    mlflow.set_experiment(f"{user_home}/medicare_models")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            **BATCH_RF_PARAMS,
            "trees_per_region": TREES_PER_REGION,
            "n_regions": len(CENSUS_REGIONS),
            "sample_frac": sample,
            "source": "local",
            "backend": "sklearn_warm_start",
            "strategy": "warm_start_region_batch",
            "target_transform": "log1p",
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

            cumulative_trees += TREES_PER_REGION
            rf.n_estimators = cumulative_trees
            rf.fit(X_train, y_train)

            print(f"  [{region_name}] {len(X_train):,} train / {len(X_val):,} val — "
                  f"trees: {cumulative_trees}")

        if not X_val_parts:
            print("No validation data collected — aborting.")
            return

        X_val_all = np.vstack(X_val_parts)
        y_val_all = np.concatenate(y_val_parts)

        mlflow.log_param("total_trees", cumulative_trees)
        log_metrics(y_val_all, rf.predict(X_val_all), prefix="test_")

        importances = pd.Series(
            rf.feature_importances_, index=active_features
        ).sort_values(ascending=False)
        print("\nTop feature importances:\n", importances.head(10))
        mlflow.log_dict(importances.to_dict(), "feature_importances.json")
        mlflow.sklearn.log_model(rf, artifact_path="rf_model")
        print(f"RF batch run complete — {cumulative_trees} trees. Model logged to MLflow.")


# ── Full mode: load everything, RandomizedSearchCV ───────────────────────────

def _get_rf_backend_full():
    """Return (RandomForestRegressor class, backend name, param_dist) for full mode."""
    try:
        from cuml.ensemble import RandomForestRegressor as cuRF
        n_features = len(FEATURES)
        param_dist = {
            "n_estimators":      [100, 200, 400],
            "max_depth":         [10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf":  [1, 2, 4],
            "max_features":      [round(n_features ** 0.5 / n_features, 2),
                                  round(np.log2(n_features) / n_features, 2)],
        }
        print("Backend: cuML (GPU)")
        return cuRF, "cuml_gpu", param_dist
    except ImportError:
        print("Backend: scikit-learn (CPU)")
        return RandomForestRegressor, "sklearn_cpu", PARAM_DIST


def train_full(gold_dir: str, sample: float, active_features: list[str], split: str):
    user_home  = configure_databricks_mlflow()
    gold_files = _list_gold_files(gold_dir)
    RFRegressor, backend, param_dist = _get_rf_backend_full()
    print(f"Features ({len(active_features)}): {active_features}")

    X, y, year = _load_region(gold_files, list(gold_files.keys()), active_features, sample)
    if X is None:
        raise RuntimeError("No gold data found.")

    if split == "temporal":
        X_train, X_test, y_train, y_test = _temporal_split(X, y, year)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    print(f"Train: {len(y_train):,}  |  Test: {len(y_test):,}")

    ablation = "Avg_Sbmtd_Chrg" not in active_features
    run_name = "rf_no_charge_local" if ablation else "rf_randomized_search_local"

    rf_kwargs = {"random_state": 42}
    if backend == "sklearn_cpu":
        rf_kwargs["n_jobs"] = -1

    mlflow.set_experiment(f"{user_home}/medicare_models")
    with mlflow.start_run(run_name=run_name):
        search = RandomizedSearchCV(
            RFRegressor(**rf_kwargs),
            param_distributions=param_dist,
            n_iter=20, cv=5,
            scoring="neg_root_mean_squared_error",
            random_state=42, verbose=1,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_

        mlflow.log_params({
            **search.best_params_,
            "sample_frac": sample, "source": "local",
            "backend": backend,
            "strategy": "full_randomized_search",
            "target_transform": "log1p",
            "split_strategy": split,
            "n_features": len(active_features),
            "ablation_avg_submitted_charge": ablation,
        })
        log_metrics(y_train, best.predict(X_train), prefix="train_")
        log_metrics(y_test,  best.predict(X_test),  prefix="test_")

        importances = pd.Series(
            best.feature_importances_, index=active_features
        ).sort_values(ascending=False)
        print("\nTop feature importances:\n", importances.head(10))
        mlflow.log_dict(importances.to_dict(), "feature_importances.json")
        mlflow.sklearn.log_model(best, artifact_path="rf_model")
        print(f"RF full run complete ({backend}). Model logged to MLflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="Path to gold directory with per-state parquets")
    parser.add_argument("--mode", choices=["batch", "full"], default="batch",
                        help="Training mode: 'batch' (warm_start by region) or 'full'")
    parser.add_argument("--sample", type=float, default=0.3,
                        help="Fraction of rows to load (default: 0.3)")
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
