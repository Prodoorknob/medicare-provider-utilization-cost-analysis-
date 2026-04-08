"""
train_oop_local.py — Stage 2 OOP Quantile Regression, logs to Databricks MLflow

Trains 3 XGBoost quantile regressors (P10, P50, P90) to predict patient
per-service out-of-pocket costs using:
  - Provider-side features (allowed amount, specialty, risk score, etc.)
  - Beneficiary-side features (age, income, chronic count, dual eligible, etc.)
  - Census region

The allowed amount from Stage 1 is the PRIMARY input feature — this is
where the two-stage pipeline connects: Stage 1 predicts what Medicare allows,
Stage 2 predicts what the patient pays out of pocket.

Training data:
  - Default: local_pipeline/mcbs_synthetic/synthetic_oop.parquet (synthetic)
  - Drop-in replacement: real MCBS LDS data with the same column schema

Required env vars:
    DATABRICKS_HOST   e.g. https://<workspace>.azuredatabricks.net
    DATABRICKS_TOKEN  Personal Access Token or service principal secret

Usage:
    python modeling/train_oop_local.py
    python modeling/train_oop_local.py --sample 0.3 --rounds 500
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Fix Windows console encoding for MLflow emoji output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DATA = os.path.join(
    _PROJECT_ROOT, "local_pipeline", "mcbs_synthetic", "synthetic_oop.parquet"
)

TARGET = "per_service_oop"

FEATURES = [
    # Provider-side (from gold)
    "Avg_Mdcr_Alowd_Amt",      # Stage 1 target → Stage 2 feature
    "Bene_Avg_Risk_Scre",       # HCC risk score
    "Rndrng_Prvdr_Type_idx",    # Specialty
    "hcpcs_bucket",             # Service category
    "place_of_srvc_flag",       # Facility vs office
    # Beneficiary-side (from MCBS / synthetic)
    "census_region",            # Geographic (1-4)
    "age",                      # Demographic
    "sex",                      # Demographic
    "income",                   # Socioeconomic bracket
    "chronic_count",            # Health burden
    "dual_eligible",            # Medicaid dual status
    "has_supplemental",         # Private supplemental insurance
]

QUANTILES = [0.1, 0.5, 0.9]
QUANTILE_LABELS = ["p10", "p50", "p90"]


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
# Device detection (matches train_xgb_local.py)
# ---------------------------------------------------------------------------
def _detect_device() -> str:
    """Use GPU if XGBoost was built with CUDA support, else CPU."""
    try:
        _d = xgb.DMatrix(np.zeros((1, 1)))
        xgb.train({"device": "cuda", "verbosity": 0}, _d, num_boost_round=1)
        return "cuda"
    except Exception:
        return "cpu"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Quantile (pinball) loss — proper scoring rule for quantile regression."""
    diff = y_true - y_pred
    return np.mean(np.where(diff >= 0, alpha * diff, (alpha - 1) * diff))


def coverage(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of actuals that fall below the predicted quantile."""
    return float(np.mean(y_true <= y_pred))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(data_path: str, sample: float, n_rounds: int, patience: int):
    device = _detect_device()
    print(f"Device: {device}")

    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"  Total rows: {len(df):,}")

    if sample < 1.0:
        df = df.sample(frac=sample, random_state=42)
        print(f"  Sampled {sample:.0%}: {len(df):,} rows")

    # Validate columns
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    # Prepare arrays
    X = df[FEATURES].astype("float64").values
    y = df[TARGET].astype("float64").values
    print(f"  Features: {len(FEATURES)}, Target: {TARGET}")
    print(f"  Target stats: mean=${y.mean():,.2f}, median=${np.median(y):,.2f}, "
          f"std=${y.std():,.2f}")

    # Train/test split (80/20, same seed as Stage 1)
    n_val = int(len(y) * 0.2)
    idx = np.random.RandomState(42).permutation(len(y))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    print(f"  Train: {len(train_idx):,} | Test: {n_val:,}")

    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx], feature_names=FEATURES)
    dval   = xgb.DMatrix(X[val_idx],   label=y[val_idx],   feature_names=FEATURES)

    # Base XGB params
    base_params = {
        "learning_rate":    0.05,
        "max_depth":        6,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "tree_method":      "hist",
        "device":           device,
        "seed":             42,
    }

    # MLflow
    user_home = configure_databricks_mlflow()
    mlflow.set_experiment(f"{user_home}/medicare_models")

    with mlflow.start_run(run_name="xgb_quantile_oop_local"):
        mlflow.log_params({
            **base_params,
            "model":            "XGBoost_Quantile_OOP",
            "stage":            2,
            "target":           TARGET,
            "quantiles":        str(QUANTILES),
            "n_features":       len(FEATURES),
            "features":         str(FEATURES),
            "n_rounds":         n_rounds,
            "early_stopping":   patience,
            "sample_frac":      sample,
            "n_train":          len(train_idx),
            "n_test":           n_val,
            "data_source":      data_path,
            "source":           "local",
        })

        boosters = {}
        all_metrics = {}

        for alpha, label in zip(QUANTILES, QUANTILE_LABELS):
            print(f"\n{'='*60}")
            print(f"Training {label.upper()} (quantile_alpha={alpha})")
            print(f"{'='*60}")

            params = {
                **base_params,
                "objective":      "reg:quantileerror",
                "quantile_alpha": alpha,
            }

            evals_result = {}
            booster = xgb.train(
                params, dtrain,
                num_boost_round=n_rounds,
                evals=[(dval, "val")],
                early_stopping_rounds=patience,
                evals_result=evals_result,
                verbose_eval=50,
            )

            # Predict on test set
            y_pred = booster.predict(dval)
            y_true = y[val_idx]

            # Metrics
            mae  = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2   = r2_score(y_true, y_pred)
            cov  = coverage(y_true, y_pred)
            pbl  = pinball_loss(y_true, y_pred, alpha)
            best_iter = booster.best_iteration

            print(f"\n  {label} Results:")
            print(f"    MAE      = ${mae:,.2f}")
            print(f"    RMSE     = ${rmse:,.2f}")
            print(f"    R2       = {r2:.4f}")
            print(f"    Coverage = {cov:.1%} (target: {alpha:.0%})")
            print(f"    Pinball  = {pbl:.4f}")
            print(f"    Best iteration: {best_iter}")

            metrics = {
                f"{label}_mae":      mae,
                f"{label}_rmse":     rmse,
                f"{label}_r2":       r2,
                f"{label}_coverage": cov,
                f"{label}_pinball":  pbl,
                f"{label}_best_iter": best_iter,
            }
            all_metrics.update(metrics)

            # Feature importance
            importances = booster.get_score(importance_type="gain")
            mlflow.log_dict(importances, f"feature_importances_{label}.json")

            # Log model
            mlflow.xgboost.log_model(booster, artifact_path=f"xgb_oop_{label}")

            boosters[label] = booster

        # Log all metrics at once
        mlflow.log_metrics(all_metrics)

        # Also log the P50 metrics as the "main" test metrics (for comparison table)
        mlflow.log_metrics({
            "test_mae":  all_metrics["p50_mae"],
            "test_rmse": all_metrics["p50_rmse"],
            "test_r2":   all_metrics["p50_r2"],
        })

        print(f"\n{'='*60}")
        print("Summary — All Quantiles")
        print(f"{'='*60}")
        print(f"  {'Quantile':<10} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'Coverage':>10} {'Pinball':>10}")
        print(f"  {'-'*58}")
        for alpha, label in zip(QUANTILES, QUANTILE_LABELS):
            print(f"  {label.upper():<10} "
                  f"${all_metrics[f'{label}_mae']:>9,.2f} "
                  f"${all_metrics[f'{label}_rmse']:>9,.2f} "
                  f"{all_metrics[f'{label}_r2']:>8.4f} "
                  f"{all_metrics[f'{label}_coverage']:>9.1%} "
                  f"{all_metrics[f'{label}_pinball']:>10.4f}")

        print(f"\n  MLflow run logged to experiment: {user_home}/medicare_models")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2 OOP quantile regression (P10/P50/P90) using XGBoost."
    )
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="Path to OOP training data (synthetic or real LDS)")
    parser.add_argument("--sample", type=float, default=0.3,
                        help="Fraction of rows to sample (default: 0.3)")
    parser.add_argument("--rounds", type=int, default=300,
                        help="Max boosting rounds per quantile (default: 300)")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience (default: 30)")
    args = parser.parse_args()
    train(args.data, args.sample, args.rounds, args.patience)
