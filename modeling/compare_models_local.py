"""
compare_models_local.py — Unified Metrics Table + Paired t-Test (remote Databricks MLflow)
Fetches the best run from each experiment on the Databricks MLflow tracking server,
builds a comparison table, and runs a paired t-test on residuals.

Required env vars:
    DATABRICKS_HOST   e.g. https://<workspace>.azuredatabricks.net
    DATABRICKS_TOKEN  Personal Access Token or service principal secret

Usage:
    python modeling/compare_models_local.py
    python modeling/compare_models_local.py --data local_pipeline/gold/gold.parquet
"""

import os
import argparse
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
from scipy import stats

DEFAULT_DATA  = os.path.join("local_pipeline", "gold")
TARGET        = "Avg_Mdcr_Alowd_Amt"
FEATURES      = [
    "Rndrng_Prvdr_Type_idx", "Rndrng_Prvdr_State_Abrvtn_idx",
    "HCPCS_Cd_idx", "hcpcs_bucket", "place_of_srvc_flag",
    "Bene_Avg_Risk_Scre", "log_srvcs", "log_benes",
    "Avg_Sbmtd_Chrg", "srvcs_per_bene",
    "specialty_bucket", "pos_bucket", "hcpcs_target_enc",
    # year, is_covid_era: added after silver is re-run with year injection
]
MODEL_RUN_NAMES = {
    "GLM":          "glm_sgd_local",
    "RandomForest": "rf_randomized_search_local",
    "XGBoost":      "xgb_extmem_local",
    "CatBoost":     "catboost_local",
    "LightGBM":     "lgbm_local",
    "LSTM":         "lstm_local",
}

# Stage 2 OOP model (separate target: per_service_oop, not Avg_Mdcr_Alowd_Amt)
OOP_RUN_NAMES = {
    "XGB_Quantile_OOP": "xgb_quantile_oop_local",
}


def configure_databricks_mlflow() -> str:
    """Configure MLflow and return the current user's workspace home path."""
    import requests
    host  = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")
    if not host or not token:
        raise EnvironmentError(
            "DATABRICKS_HOST and DATABRICKS_TOKEN must be set to pull from Databricks MLflow.\n"
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
    print(f"MLflow tracking URI set to Databricks: {host}  (user: {username})\n")
    return f"/Users/{username}"


def fetch_best_run(experiment_name: str, run_name: str) -> tuple[str | None, dict]:
    """Return (run_id, metrics) for the lowest test_rmse run matching run_name."""
    client = mlflow.tracking.MlflowClient()
    exp    = client.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"  [WARN] Experiment not found: {experiment_name}")
        return None, {}
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["metrics.test_rmse ASC"],
        max_results=1,
    )
    if not runs:
        print(f"  [WARN] No runs found with name '{run_name}'")
        return None, {}
    return runs[0].info.run_id, runs[0].data.metrics


def build_comparison_table(experiment_name: str) -> pd.DataFrame:
    rows = []
    run_ids = {}
    for model_name, run_name in MODEL_RUN_NAMES.items():
        run_id, metrics = fetch_best_run(experiment_name, run_name)
        run_ids[model_name] = run_id
        rows.append({
            "Model":     model_name,
            "Run ID":    (run_id or "")[:8],
            "Test MAE":  round(metrics.get("test_mae",  float("nan")), 2),
            "Test RMSE": round(metrics.get("test_rmse", float("nan")), 2),
            "Test R²":   round(metrics.get("test_r2",   float("nan")), 4),
        })
    return pd.DataFrame(rows).sort_values("Test RMSE"), run_ids


def paired_t_test(residuals_a: np.ndarray, residuals_b: np.ndarray, label_a: str, label_b: str):
    t_stat, p_value = stats.ttest_rel(np.abs(residuals_a), np.abs(residuals_b))
    sig = "YES (p<0.05)" if p_value < 0.05 else "NO"
    print(f"  {label_a} vs {label_b}: t={t_stat:.3f}, p={p_value:.4f} — Significant difference: {sig}")


def compute_residuals(run_id: str, X_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray | None:
    """Download the logged model from MLflow and return residuals on the test split."""
    if run_id is None:
        return None
    try:
        model_uri = f"runs:/{run_id}/rf_model"   # try RF first; GLM/XGB have different artifact names
        model     = mlflow.sklearn.load_model(model_uri)
        return (y_test.values - model.predict(X_test)).astype(float)
    except Exception:
        pass
    try:
        model_uri = f"runs:/{run_id}/xgb_model"
        model     = mlflow.xgboost.load_model(model_uri)
        return (y_test.values - model.predict(X_test)).astype(float)
    except Exception:
        pass
    try:
        model_uri = f"runs:/{run_id}/glm_model"
        model     = mlflow.sklearn.load_model(model_uri)
        return (y_test.values - model.predict(X_test)).astype(float)
    except Exception:
        pass
    try:
        model_uri = f"runs:/{run_id}/catboost_model"
        import mlflow.catboost
        model     = mlflow.catboost.load_model(model_uri)
        return (y_test.values - model.predict(X_test.values)).astype(float)
    except Exception:
        pass
    try:
        model_uri = f"runs:/{run_id}/lgbm_model"
        import mlflow.lightgbm
        model     = mlflow.lightgbm.load_model(model_uri)
        return (y_test.values - model.predict(X_test.values)).astype(float)
    except Exception:
        pass
    try:
        # LSTM operates on sequences, not flat features -- skip residual computation
        model_uri = f"runs:/{run_id}/lstm_model"
        mlflow.pytorch.load_model(model_uri)
        print(f"  [INFO] LSTM model found but residuals not applicable (sequence model)")
        return None
    except Exception as e:
        print(f"  [WARN] Could not load model for run {run_id}: {e}")
        return None


def main(data_path: str):
    user_home = configure_databricks_mlflow()
    experiment_name = f"{user_home}/medicare_models"

    print("=== Model Comparison (fetched from Databricks MLflow) ===")
    table, run_ids = build_comparison_table(experiment_name)
    print(table.to_string(index=False))
    print()
    print("  NOTE — LSTM R² is NOT comparable to GLM/RF/XGB:")
    print("    GLM/RF/XGB: R² on millions of individual service records (80/20 random split)")
    print("    LSTM:       R² on ~17K group-year means (specialty×bucket×state, 2022-2023 temporal split)")
    print("    Group means are smoother -> LSTM R² is structurally inflated vs. individual-record models.")
    print("    For apples-to-apples, run RF/XGB with --split temporal and compare at group level.")

    # Paired t-test — requires loading the held-out test split locally
    if os.path.exists(data_path):
        print("\n=== Paired t-Test on Test-Set Residuals ===")
        import glob
        pq_files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
        df = pd.concat([pd.read_parquet(f) for f in pq_files], ignore_index=True).dropna(subset=FEATURES + [TARGET])
        present  = [c for c in FEATURES if c in df.columns]
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(df[present], df[TARGET], test_size=0.2, random_state=42)

        residuals = {name: compute_residuals(rid, X_test, y_test) for name, rid in run_ids.items()}
        pairs     = [
            ("CatBoost", "XGBoost"), ("CatBoost", "LightGBM"),
            ("LightGBM", "XGBoost"), ("RandomForest", "XGBoost"),
            ("GLM", "XGBoost"), ("GLM", "RandomForest"),
        ]
        for a, b in pairs:
            if residuals.get(a) is not None and residuals.get(b) is not None:
                paired_t_test(residuals[a], residuals[b], a, b)
    else:
        print(f"\n[INFO] Gold parquet not found at '{data_path}' — skipping t-test.")

    # Stage 2 OOP model (different target — separate section)
    print("\n=== Stage 2: OOP Quantile Model (fetched from Databricks MLflow) ===")
    oop_rows = []
    for model_name, run_name in OOP_RUN_NAMES.items():
        run_id, metrics = fetch_best_run(experiment_name, run_name)
        if run_id is None:
            print(f"  [INFO] No OOP model run found yet — train with: python modeling/train_oop_local.py")
            break
        oop_rows.append({
            "Model":        model_name,
            "Run ID":       (run_id or "")[:8],
            "P10 MAE":      round(metrics.get("p10_mae",      float("nan")), 2),
            "P50 MAE":      round(metrics.get("p50_mae",      float("nan")), 2),
            "P90 MAE":      round(metrics.get("p90_mae",      float("nan")), 2),
            "P10 Coverage": round(metrics.get("p10_coverage",  float("nan")), 3),
            "P50 Coverage": round(metrics.get("p50_coverage",  float("nan")), 3),
            "P90 Coverage": round(metrics.get("p90_coverage",  float("nan")), 3),
        })
    if oop_rows:
        oop_table = pd.DataFrame(oop_rows)
        print(oop_table.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to gold directory with per-state parquets (for t-test)")
    args = parser.parse_args()
    main(args.data)
