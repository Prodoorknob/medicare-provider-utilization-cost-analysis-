"""
compare_models.py — Unified Metrics Table + Paired t-Test
Pulls the latest run from each MLflow experiment, builds a comparison table,
and runs a paired t-test on test-set residuals to check for significant differences.

Usage:
    python modeling/compare_models.py
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
from scipy import stats

EXPERIMENTS = {
    "GLM":          "/medicare/glm_baseline",
    "RandomForest": "/medicare/random_forest",
    "XGBoost":      "/medicare/xgboost",
}
METRIC_KEYS = ["test_mae", "test_rmse", "test_r2"]

def fetch_best_run(experiment_name: str) -> dict:
    """Return metrics from the best (lowest test_rmse) run in an experiment."""
    client = mlflow.tracking.MlflowClient()
    exp    = client.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"  [WARN] Experiment not found: {experiment_name}")
        return {}
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.test_rmse ASC"],
        max_results=1,
    )
    if not runs:
        return {}
    return runs[0].data.metrics

def build_comparison_table() -> pd.DataFrame:
    rows = []
    for model_name, exp_name in EXPERIMENTS.items():
        metrics = fetch_best_run(exp_name)
        rows.append({
            "Model":     model_name,
            "Test MAE":  round(metrics.get("test_mae",  float("nan")), 2),
            "Test RMSE": round(metrics.get("test_rmse", float("nan")), 2),
            "Test R²":   round(metrics.get("test_r2",   float("nan")), 4),
        })
    return pd.DataFrame(rows).sort_values("Test RMSE")

def paired_t_test(residuals_a: np.ndarray, residuals_b: np.ndarray, label_a: str, label_b: str):
    """Two-sided paired t-test on absolute residuals."""
    t_stat, p_value = stats.ttest_rel(np.abs(residuals_a), np.abs(residuals_b))
    sig = "YES" if p_value < 0.05 else "NO"
    print(f"  {label_a} vs {label_b}: t={t_stat:.3f}, p={p_value:.4f} — Significant: {sig}")

def main():
    print("=== Model Comparison Table ===")
    table = build_comparison_table()
    print(table.to_string(index=False))

    # TODO: load held-out predictions from each model to compute residuals,
    # then call paired_t_test() between the top-2 models.
    # Example skeleton:
    # residuals_rf  = y_test - rf_model.predict(X_test)
    # residuals_xgb = y_test - xgb_model.predict(X_test)
    # paired_t_test(residuals_rf, residuals_xgb, "RF", "XGBoost")

    print("\nDone. Run individual train_*.py scripts first to populate MLflow experiments.")

if __name__ == "__main__":
    main()
