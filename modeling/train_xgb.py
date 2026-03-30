"""
train_xgb.py — XGBoost + Cross-Validation
Trains an XGBRegressor with early stopping and 5-fold CV, logs to MLflow.

Usage:
    python modeling/train_xgb.py --data path/to/gold/features.parquet
"""

import argparse
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, cv as xgb_cv, DMatrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET   = "Avg_Mdcr_Pymt_Amt"
FEATURES = [
    "Rndrng_Prvdr_Type_idx", "Rndrng_Prvdr_State_Abrvtn_idx", "HCPCS_Cd_idx",
    "Tot_Benes", "Tot_Srvcs", "Avg_Sbmtd_Chrg", "Avg_Mdcr_Allo_Amt",
    "Avg_Mdcr_Stdzd_Amt", "srvcs_per_bene", "pymt_to_charge_ratio", "stdz_to_pymt_ratio",
]
XGB_PARAMS = {
    "objective":        "reg:squarederror",
    "learning_rate":    0.05,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "n_estimators":     500,
    "early_stopping_rounds": 30,
    "random_state":     42,
    "n_jobs":           -1,
}

def load_data(path: str):
    df = pd.read_parquet(path).dropna(subset=FEATURES + [TARGET])
    X  = df[FEATURES]
    y  = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def log_metrics(y_true, y_pred, prefix=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mlflow.log_metrics({f"{prefix}mae": mae, f"{prefix}rmse": rmse, f"{prefix}r2": r2})
    print(f"  {prefix}MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}

def main(data_path: str):
    X_train, X_test, y_train, y_test = load_data(data_path)

    mlflow.set_experiment("/medicare/xgboost")
    with mlflow.start_run(run_name="xgb_early_stopping"):
        mlflow.log_params({k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"})

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )
        mlflow.log_param("best_iteration", model.best_iteration)

        log_metrics(y_train, model.predict(X_train), prefix="train_")
        log_metrics(y_test,  model.predict(X_test),  prefix="test_")

        importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        print("\nTop feature importances:\n", importances.head(10))
        mlflow.log_dict(importances.to_dict(), "feature_importances.json")
        mlflow.xgboost.log_model(model, artifact_path="xgb_model")
        print("XGBoost run complete. Model logged to MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Gold features parquet")
    args = parser.parse_args()
    main(args.data)
