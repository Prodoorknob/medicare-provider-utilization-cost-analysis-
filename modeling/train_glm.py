"""
train_glm.py — Generalized Linear Model baseline
Reads the Gold feature parquet, trains a GLM (Tweedie/Gamma for cost),
and logs all artifacts to Databricks MLflow.

Usage:
    python modeling/train_glm.py --data path/to/gold/features.parquet
"""

import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import TweedieRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET  = "Avg_Mdcr_Alowd_Amt"
FEATURES = [
    "Rndrng_Prvdr_Type_idx", "Rndrng_Prvdr_State_Abrvtn_idx",
    "HCPCS_Cd_idx", "hcpcs_bucket", "place_of_srvc_flag",
    "Bene_Avg_Risk_Scre", "log_srvcs", "log_benes",
    "Avg_Sbmtd_Chrg", "srvcs_per_bene",
]

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

    mlflow.set_experiment("/medicare/glm_baseline")
    with mlflow.start_run(run_name="glm_tweedie"):
        params = {"power": 1.5, "alpha": 0.1, "max_iter": 300}
        mlflow.log_params(params)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("glm",    TweedieRegressor(**params)),
        ])
        pipe.fit(X_train, y_train)

        log_metrics(y_train, pipe.predict(X_train), prefix="train_")
        log_metrics(y_test,  pipe.predict(X_test),  prefix="test_")
        mlflow.sklearn.log_model(pipe, artifact_path="glm_model")
        print("GLM run complete. Model logged to MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Gold features parquet")
    args = parser.parse_args()
    main(args.data)
