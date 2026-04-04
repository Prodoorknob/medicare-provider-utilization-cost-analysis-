"""
train_rf.py — Random Forest + Hyperparameter Tuning
Trains a RandomForestRegressor with RandomizedSearchCV and logs to MLflow.

Usage:
    python modeling/train_rf.py --data path/to/gold/features.parquet
"""

import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET   = "Avg_Mdcr_Alowd_Amt"
FEATURES = [
    "Rndrng_Prvdr_Type_idx", "Rndrng_Prvdr_State_Abrvtn_idx",
    "HCPCS_Cd_idx", "hcpcs_bucket", "place_of_srvc_flag",
    "Bene_Avg_Risk_Scre", "log_srvcs", "log_benes",
    "Avg_Sbmtd_Chrg", "srvcs_per_bene",
]
PARAM_DIST = {
    "n_estimators":      [100, 200, 400],
    "max_depth":         [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2"],
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

    mlflow.set_experiment("/medicare/random_forest")
    with mlflow.start_run(run_name="rf_randomized_search"):
        search = RandomizedSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            param_distributions=PARAM_DIST,
            n_iter=20,
            cv=5,
            scoring="neg_root_mean_squared_error",
            random_state=42,
            verbose=1,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_

        mlflow.log_params(search.best_params_)
        log_metrics(y_train, best.predict(X_train), prefix="train_")
        log_metrics(y_test,  best.predict(X_test),  prefix="test_")

        importances = pd.Series(best.feature_importances_, index=FEATURES).sort_values(ascending=False)
        print("\nTop feature importances:\n", importances.head(10))
        mlflow.log_dict(importances.to_dict(), "feature_importances.json")
        mlflow.sklearn.log_model(best, artifact_path="rf_model")
        print("RF run complete. Best model logged to MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Gold features parquet")
    args = parser.parse_args()
    main(args.data)
