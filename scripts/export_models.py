"""Export model artifacts from MLflow to api/models/artifacts/ for deployment.

Usage:
    python scripts/export_models.py

    Or manually place model files:
      - api/models/artifacts/lgbm_model.txt     (LightGBM booster)
      - api/models/artifacts/xgb_oop_p10.ubj    (XGBoost quantile P10)
      - api/models/artifacts/xgb_oop_p50.ubj    (XGBoost quantile P50)
      - api/models/artifacts/xgb_oop_p90.ubj    (XGBoost quantile P90)

    Encoding files are copied from local_pipeline/gold/ automatically.
"""

import json
import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "api", "models", "artifacts")
GOLD_DIR = os.path.join(PROJECT_ROOT, "local_pipeline", "gold")


def copy_encoding_files():
    """Copy label_encoders.json and hcpcs_target_enc.json from gold to artifacts."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    for fname in ("label_encoders.json", "hcpcs_target_enc.json"):
        src = os.path.join(GOLD_DIR, fname)
        dst = os.path.join(ARTIFACTS_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            size_kb = os.path.getsize(dst) / 1024
            print(f"  Copied {fname} ({size_kb:.0f} KB)")
        else:
            print(f"  WARNING: {src} not found")


def export_from_mlflow():
    """Attempt to download model artifacts from Databricks MLflow."""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    except ImportError:
        pass

    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")

    if not host or not token:
        print("  DATABRICKS_HOST/TOKEN not set — skipping MLflow export.")
        print("  Place model files manually in api/models/artifacts/")
        return

    try:
        import mlflow
        mlflow.set_tracking_uri(f"databricks")

        # Find the best LightGBM run
        experiment_path = f"/Users/{os.path.expanduser('~').split(os.sep)[-1]}/medicare_models"
        print(f"  Searching MLflow experiment: {experiment_path}")

        experiment = mlflow.get_experiment_by_name(experiment_path)
        if experiment is None:
            print(f"  Experiment not found at {experiment_path}")
            return

        # Search for LightGBM runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.model_type = 'lightgbm'",
            order_by=["metrics.test_rmse ASC"],
            max_results=1,
        )
        if not runs.empty:
            run_id = runs.iloc[0]["run_id"]
            print(f"  Found LightGBM run: {run_id}")
            dst = os.path.join(ARTIFACTS_DIR, "lgbm_model.txt")
            mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="lgbm_model", dst_path=ARTIFACTS_DIR
            )
            print(f"  Downloaded LightGBM model")

        # Search for OOP quantile runs
        for q in ("p10", "p50", "p90"):
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.model_type = 'xgb_oop_{q}'",
                order_by=["metrics.{q}_mae ASC"],
                max_results=1,
            )
            if not runs.empty:
                run_id = runs.iloc[0]["run_id"]
                mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path=f"xgb_oop_{q}", dst_path=ARTIFACTS_DIR
                )
                print(f"  Downloaded XGBoost OOP {q}")

    except Exception as e:
        print(f"  MLflow export failed: {e}")
        print("  Place model files manually in api/models/artifacts/")


def check_artifacts():
    """Report which artifacts are present."""
    print("\nArtifact status:")
    expected = [
        "lgbm_model.txt",
        "xgb_oop_p10.ubj", "xgb_oop_p50.ubj", "xgb_oop_p90.ubj",
        "label_encoders.json", "hcpcs_target_enc.json",
    ]
    for fname in expected:
        path = os.path.join(ARTIFACTS_DIR, fname)
        if os.path.exists(path):
            size = os.path.getsize(path)
            unit = "KB" if size < 1_000_000 else "MB"
            val = size / 1024 if unit == "KB" else size / 1_048_576
            print(f"  OK  {fname} ({val:.1f} {unit})")
        else:
            print(f"  MISSING  {fname}")


if __name__ == "__main__":
    print("=== Medicare Model Export ===\n")

    print("Step 1: Copy encoding files from local_pipeline/gold/")
    copy_encoding_files()

    print("\nStep 2: Export models from MLflow (if credentials available)")
    export_from_mlflow()

    check_artifacts()

    print("\nDone. If model files are missing, place them manually:")
    print(f"  {ARTIFACTS_DIR}/")
