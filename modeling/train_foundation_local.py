"""
train_foundation_local.py — Zero-shot time-series forecasting with foundation models

Replaces/benchmarks against V1 LSTM for Medicare allowed amount forecasting.
Uses Chronos-Bolt and TimesFM pretrained models on the same sequences.parquet
with identical temporal split (train <= 2021, eval 2022-2023, forecast 2024-2026).

No gradient-based training — purely zero-shot inference.

Optional env vars (for MLflow logging):
    DATABRICKS_HOST   e.g. https://<workspace>.azuredatabricks.net
    DATABRICKS_TOKEN  Personal Access Token or service principal secret

Usage:
    python modeling/train_foundation_local.py --model chronos
    python modeling/train_foundation_local.py --model timesfm
    python modeling/train_foundation_local.py --model all
    python modeling/train_foundation_local.py --model chronos --batch-size 256 --num-samples 100
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SEQ      = os.path.join(_PROJECT_ROOT, "local_pipeline", "lstm", "sequences.parquet")
DEFAULT_ENCODERS = os.path.join(_PROJECT_ROOT, "local_pipeline", "gold", "label_encoders.json")
DEFAULT_OUTPUT   = os.path.join(_PROJECT_ROOT, "local_pipeline", "lstm")

GROUP_KEYS = [
    "Rndrng_Prvdr_Type_idx",
    "hcpcs_bucket",
    "Rndrng_Prvdr_State_Abrvtn_idx",
]

LSTM_BASELINE = {"test_mae": 8.84, "test_rmse": 36.42, "test_r2": 0.886}


# ---------------------------------------------------------------------------
# MLflow (optional)
# ---------------------------------------------------------------------------
def configure_databricks_mlflow() -> str:
    import requests, mlflow
    host  = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")
    if not host or not token:
        raise EnvironmentError("DATABRICKS_HOST and DATABRICKS_TOKEN must be set.")
    mlflow.set_tracking_uri("databricks")
    resp = requests.get(
        f"{host}/api/2.0/preview/scim/v2/Me",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    resp.raise_for_status()
    username = resp.json().get("userName", "unknown")
    return f"/Users/{username}"


# ---------------------------------------------------------------------------
# Data loading & preparation
# ---------------------------------------------------------------------------
def load_sequences(seq_path: str, min_years: int = 3) -> pd.DataFrame:
    df = pd.read_parquet(seq_path)
    before = len(df)
    df = df[df["n_years"] >= min_years].reset_index(drop=True)
    print(f"Loaded {before} groups, kept {len(df)} with >= {min_years} years")
    for col in GROUP_KEYS:
        df[col] = df[col].astype(int)
    return df


def prepare_eval_data(df: pd.DataFrame, train_end_year: int = 2021):
    """Split each series into context (<=2021) and ground truth (>2021).

    Foundation models operate on RAW dollar values (no log1p).
    """
    records = []
    for _, row in df.iterrows():
        years  = np.array(row["years"])
        values = np.array(row["target_seq"], dtype=np.float64)
        ptype  = row["Rndrng_Prvdr_Type_idx"]
        state  = row["Rndrng_Prvdr_State_Abrvtn_idx"]
        bucket = row["hcpcs_bucket"]

        train_mask = years <= train_end_year
        context    = values[train_mask]
        eval_mask  = years > train_end_year
        eval_years = years[eval_mask]
        eval_vals  = values[eval_mask]

        if len(context) < 2:
            continue

        records.append({
            "context":     context,
            "eval_years":  eval_years,
            "eval_values": eval_vals,
            "full_values": values,
            "all_years":   years,
            "n_eval":      len(eval_vals),
            "ptype":       ptype,
            "state":       state,
            "bucket":      bucket,
        })

    n_with_eval = sum(1 for r in records if r["n_eval"] > 0)
    print(f"Prepared {len(records)} groups ({n_with_eval} with 2022-2023 eval data)")
    return records


# ---------------------------------------------------------------------------
# Chronos-Bolt inference
# ---------------------------------------------------------------------------
def run_chronos(records, batch_size=512, num_samples=50, device="cuda"):
    import torch
    from chronos import BaseChronosPipeline

    print("\n=== Chronos-Bolt Inference ===")
    t0 = time.time()

    pipeline = BaseChronosPipeline.from_pretrained(
        "autogluon/chronos-bolt-base",
        device_map=device,
        torch_dtype=torch.float32,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # --- Evaluation pass: context=years<=2021, predict 2022-2023 ---
    print("  Evaluation pass (context <= 2021, predict 2022-2023)...")
    eval_preds = []
    eval_samples_all = []
    t1 = time.time()

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        contexts = [torch.tensor(r["context"], dtype=torch.float32) for r in batch]
        max_horizon = max(r["n_eval"] for r in batch if r["n_eval"] > 0)

        if max_horizon == 0:
            for r in batch:
                eval_preds.append(np.array([]))
                eval_samples_all.append(None)
            continue

        samples = pipeline.predict(
            contexts,
            prediction_length=max_horizon,
            num_samples=num_samples,
        )  # (B, num_samples, horizon)

        for i, r in enumerate(batch):
            h = r["n_eval"]
            if h == 0:
                eval_preds.append(np.array([]))
                eval_samples_all.append(None)
                continue
            s = samples[i, :, :h].cpu().numpy()  # (num_samples, h)
            s = np.clip(s, 0, None)
            eval_preds.append(np.median(s, axis=0))
            eval_samples_all.append(s)

        if (start // batch_size + 1) % 10 == 0:
            print(f"    Eval batch {start // batch_size + 1}/{(len(records) - 1) // batch_size + 1}")

    print(f"  Eval pass: {time.time() - t1:.1f}s")

    # --- Forecast pass: full context, predict 3 years ---
    print("  Forecast pass (full context, predict 2024-2026)...")
    forecasts = []
    t2 = time.time()

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        contexts = [torch.tensor(r["full_values"], dtype=torch.float32) for r in batch]

        samples = pipeline.predict(
            contexts,
            prediction_length=3,
            num_samples=num_samples,
        )

        for i, r in enumerate(batch):
            s = samples[i].cpu().numpy()  # (num_samples, 3)
            s = np.clip(s, 0, None)
            forecasts.append({
                "ptype":      r["ptype"],
                "state":      r["state"],
                "bucket":     r["bucket"],
                "samples":    s,
                "last_year":  int(r["all_years"][-1]),
                "last_value": float(r["full_values"][-1]),
                "n_history":  len(r["all_years"]),
            })

        if (start // batch_size + 1) % 10 == 0:
            print(f"    Forecast batch {start // batch_size + 1}/{(len(records) - 1) // batch_size + 1}")

    elapsed = time.time() - t0
    print(f"  Forecast pass: {time.time() - t2:.1f}s | Total: {elapsed:.1f}s")
    return eval_preds, forecasts, elapsed


# ---------------------------------------------------------------------------
# TimesFM inference
# ---------------------------------------------------------------------------
def run_timesfm(records, batch_size=512, device="cuda"):
    print("\n=== TimesFM 2.5 Inference ===")
    t0 = time.time()

    import timesfm

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained()
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # --- Evaluation pass ---
    print("  Evaluation pass (context <= 2021, predict 2022-2023)...")
    eval_preds = []
    t1 = time.time()

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        contexts = [r["context"].astype(np.float32) for r in batch]
        max_horizon = max(r["n_eval"] for r in batch if r["n_eval"] > 0)

        if max_horizon == 0:
            for _ in batch:
                eval_preds.append(np.array([]))
            continue

        point_forecasts, _ = model.forecast(
            inputs=contexts,
            freq=[0] * len(contexts),  # 0 = annual
            prediction_length=max_horizon,
        )

        for i, r in enumerate(batch):
            h = r["n_eval"]
            if h == 0:
                eval_preds.append(np.array([]))
                continue
            pred = np.clip(point_forecasts[i][:h], 0, None)
            eval_preds.append(pred)

        if (start // batch_size + 1) % 10 == 0:
            print(f"    Eval batch {start // batch_size + 1}/{(len(records) - 1) // batch_size + 1}")

    print(f"  Eval pass: {time.time() - t1:.1f}s")

    # --- Forecast pass ---
    print("  Forecast pass (full context, predict 2024-2026)...")
    forecasts = []
    t2 = time.time()

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        contexts = [r["full_values"].astype(np.float32) for r in batch]

        point_forecasts, _ = model.forecast(
            inputs=contexts,
            freq=[0] * len(contexts),
            prediction_length=3,
        )

        for i, r in enumerate(batch):
            pred = np.clip(point_forecasts[i][:3], 0, None)
            # TimesFM returns point forecasts; estimate uncertainty from residuals
            forecasts.append({
                "ptype":      r["ptype"],
                "state":      r["state"],
                "bucket":     r["bucket"],
                "point_pred": pred,  # (3,)
                "last_year":  int(r["all_years"][-1]),
                "last_value": float(r["full_values"][-1]),
                "n_history":  len(r["all_years"]),
            })

        if (start // batch_size + 1) % 10 == 0:
            print(f"    Forecast batch {start // batch_size + 1}/{(len(records) - 1) // batch_size + 1}")

    elapsed = time.time() - t0
    print(f"  Forecast pass: {time.time() - t2:.1f}s | Total: {elapsed:.1f}s")
    return eval_preds, forecasts, elapsed


# ---------------------------------------------------------------------------
# Evaluation (identical to LSTM evaluate() but on raw dollar values)
# ---------------------------------------------------------------------------
def evaluate_foundation(records, eval_preds, model_name):
    """Compute dollar-scale metrics matching LSTM evaluation exactly."""
    all_preds, all_targets = [], []

    for i, r in enumerate(records):
        if r["n_eval"] == 0:
            continue
        pred = np.clip(eval_preds[i][:r["n_eval"]], 0, None)
        all_preds.append(pred)
        all_targets.append(r["eval_values"][:r["n_eval"]])

    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    mae  = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2   = r2_score(targets, preds)
    n    = len(targets)

    print(f"\n  {model_name} Validation (2022-2023) — GROUP-LEVEL temporal predictions:")
    print(f"    N    = {n:,} group-year observations (specialty x bucket x state x year)")
    print(f"    MAE  = ${mae:,.2f}")
    print(f"    RMSE = ${rmse:,.2f}")
    print(f"    R2   = {r2:.4f}")
    print(f"  NOTE: Same evaluation granularity as LSTM (group-year means, NOT individual records).")

    return {"test_mae": mae, "test_rmse": rmse, "test_r2": r2, "eval_n_groups": n}


# ---------------------------------------------------------------------------
# Build forecast DataFrames (identical schema to LSTM output)
# ---------------------------------------------------------------------------
def build_forecast_df_chronos(forecasts):
    """Build forecast DataFrame from Chronos samples (native quantiles)."""
    rows = []
    for f in forecasts:
        samples = f["samples"]  # (num_samples, 3)
        for step in range(3):
            s = samples[:, step]
            rows.append({
                "Rndrng_Prvdr_Type_idx":          float(f["ptype"]),
                "hcpcs_bucket":                   float(f["bucket"]),
                "Rndrng_Prvdr_State_Abrvtn_idx":  float(f["state"]),
                "forecast_year":                  2024 + step,
                "forecast_mean":                  float(np.mean(s)),
                "forecast_std":                   float(np.std(s)),
                "forecast_p10":                   float(np.percentile(s, 10)),
                "forecast_p50":                   float(np.median(s)),
                "forecast_p90":                   float(np.percentile(s, 90)),
                "last_known_year":                f["last_year"],
                "last_known_value":               f["last_value"],
                "n_history_years":                f["n_history"],
            })
    return pd.DataFrame(rows)


def build_forecast_df_timesfm(forecasts, eval_records, eval_preds):
    """Build forecast DataFrame from TimesFM point forecasts.

    Since TimesFM gives point forecasts only, estimate P10/P90 from
    evaluation residual distribution (calibrated prediction intervals).
    """
    # Compute residual distribution from eval pass
    residuals = []
    for i, r in enumerate(eval_records):
        if r["n_eval"] == 0:
            continue
        pred = np.clip(eval_preds[i][:r["n_eval"]], 0, None)
        residuals.extend(pred - r["eval_values"][:r["n_eval"]])
    residuals = np.array(residuals)
    std_residual = np.std(residuals) if len(residuals) > 0 else 0.0
    p10_offset = np.percentile(residuals, 10) if len(residuals) > 0 else 0.0
    p90_offset = np.percentile(residuals, 90) if len(residuals) > 0 else 0.0
    print(f"  TimesFM residual calibration: std=${std_residual:.2f}, "
          f"P10_offset=${p10_offset:.2f}, P90_offset=${p90_offset:.2f}")

    rows = []
    for f in forecasts:
        pred = f["point_pred"]  # (3,)
        for step in range(3):
            p = float(pred[step])
            rows.append({
                "Rndrng_Prvdr_Type_idx":          float(f["ptype"]),
                "hcpcs_bucket":                   float(f["bucket"]),
                "Rndrng_Prvdr_State_Abrvtn_idx":  float(f["state"]),
                "forecast_year":                  2024 + step,
                "forecast_mean":                  p,
                "forecast_std":                   std_residual,
                "forecast_p10":                   max(0.0, p - p90_offset),  # flip sign
                "forecast_p50":                   p,
                "forecast_p90":                   max(0.0, p - p10_offset),
                "last_known_year":                f["last_year"],
                "last_known_value":               f["last_value"],
                "n_history_years":                f["n_history"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_comparison(all_metrics, output_dir, label_encoders=None):
    """Bar chart comparing foundation models vs LSTM baseline."""
    os.makedirs(output_dir, exist_ok=True)

    models = list(all_metrics.keys())
    mae_vals  = [all_metrics[m]["test_mae"]  for m in models]
    rmse_vals = [all_metrics[m]["test_rmse"] for m in models]
    r2_vals   = [all_metrics[m]["test_r2"]   for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # MAE
    colors = ["coral" if "LSTM" in m else "steelblue" for m in models]
    axes[0].barh(models, mae_vals, color=colors, edgecolor="white")
    axes[0].set_xlabel("MAE ($)")
    axes[0].set_title("Mean Absolute Error (lower is better)")
    for i, v in enumerate(mae_vals):
        axes[0].text(v + 0.3, i, f"${v:.2f}", va="center", fontsize=9)

    # RMSE
    axes[1].barh(models, rmse_vals, color=colors, edgecolor="white")
    axes[1].set_xlabel("RMSE ($)")
    axes[1].set_title("Root Mean Squared Error (lower is better)")
    for i, v in enumerate(rmse_vals):
        axes[1].text(v + 0.3, i, f"${v:.2f}", va="center", fontsize=9)

    # R2
    axes[2].barh(models, r2_vals, color=colors, edgecolor="white")
    axes[2].set_xlabel("R2")
    axes[2].set_title("R-Squared (higher is better)")
    axes[2].set_xlim(0, 1.05)
    for i, v in enumerate(r2_vals):
        axes[2].text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=9)

    fig.suptitle("Foundation Models vs LSTM — Temporal Forecast Evaluation (2022-2023)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "foundation_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_scatter(records, eval_preds, model_name, output_dir):
    """Predicted vs actual scatter plot."""
    os.makedirs(output_dir, exist_ok=True)
    all_p, all_t = [], []
    for i, r in enumerate(records):
        if r["n_eval"] == 0:
            continue
        all_p.append(np.clip(eval_preds[i][:r["n_eval"]], 0, None))
        all_t.append(r["eval_values"][:r["n_eval"]])
    preds   = np.concatenate(all_p)
    targets = np.concatenate(all_t)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(targets[::3], preds[::3], alpha=0.08, s=2, color="steelblue")
    mx = np.percentile(targets, 99)
    ax.plot([0, mx], [0, mx], "r--", alpha=0.6, linewidth=1)
    r2 = r2_score(targets, preds)
    ax.set_xlabel("Actual ($)", fontsize=11)
    ax.set_ylabel("Predicted ($)", fontsize=11)
    ax.set_title(f"{model_name}: R2={r2:.4f}", fontsize=12)
    ax.set_xlim(0, mx * 1.05)
    ax.set_ylim(0, mx * 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    path = os.path.join(output_dir, f"{safe_name}_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    print(f"Device target: {args.device}")
    print(f"Batch size: {args.batch_size}, Num samples: {args.num_samples}")

    # Load sequences
    df = load_sequences(args.data, min_years=args.min_years)
    records = prepare_eval_data(df, train_end_year=2021)

    # Load label encoders for visualization
    label_encoders = None
    if os.path.exists(args.label_encoders):
        with open(args.label_encoders) as f:
            label_encoders = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    all_metrics = {"LSTM (baseline)": LSTM_BASELINE}
    all_plots = []
    model_timings = {}

    # --- Chronos-Bolt ---
    if args.model in ("chronos", "all"):
        try:
            eval_preds, forecasts, elapsed = run_chronos(
                records, batch_size=args.batch_size,
                num_samples=args.num_samples, device=args.device,
            )
            metrics = evaluate_foundation(records, eval_preds, "Chronos-Bolt")
            all_metrics["Chronos-Bolt"] = metrics
            model_timings["Chronos-Bolt"] = elapsed

            fdf = build_forecast_df_chronos(forecasts)
            fpath = os.path.join(args.output_dir, "forecast_chronos_2024_2026.parquet")
            fdf.to_parquet(fpath, index=False)
            print(f"  Forecast saved -> {fpath} ({len(fdf):,} rows)")

            p = plot_scatter(records, eval_preds, "Chronos-Bolt", plot_dir)
            all_plots.append(p)

        except ImportError:
            print("\n  [SKIP] chronos-forecasting not installed. Install with:")
            print('    pip install "chronos-forecasting[gpu]"')
        except Exception as e:
            print(f"\n  [ERROR] Chronos-Bolt failed: {e}")
            import traceback; traceback.print_exc()

    # --- TimesFM ---
    if args.model in ("timesfm", "all"):
        try:
            eval_preds_tfm, forecasts_tfm, elapsed = run_timesfm(
                records, batch_size=args.batch_size, device=args.device,
            )
            metrics = evaluate_foundation(records, eval_preds_tfm, "TimesFM-2.5")
            all_metrics["TimesFM-2.5"] = metrics
            model_timings["TimesFM-2.5"] = elapsed

            fdf = build_forecast_df_timesfm(forecasts_tfm, records, eval_preds_tfm)
            fpath = os.path.join(args.output_dir, "forecast_timesfm_2024_2026.parquet")
            fdf.to_parquet(fpath, index=False)
            print(f"  Forecast saved -> {fpath} ({len(fdf):,} rows)")

            p = plot_scatter(records, eval_preds_tfm, "TimesFM-2.5", plot_dir)
            all_plots.append(p)

        except ImportError:
            print("\n  [SKIP] timesfm not installed. Install with:")
            print("    pip install timesfm")
        except Exception as e:
            print(f"\n  [ERROR] TimesFM failed: {e}")
            import traceback; traceback.print_exc()

    # --- Comparison ---
    if len(all_metrics) > 1:
        print("\n=== Foundation Model Comparison ===")
        print(f"{'Model':<20} {'MAE ($)':>10} {'RMSE ($)':>10} {'R2':>8}")
        print("-" * 52)
        for name, m in all_metrics.items():
            print(f"{name:<20} {m['test_mae']:>10.2f} {m['test_rmse']:>10.2f} {m['test_r2']:>8.4f}")

        # Best model
        non_lstm = {k: v for k, v in all_metrics.items() if "LSTM" not in k}
        if non_lstm:
            best_name = min(non_lstm, key=lambda k: non_lstm[k]["test_rmse"])
            best_m = non_lstm[best_name]
            lstm_m = LSTM_BASELINE
            print(f"\n  Best foundation model: {best_name}")
            print(f"    vs LSTM: MAE {'+' if best_m['test_mae'] > lstm_m['test_mae'] else ''}"
                  f"{best_m['test_mae'] - lstm_m['test_mae']:.2f}, "
                  f"R2 {'+' if best_m['test_r2'] > lstm_m['test_r2'] else ''}"
                  f"{best_m['test_r2'] - lstm_m['test_r2']:.4f}")

        # Comparison plot
        p = plot_comparison(all_metrics, plot_dir, label_encoders)
        all_plots.append(p)

    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, "foundation_metrics.json")
    save_metrics = {k: v for k, v in all_metrics.items() if "LSTM" not in k}
    save_metrics["_timings"] = model_timings
    with open(metrics_path, "w") as f:
        json.dump(save_metrics, f, indent=2)
    print(f"\nMetrics saved -> {metrics_path}")

    # --- MLflow (optional) ---
    if args.mlflow:
        try:
            import mlflow
            user_home = configure_databricks_mlflow()
            mlflow.set_experiment(f"{user_home}/medicare_models")

            for model_name, metrics in all_metrics.items():
                if "LSTM" in model_name:
                    continue
                safe_name = model_name.lower().replace(" ", "_").replace("-", "_").replace(".", "")
                with mlflow.start_run(run_name=f"{safe_name}_local"):
                    mlflow.log_params({
                        "model":            model_name,
                        "type":             "foundation_model",
                        "training":         "zero-shot (no training)",
                        "batch_size":       args.batch_size,
                        "num_samples":      args.num_samples if "chronos" in safe_name else "N/A",
                        "min_seq_length":   args.min_years,
                        "train_end_year":   2021,
                        "n_groups":         len(records),
                        "device":           args.device,
                        "source":           "local",
                    })
                    mlflow.log_metrics({
                        "test_mae":      metrics["test_mae"],
                        "test_rmse":     metrics["test_rmse"],
                        "test_r2":       metrics["test_r2"],
                        "eval_n_groups": metrics.get("eval_n_groups", 0),
                        "inference_time_s": model_timings.get(model_name, 0),
                    })
                    mlflow.log_param("eval_level",
                                     "group_temporal_2022_2023 — same as LSTM, NOT comparable to RF/XGB")
                    # Log artifacts
                    if os.path.exists(metrics_path):
                        mlflow.log_artifact(metrics_path, artifact_path="metrics")
                    for p in all_plots:
                        if os.path.exists(p):
                            mlflow.log_artifact(p, artifact_path="plots")
                    print(f"  MLflow run logged: {safe_name}_local")

        except EnvironmentError:
            print("\n  [WARN] MLflow env vars not set — skipping logging.")
            print("         Set DATABRICKS_HOST and DATABRICKS_TOKEN to enable.")
        except Exception as e:
            print(f"\n  [WARN] MLflow logging failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Foundation model time-series forecasting for Medicare allowed amounts"
    )
    parser.add_argument("--data", default=DEFAULT_SEQ,
                        help="Path to sequences.parquet")
    parser.add_argument("--label-encoders", default=DEFAULT_ENCODERS,
                        help="Path to label_encoders.json")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT,
                        help="Output directory for forecasts and plots")
    parser.add_argument("--model", choices=["chronos", "timesfm", "all"],
                        default="all", help="Which model(s) to run (default: all)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size for inference (default: 512)")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of samples for Chronos quantiles (default: 50)")
    parser.add_argument("--min-years", type=int, default=3,
                        help="Minimum sequence length (default: 3)")
    parser.add_argument("--device", default="auto",
                        help="Device: cuda, cpu, or auto (default: auto)")
    parser.add_argument("--mlflow", action="store_true",
                        help="Enable MLflow logging to Databricks (requires env vars)")
    args = parser.parse_args()

    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    main(args)
