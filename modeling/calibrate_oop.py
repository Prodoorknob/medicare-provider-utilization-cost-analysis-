"""calibrate_oop.py — Compute CQR calibration constants for the production OOP models.

Reproduces V2_04's 60/20/20 split (RandomState(42)) on synthetic_oop.parquet, predicts
on the calibration slice using the production .cbm artifacts in api/models/artifacts/,
and saves both symmetric and asymmetric conformal-quantile-regression constants to
api/models/artifacts/oop_calibration.json.

Validates by reproducing V2_04's logged symmetric q_hat (MLflow: 3.6884).

Usage:
    python modeling/calibrate_oop.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "local_pipeline", "mcbs_synthetic", "synthetic_oop.parquet")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "api", "models", "artifacts")

OOP_FEATURES = [
    "Avg_Mdcr_Alowd_Amt", "Bene_Avg_Risk_Scre", "Rndrng_Prvdr_Type_idx",
    "hcpcs_bucket", "place_of_srvc_flag", "census_region",
    "age", "sex", "income", "chronic_count", "dual_eligible", "has_supplemental",
]
OOP_CAT_IDX = [2, 3, 4, 5]
TARGET = "per_service_oop"

ALPHA = 0.20  # 80% interval target — matches V2_04
EXPECTED_SYMMETRIC_Q_HAT = 3.6884  # from MLflow run 380046f73c7445b4815ef77808bea4d7


def load_models() -> dict[str, CatBoostRegressor]:
    models = {}
    for label in ("p10", "p50", "p90"):
        path = os.path.join(ARTIFACTS_DIR, f"oop_mono_{label}.cbm")
        if not os.path.exists(path):
            sys.exit(f"Missing artifact: {path}")
        m = CatBoostRegressor()
        m.load_model(path)
        models[label] = m
    return models


def make_pool(X: np.ndarray, y: np.ndarray | None = None) -> Pool:
    df = pd.DataFrame(X, columns=OOP_FEATURES)
    for i in OOP_CAT_IDX:
        df[OOP_FEATURES[i]] = df[OOP_FEATURES[i]].astype(int)
    return Pool(df, label=y, cat_features=OOP_CAT_IDX, feature_names=OOP_FEATURES)


def main() -> None:
    print(f"Loading {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH, columns=OOP_FEATURES + [TARGET])
    df = df.dropna(subset=OOP_FEATURES + [TARGET])
    n = len(df)
    print(f"  Rows after dropna: {n:,}")

    X = df[OOP_FEATURES].values.astype("float64")
    y = df[TARGET].values.astype("float64")
    del df

    # Reproduce V2_04 60/20/20 split
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    n_tr = int(n * 0.6)
    n_cal = int(n * 0.2)
    cal_idx = idx[n_tr:n_tr + n_cal]
    te_idx = idx[n_tr + n_cal:]

    X_cal, y_cal = X[cal_idx], y[cal_idx]
    X_test, y_test = X[te_idx], y[te_idx]
    print(f"  Cal: {len(X_cal):,}  Test: {len(X_test):,}")

    print("\nLoading production models from", ARTIFACTS_DIR)
    models = load_models()

    # ---- Predictions on calibration set ----
    print("\nPredicting on calibration set...")
    t0 = time.time()
    pool_cal = make_pool(X_cal)
    cal_p10_raw = models["p10"].predict(pool_cal)
    cal_p50 = models["p50"].predict(pool_cal)
    cal_p90_raw = models["p90"].predict(pool_cal)
    print(f"  done in {time.time()-t0:.0f}s")

    # Non-crossing — same convention as production API and V2_04
    cal_p10 = np.maximum(np.minimum(cal_p10_raw, cal_p50), 0)
    cal_p90 = np.maximum(cal_p90_raw, cal_p50)

    # ---- Symmetric CQR (V2_04 reproduction) ----
    cal_scores_sym = np.maximum(cal_p10 - y_cal, y_cal - cal_p90)
    n_cal_actual = len(cal_scores_sym)
    q_hat_sym = float(np.quantile(
        cal_scores_sym, (1 - ALPHA) * (1 + 1 / n_cal_actual)
    ))
    print(f"\nSymmetric q_hat: {q_hat_sym:.4f}  (V2_04 MLflow: {EXPECTED_SYMMETRIC_Q_HAT})")
    delta = abs(q_hat_sym - EXPECTED_SYMMETRIC_Q_HAT)
    if delta > 0.01:
        print(f"  WARNING: drift of {delta:.4f} from logged value — check split reproduction")
    else:
        print(f"  OK: matches V2_04 within {delta:.6f}")

    # ---- Asymmetric CQR (split alpha/2 on each tail) ----
    err_lo = cal_p10 - y_cal      # positive when y < p10 (lower miscoverage)
    err_hi = y_cal - cal_p90      # positive when y > p90 (upper miscoverage)
    level = 1 - ALPHA / 2          # 0.90 for 80% interval
    q_lo = float(np.quantile(err_lo, level * (1 + 1 / n_cal_actual)))
    q_hi = float(np.quantile(err_hi, level * (1 + 1 / n_cal_actual)))
    print(f"\nAsymmetric (per-tail at alpha/2={ALPHA/2}):")
    print(f"  q_lo (subtract from P10): {q_lo:+.4f}")
    print(f"  q_hi (add to P90):        {q_hi:+.4f}")

    # ---- Verify on test set ----
    print("\nValidating on held-out test set...")
    pool_test = make_pool(X_test)
    pred_p10_raw = models["p10"].predict(pool_test)
    pred_p50 = models["p50"].predict(pool_test)
    pred_p90_raw = models["p90"].predict(pool_test)
    pred_p10 = np.maximum(np.minimum(pred_p10_raw, pred_p50), 0)
    pred_p90 = np.maximum(pred_p90_raw, pred_p50)

    raw_cov = float(np.mean((y_test >= pred_p10) & (y_test <= pred_p90)))
    raw_p10_marginal = float(np.mean(y_test <= pred_p10))
    raw_p90_marginal = float(np.mean(y_test <= pred_p90))
    raw_width = float(np.mean(pred_p90 - pred_p10))

    sym_lo = np.maximum(pred_p10 - q_hat_sym, 0)
    sym_hi = pred_p90 + q_hat_sym
    sym_cov = float(np.mean((y_test >= sym_lo) & (y_test <= sym_hi)))
    sym_width = float(np.mean(sym_hi - sym_lo))

    asym_lo = np.maximum(pred_p10 - q_lo, 0)
    asym_hi = pred_p90 + q_hi
    asym_cov = float(np.mean((y_test >= asym_lo) & (y_test <= asym_hi)))
    asym_p10_marginal = float(np.mean(y_test <= asym_lo))
    asym_p90_marginal = float(np.mean(y_test <= asym_hi))
    asym_width = float(np.mean(asym_hi - asym_lo))

    print(f"\n{'method':<14} {'P10-P90 cov':>12} {'P10 marg':>10} {'P90 marg':>10} {'width $':>10}")
    print("-" * 60)
    print(f"{'raw':<14} {raw_cov:>12.4f} {raw_p10_marginal:>10.4f} {raw_p90_marginal:>10.4f} {raw_width:>10.2f}")
    print(f"{'symmetric':<14} {sym_cov:>12.4f} {'-':>10} {'-':>10} {sym_width:>10.2f}")
    print(f"{'asymmetric':<14} {asym_cov:>12.4f} {asym_p10_marginal:>10.4f} {asym_p90_marginal:>10.4f} {asym_width:>10.2f}")
    print(f"{'target':<14} {1-ALPHA:>12.4f} {ALPHA/2:>10.4f} {1-ALPHA/2:>10.4f} {'-':>10}")

    # ---- Save calibration sidecar ----
    out_path = os.path.join(ARTIFACTS_DIR, "oop_calibration.json")
    payload = {
        "version": 1,
        "alpha": ALPHA,
        "split": "60_20_20",
        "n_calibration": n_cal_actual,
        "n_test": len(X_test),
        "method_default": "asymmetric",
        "symmetric": {
            "q_hat": q_hat_sym,
            "test_coverage": sym_cov,
            "test_width": sym_width,
        },
        "asymmetric": {
            "q_lo": q_lo,
            "q_hi": q_hi,
            "test_coverage": asym_cov,
            "test_p10_marginal": asym_p10_marginal,
            "test_p90_marginal": asym_p90_marginal,
            "test_width": asym_width,
        },
        "raw_baseline": {
            "test_coverage": raw_cov,
            "test_p10_marginal": raw_p10_marginal,
            "test_p90_marginal": raw_p90_marginal,
            "test_width": raw_width,
        },
        "notes": (
            "Asymmetric CQR with alpha/2 per tail. Apply at inference: "
            "p10' = max(0, pred_p10 - q_lo); p90' = pred_p90 + q_hi. "
            "Then enforce non-crossing against P50."
        ),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved calibration to {out_path}")


if __name__ == "__main__":
    main()
