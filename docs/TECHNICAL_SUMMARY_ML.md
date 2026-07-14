# AllowanceMap — Technical Summary 1: ML & Data Pipeline

**Role:** Sole developer — data engineering, modeling, evaluation, deployment.
**Client/Context:** Portfolio-scale healthcare ML on CMS Medicare Physician & Practitioner public data: 11 years (2013–2023), 126.8M provider-service rows, ~6,000 HCPCS procedure codes, all 50 states + territories, 1.76M unique providers.
**Status:** Live in production (Railway API + Vercel frontend) as of July 2026; modeling track closed.

## What it is
A two-stage cost-prediction system for Medicare services: Stage 1 predicts the Medicare allowed amount for a (specialty, procedure, state, provider-profile) combination; Stage 2 predicts the patient's out-of-pocket range as calibrated P10/P50/P90 quantiles. A third track forecasts specialty-level allowed amounts for 2024–2026. All three are served by a FastAPI real-time inference API that replaced pre-computed database lookups.

## Architecture
**CMS API → bronze (raw+year) → silver (typed, IQR-cleaned) → gold (10 engineered features, per-state parquet) → six model families → FastAPI (Railway) → Next.js (Vercel).**

- **Medallion pipeline, dual execution:** every stage has a Databricks/PySpark/Delta production variant and a pandas/pyarrow local variant sharing the same contracts; both log to a unified Databricks MLflow experiment. Region-batched training modes (incremental boosting / warm-start per census region) let 126.8M rows train on a 16GB consumer GPU.
- **Stage 1 — LightGBM, deliberately charge-free:** the charge-aware variant scored R² 0.9575, but API users don't reliably know the submitted charge, so the no-charge variant (R² 0.943, MAE ~$7 on a fair temporal holdout) ships. Ablation showed the charge feature bought only 0.01–0.02 R² — robustness beat marginal accuracy.
- **Stage 2 — CatBoost monotonic quantile regression** (P10/P50/P90) on a synthetic per-service OOP dataset bridged from the MCBS survey via a region×specialty crosswalk, with monotone constraints for product safety (predicted OOP can't decrease as allowed amount rises).
- **Forecast — stacked ensemble:** LightGBM stacker over LSTM, Chronos-Bolt, and history features (R² 0.8852 on 2022–23 temporal holdout, N=32,481). A multivariate Temporal Fusion Transformer scored 0.869, confirming a signal ceiling ≈0.885 at annual resolution rather than a modeling gap.

## Hard problems solved
- **Found and removed target leakage** pre-training: `Avg_Mdcr_Pymt_Amt` and `Avg_Mdcr_Stdzd_Amt` are arithmetically derived from the target and would have produced a fake near-1.0 R².
- **Fixed an inflated LSTM evaluation:** the original eval used 1-step teacher forcing; re-running as a true autoregressive rollout dropped R² from 0.886 to 0.869 and RMSE from 36.4 to 18.9 — the honest number is what's reported, and the stacker was benchmarked against it.
- **Calibrated Stage 2 intervals with asymmetric conformal (CQR) correction:** the raw P90 only covered 67.5% of held-out actuals; a per-quantile calibration sidecar (q_lo≈$0.0004, q_hi≈$14.47) applied at inference restores the nominal 90% coverage without retraining.
- **Red-teamed my own headline metric:** a two-round falsification study rebuilt the official CMS fee schedule (RVU×GPCI, modifier-aware) as a comparator. Round 1 suggested the model beat the fee schedule by ~$92 MAE on radiology; round 2's MOD-26-aware comparator collapsed that to ~$26 and showed the model's real skill is *implicit modifier inference* — on ~65% of volume it matches a fee-schedule lookup. The R² 0.943 is real but is now documented as largely re-deriving administered prices, which redefined what the model should be used for downstream.
- **Trained CatBoost monotonic on CPU by necessity** (CatBoost GPU doesn't support monotone constraints) and re-planned compute after discovering initial estimates were 3–5× low (3,000 iterations on 126.8M rows ≈ 105 min on an A100; a 5-fold ensemble took 13.3 hrs).

## Verified outcomes
- Stage 1 production model: R² 0.943, MAE ~$7 on temporal holdout across 126.8M rows.
- Forecast stacker: MAE $8.74 / R² 0.8852 vs LSTM 9.82/0.8689 and TFT 9.23/0.8691.
- Stage 2 interval coverage: 67.5% → 90% nominal after CQR calibration.
- Falsification: overall model-vs-fee-schedule gap on the audit sample = $10.39 vs $17.03 MAE (model ahead by ~$7, not the ~$18 originally claimed).

## Tech stack
Python · pandas · PyArrow · PySpark · Delta Lake · Databricks · MLflow · LightGBM · XGBoost · CatBoost · scikit-learn · PyTorch (LSTM, TFT) · Chronos-Bolt · conformal prediction (CQR) · FastAPI · Docker · Railway · Next.js · MUI · Vercel · CUDA/Colab (A100/T4)
