# MODELING.md — Medicare Cost Prediction Model Zoo

> Comprehensive reference for every model trained across Phases 1-8 of the Medicare Provider Cost & Patient OOP Prediction project. This document is the authoritative source for "which model does what, which one is in production, and why."
>
> Last updated: 2026-04-14 (end of Phase 8 — forecast track closed)
> See also: `PROGRESS.md` (chronological log), `V2_MODEL_SPEC.md` (original V2 implementation spec)

---

## Table of Contents

1. [Two-Stage Pipeline Overview](#two-stage-pipeline-overview)
2. [Production Models (Deploy These)](#production-models-deploy-these)
3. [Stage 1: Allowed Amount Prediction](#stage-1-allowed-amount-prediction)
4. [Stage 2: Out-of-Pocket (OOP) Quantile Regression](#stage-2-out-of-pocket-oop-quantile-regression)
5. [Forecast Track: 2024–2026 Prediction](#forecast-track-20242026-prediction)
6. [Notable Negative Results](#notable-negative-results)
7. [Known Bugs & Measurement Issues](#known-bugs--measurement-issues)
8. [Full Feature Inventory](#full-feature-inventory)
9. [Compute Spend Summary](#compute-spend-summary)
10. [Backlog](#backlog)

---

## Two-Stage Pipeline Overview

The project delivers two independent but composable models:

```
         ┌────────────────────────┐      ┌────────────────────────┐
User  →  │  Stage 1: Allowed Amt  │  →   │  Stage 2: Patient OOP  │  → $
input    │  LightGBM V2 no-charge │      │  XGB Quantile V1       │
         │  R² = 0.9428           │      │  P50 R² = 0.40         │
         └────────────────────────┘      └────────────────────────┘
                                         P10 / P50 / P90 distribution

Plus (independent track):

         ┌─────────────────────────────────┐
Year  →  │  Forecast: 2024-2026 projections│
         │  LGB Stacker V2_12 (LSTM+Chronos)│
         │  R² = 0.8852                     │
         └─────────────────────────────────┘
```

Stage 1 predicts **what Medicare pays** for a given (specialty × HCPCS × state × POS) combination. Stage 2 takes Stage 1's output and predicts **how much of that the patient pays out-of-pocket** given their demographic and coverage context. The forecast track is separate — it projects allowed amounts into future years.

---

## Production Models (Deploy These)

| Purpose | Model | Run Name (MLflow) | Metrics | Artifact |
|---|---|---|---|---|
| **Stage 1** (allowed amount, no-charge) | LightGBM V2 no-charge | `lgbm_v2_no_charge_colab` | R² 0.9428, MAE $7.70, RMSE $15.77 | Deployed via Railway API (pickled LightGBM booster) |
| **Stage 2** (OOP quantile regression) | XGB Quantile V1 | `xgb_quantile_oop_local` | P50 R² 0.40, P50 cov 50.0%, P90 cov 90.0% | `oop_p10.xgb`, `oop_p50.xgb`, `oop_p90.xgb` on Railway |
| **Forecast** (2024-2026 projections) | LGB Stacker V2_12 | `lgb_stacker_v2_12_colab` | R² 0.8852, MAE $8.74, RMSE $17.69 | `stacker_forecast_2024_2026.parquet` in Drive `v2_artifacts/predictions/` |

**Why not the absolute best metric in each track:**
- **Stage 1 full-charge LightGBM (R² 0.9575)** beats no-charge by +0.015 R², but requires `Avg_Sbmtd_Chrg` as input. The Railway API serves user estimates where the submitted charge is not known at request time, so the no-charge variant is the production deployment despite the small metric gap.
- **Ensemble V2 (R² 0.9580)** beats LightGBM V2 full-charge by only +0.0005 R² for 13.3 hrs of 5-fold compute. Not worth the operational complexity.
- **Forecast Stacker (R² 0.8852)** beats fair LSTM (0.8689), Chronos (0.8576), and multivariate TFT (0.8691). See [Forecast Track](#forecast-track-20242026-prediction) for why the stacker won and why we stopped there.

---

## Stage 1: Allowed Amount Prediction

**Target:** `Avg_Mdcr_Alowd_Amt` (dollar value per group-year)
**Evaluation level:** individual group-HCPCS rows (126.8M in V2 full training, 30% sample in V1)
**Split:** 80/20 random, `random_state=42`
**Target transform:** `log1p` then inverse at prediction time

### V2 Stage 1 (Colab Pro, full 126.8M rows) — Final Results

| Model | MAE | RMSE | R² | Status |
|---|---:|---:|---:|---|
| **LightGBM V2 (full)** | **$6.73** | **$13.59** | **0.9575** | Best-in-class, charge-required |
| Ensemble V2 (5-fold stack) | $6.68 | $13.50 | 0.9580 | +0.0005 over LGB → not deployed |
| XGBoost V2 | $7.73 | $15.43 | 0.9452 | Fallback |
| CatBoost V2 | $10.88 | $20.10 | 0.9070 | Weakest of the three; kept for ensemble diversity |
| **LightGBM V2 no-charge** 🏆 | **$7.70** | **$15.77** | **0.9428** | **PRODUCTION** |
| XGBoost V2 no-charge | $8.66 | $17.19 | 0.9319 | — |
| CatBoost V2 no-charge | $12.10 | $22.34 | 0.8849 | — |

### V1 Stage 1 (local, 30% sample, regional batching) — Archived

| Model | R² | Notes |
|---|---:|---|
| RF V1 | 0.8843 | Best V1 model; regional batch + warm_start |
| XGBoost V1 | 0.8331 | Incremental by Census region |
| GLM V1 | diverged | SGD baseline needed more tuning |

V2 dominated V1 by +0.07 R² across the board — full data was the single biggest lever, not any architectural change.

### Stage 1 Feature Set (13 features, or 12 without charge)

1. `Rndrng_Prvdr_Type_idx` — encoded specialty
2. `Rndrng_Prvdr_State_Abrvtn_idx` — encoded state
3. `HCPCS_Cd_idx` — encoded raw HCPCS (~6K unique)
4. `hcpcs_bucket` — coarse clinical category (0-5)
5. `place_of_srvc_flag` — binary facility vs office
6. `Bene_Avg_Risk_Scre` — NPI-level HCC risk score
7. `log_srvcs` — log1p(Tot_Srvcs)
8. `log_benes` — log1p(Tot_Benes)
9. `Avg_Sbmtd_Chrg` — submitted charge (**dropped in no-charge variant**)
10. `srvcs_per_bene` — services per beneficiary
11. `specialty_bucket` — coarse specialty grouping
12. `pos_bucket` — place-of-service bucket
13. `hcpcs_target_enc` — HCPCS target encoding

### Key Takeaway: The No-Charge Ablation

Removing `Avg_Sbmtd_Chrg` only drops R² by 0.01–0.02. This means the V2 LightGBM is **genuinely predictive via procedure identity and provider features**, not via a near-leaky charge signal (which dominated V1 at 61.8% feature importance). The no-charge model is therefore robust and deployable in a user-facing API where charge is unknown.

---

## Stage 2: Out-of-Pocket (OOP) Quantile Regression

**Target:** Patient out-of-pocket cost at three quantiles (P10, P50, P90)
**Data source:** Synthetic MCBS (Track B — real MCBS LDS is constrained by privacy agreements)
**Evaluation:** MAE, RMSE, R² at each quantile + coverage (empirical vs nominal quantile rate)

### Stage 2 Results — V1 Wins

| Model | P50 MAE | P50 RMSE | P50 R² | P50 Coverage | P90 Coverage | Status |
|---|---:|---:|---:|---:|---:|---|
| **XGB Quantile V1** 🏆 | **$9.78** | **$18.28** | **0.400** | **50.0%** | **90.0%** | **PRODUCTION** |
| CatBoost Mono V2 | $10.55 | $21.34 | 0.173 | — | — | Monotonicity too restrictive |
| CatBoost ZI V2 | $11.95 | $24.15 | -0.054 | — | — | Gate+regression compound errors |

### Why V2 Lost on OOP

Both V2 Stage 2 variants were well-motivated in theory but failed on synthetic data:

- **V2_04 CatBoost Monotonic** applied 5 domain constraints (allowed ↑ → OOP ↑, income ↑ → OOP ↑, chronic ↑ → OOP ↑, dual ↓ OOP, supplemental ↓ OOP) plus non-crossing quantiles via post-hoc sorting and Conformalized Quantile Regression (CQR) calibration on a 60/20/20 split. The constraints were correct a priori but **fought the synthetic data distribution**, which didn't perfectly obey them.
- **V2_05 Zero-Inflated** split the problem into a gate classifier (P(OOP=0)) and conditional regression. Errors from the gate classifier compounded into the regression and the combined prediction was worse than a single-model baseline.

**Interpretation:** On real MCBS LDS data with actual patient demographics and true OOP distributions, monotonicity constraints would likely help. Synthetic data makes architectural innovation look worse than it is. Lesson: don't over-engineer on proxy data.

### Stage 2 Feature Set (12 features)

Core Stage 1 output becomes a Stage 2 input:

1. `Avg_Mdcr_Alowd_Amt` (Stage 1 prediction) — **connects the two stages**
2. `Bene_Avg_Risk_Scre`
3. `specialty_bucket`
4. `hcpcs_bucket`
5. `place_of_srvc_flag`
6. `census_region`
7. `age_band`
8. `sex`
9. `income_band`
10. `n_chronic_conditions`
11. `dual_eligible` (Medicaid)
12. `has_supplemental_insurance` (Medigap)

---

## Forecast Track: 2024–2026 Prediction

**Target:** `Avg_Mdcr_Alowd_Amt` per (ptype × hcpcs_bucket × state) group, projected forward 3 years
**Evaluation:** 2022-2023 temporal holdout, aggregated to **group-year means** (32,481 group-year observations, 16,240 unique groups)
**Note:** Forecast R² is NOT directly comparable to Stage 1 R² — forecasting evaluates smoother group-level aggregates, not individual rows. The 0.885 forecast R² and 0.943 Stage 1 R² answer different questions.

### Evolution of the Forecast Track

| Phase | Notebook | Model | R² | What changed | Session |
|---|---|---|---:|---|---|
| 3 | `train_lstm_local.py` | LSTM V1 | 0.8860 ⚠ | 2-layer LSTM, 64 hidden, static embeds | Apr 8 |
| 7 | `V2_06` | Univariate TFT | 0.846 | Target-only TFT, 6-year encoder | Apr 11 |
| 8 | `V2_09` | Chronos-T5-Large | 0.8485 | Zero-shot foundation model | Apr 11 |
| 8 | `V2_10` | Chronos-Bolt raw | 0.8563 | Bolt architecture swap | Apr 12 |
| 8 | `V2_11` | Chronos-Bolt cpi_cf_deflated | 0.8576 | CPI+CF deflation, reinflate with known future | Apr 13 |
| 8 | `V2_12` | **LGB Stacker** 🏆 | **0.8852** | **Ensemble LSTM + Chronos** | **Apr 14** |
| 8 | `V2_13` | Multivariate TFT | 0.8691 | 5 observed + 3 known-future covariates | Apr 14 |

⚠ LSTM V1 0.8860 was measured with teacher forcing. Fair autoregressive re-eval in V2_12 Cell 6 gave **LSTM V1 = 0.8689**, which is the honest baseline. See [Known Bugs](#known-bugs--measurement-issues).

### Final Leaderboard (fair 2022-2023 autoregressive holdout)

| Model | MAE | RMSE | R² | RMSE/MAE |
|---|---:|---:|---:|---:|
| LSTM V1 (fair, AR) | $9.82 | $18.91 | 0.8689 | 1.93 |
| Chronos-Bolt cpi_cf_deflated | $9.39 | $19.71 | 0.8576 | 2.10 |
| Multivariate TFT V2_13 | $9.23 | $18.79 | 0.8691 | 2.04 |
| **LGB Stacker V2_12** 🏆 | **$8.74** | **$17.69** | **0.8852** | **2.02** |

### Forecast Stacker V2_12 Architecture

**Base models (both re-trained or re-run inside V2_12 for self-containment):**
1. **LSTM V1** — retrained in-notebook on Colab A100 (~5 min), autoregressive eval from context ≤ 2021
2. **Chronos-Bolt-Base** — cpi_cf_deflated variant from V2_11 (~2 min inference)

**Meta-learner:** LightGBM, 13 features, 1000 boost rounds with early stopping on 5-fold GroupKFold CV.

**Stacker features** (with V2_12 feature importance by gain %):

| Feature | Gain % | Role |
|---|---:|---|
| `lstm_pred` | 70.48 | LSTM autoregressive prediction |
| `last_history_value` | 15.97 | Persistence anchor (year ≤ 2021) |
| `chronos_pred` | 3.63 | Chronos cpi_cf_deflated median |
| `history_mean` | 2.67 | Mean of history |
| `Rndrng_Prvdr_Type_idx` | 1.91 | Specialty (categorical) |
| `history_cv` | 1.62 | Volatility conditioning |
| `history_trend` | 1.18 | Linear trend slope |
| `Rndrng_Prvdr_State_Abrvtn_idx` | 1.16 | State (categorical) |
| `n_history_years` | 0.77 | Context length conditioning |
| `hcpcs_bucket` | 0.31 | Clinical bucket (categorical) |
| `forecast_year` | 0.27 | 2022/2023 flag |
| `cpi_factor` | 0.02 | Dead (already baked into Chronos preprocessing) |
| `cf_factor` | 0.00 | Dead |

**Cross-validation:** 5-fold GroupKFold on `(ptype, bucket, state)` tuple hash. Prevents leakage where a group's 2022 observation could be in train and its 2023 observation in test. OOF predictions produced for unbiased metrics; final model refit on all 2022-2023 data and applied to pre-computed LSTM + Chronos 2024-2026 forecasts.

**Why the stacker wins (despite similar base-model error shapes):**
1. **Prediction-level diversity** — LSTM and Chronos disagree on which specific group-years are hard; blending cancels independent errors
2. **History conditioning** — the stacker learns *when* to trust each base model based on `history_cv`, `history_trend`, `n_history_years`

**Why the lift is capped at ~0.016 R²:**
Under fair comparison, base models have near-identical error distributions (RMSE/MAE ratios 1.9–2.1 across the board). The "complementary error profiles" hypothesis from early V2_09–V2_11 writeups was wrong — that was a teacher-forcing artifact in the LSTM baseline. True diversity between LSTM and Chronos is limited.

### Forecast 2024-2026 Output Schema

`stacker_forecast_2024_2026.parquet` (LSTM-compatible schema):

```
Rndrng_Prvdr_Type_idx          int
hcpcs_bucket                    int
Rndrng_Prvdr_State_Abrvtn_idx   int
forecast_year                   int   (2024, 2025, 2026)
forecast_mean                   float
forecast_std                    float  (0.0 — stacker is point-only)
forecast_p10                    float  (= forecast_mean in stacker)
forecast_p50                    float  (= forecast_mean in stacker)
forecast_p90                    float  (= forecast_mean in stacker)
last_known_year                 int   (2023)
last_known_value                float (2023 actual)
n_history_years                 int
```

**Stacker forecast summary (20,572 groups × 3 years = 61,716 rows):**

| Year | Count | Mean | Median | Std |
|---:|---:|---:|---:|---:|
| 2024 | 20,572 | $72.19 | $64.10 | 50.95 |
| 2025 | 20,572 | $61.54 | $53.72 | 43.51 |
| 2026 | 20,572 | $61.70 | $54.15 | 44.07 |

The $10 drop from 2024 → 2025 is mean-reversion inherited from the underlying LSTM autoregressive forecast. V2_13 TFT produces a more coherent trajectory ($70 / $68 / $68) but has lower validation R², so the stacker ships.

---

## Notable Negative Results

### ❌ Charge-Ratio Derived Series (V2_10)
Dividing target by submitted charge created a "charge-normalized" univariate series for Chronos. Shape correlation with raw was 0.14 (genuinely different). **Result: R² 0.1937** — back-transforming through `submitted_charge` at eval time amplified forecast error.

### ❌ Volume-Normalized Derived Series (V2_10)
Same idea with `log_srvcs` denominator. **Result: R² −540.24.** Catastrophic. Dividing then re-multiplying by forecasted volume compounds error multiplicatively.

### ❌ CPI-Only Deflation (V2_11)
Deflate by medical CPI, forecast residual, reinflate. **Result: R² 0.8473 — WORSE than raw (0.8563).** CPI rises steadily; fee schedules don't track it. Deflating by CPI alone over-corrects and removes real signal. CF+CPI combined was the only variant that helped.

### ❌ Sequestration-Adjusted Variant (V2_11)
Reverse the 2% sequestration cut, then apply CPI deflation. **Result: R² 0.8528** — worse than cpi_cf_deflated. Sequestration is uniform ~2% so reversing it adds noise without exposing signal.

### ❌ Risk-Adjusted Derived Series (V2_10)
Divide target by risk score. **Result: R² 0.8563 — identical to raw.** `Bene_Avg_Risk_Scre` is imputed to ~1.0 for most groups so the ratio is a no-op.

### ❌ Multivariate TFT (V2_13)
The biggest negative result of the track. Added 5 observed covariates (log_srvcs, log_benes, Avg_Sbmtd_Chrg, Bene_Avg_Risk_Scre, srvcs_per_bene) as `time_varying_unknown_reals` plus 3 known-future reals (CPI, CF, covid_indicator) as decoder inputs to a TemporalFusionTransformer. **Result: R² 0.8691** — tied fair LSTM, lost to stacker. At annual resolution these covariates move in near-lockstep with the target; their past trajectories carry no signal orthogonal to the target's own history. Multivariate hypothesis: rejected for this data shape.

### ❌ Univariate TFT (V2_06)
First attempt at TFT, but with target only (no covariates). **Result: R² 0.846** — lost to LSTM. The architecture was fine; the input was starved. V2_13 confirmed TFT ties LSTM when given the same univariate signal, so TFT was never the problem.

### ❌ Hierarchical Reconciliation (V2_07)
Bottom-up forecasts were already coherent; MinTrace-style reconciliation added no adjustment. Coherence check passed but no metric change.

### ❌ TimesFM 2.5 (V2_09)
`pip install timesfm` failed in Colab during V2_09 setup. Did not evaluate.

---

## Known Bugs & Measurement Issues

### 🐛 Teacher-forcing bug in `modeling/train_lstm_local.py` `evaluate()` (line 340)

**Impact:** The reported LSTM V1 baseline (R² 0.886, RMSE $36.42) was measured with 1-step-ahead teacher forcing, not autoregressive rollout. During eval, the function fed the TRUE value at each position and asked the model to predict one step ahead — so when predicting 2023, the LSTM saw the TRUE 2022 value in the input sequence. At inference time (2024-2026 forecast), the LSTM has to feed its own predictions back autoregressively, which compounds error and produces meaningfully worse numbers.

**Measured difference:**
- Reported (teacher-forced): MAE $8.84, **RMSE $36.42**, R² 0.886
- Fair (autoregressive): MAE $9.82, **RMSE $18.91**, R² 0.8689

The RMSE difference of $17.51 is large enough to have misled the entire V2_09–V2_11 analysis. Early writeups described "LSTM has heavy right tail, Chronos has half the RMSE" — that was an artifact, not a real property. Under fair comparison all base models have RMSE/MAE ratios in 1.9–2.1 (near-Gaussian).

**Fix (backlog):** Modify `evaluate()` to accept `--eval-mode {teacher_forced, autoregressive}` and report both. V2_12 Cell 6 already contains a working batched autoregressive rollout that can be adapted.

### ⚠ Stage 1 R² vs Forecast R² — NOT directly comparable

Stage 1 LightGBM (R² 0.9575) evaluates on **individual HCPCS-row predictions** (millions of rows). Forecast models (R² 0.8852) evaluate on **group-year means** (32K aggregated observations). Group-year means are smoother than individual rows, so R² thresholds are different. Do not compare these numbers head-to-head in presentations — they answer different questions. PROGRESS.md notes this explicitly in Phase 7.

### ⚠ `gold/` vs `gold_year/` on Drive

The `gold/` directory on the project's Google Drive has parquets **without the `year` column** (year was stripped at some aggregation step). The `gold_year/` directory has year-aware parquets. V2_13 originally pointed at `gold/` and hit an Arrow schema error. Always use `gold_year/` for any notebook that needs temporal granularity.

---

## Full Feature Inventory

### Stage 1 Features (Gold parquets)

| Column | Dtype | Description |
|---|---|---|
| `Rndrng_Prvdr_Type_idx` | int | Encoded provider specialty |
| `Rndrng_Prvdr_State_Abrvtn_idx` | int | Encoded state |
| `HCPCS_Cd_idx` | int | Encoded HCPCS code (~6K unique) |
| `hcpcs_bucket` | int | Clinical bucket: 0=Anesthesia, 1=Surgery, 2=Radiology, 3=Lab, 4=Medicine/E&M, 5=HCPCS Level II |
| `place_of_srvc_flag` | int | 1=facility, 0=office |
| `Bene_Avg_Risk_Scre` | float | NPI-level HCC risk score (median imputed to ~1.0) |
| `log_srvcs` | float | log1p(Tot_Srvcs) |
| `log_benes` | float | log1p(Tot_Benes) |
| `Avg_Sbmtd_Chrg` | float | Submitted charge (dropped in no-charge variant) |
| `srvcs_per_bene` | float | Services per beneficiary |
| `specialty_bucket` | int | Coarse specialty grouping |
| `pos_bucket` | int | Place-of-service bucket |
| `hcpcs_target_enc` | float | HCPCS target encoding |
| `year` | int16 | 2013–2023 (only in `gold_year/`, not `gold/`) |
| `Avg_Mdcr_Alowd_Amt` | float | **TARGET** |

### Stage 2 Features (MCBS + Stage 1 output)

| Column | Dtype | Source |
|---|---|---|
| `Avg_Mdcr_Alowd_Amt` | float | Stage 1 output |
| `Bene_Avg_Risk_Scre` | float | Provider data |
| `specialty_bucket` | int | Gold |
| `hcpcs_bucket` | int | Gold |
| `place_of_srvc_flag` | int | Gold |
| `census_region` | str | MCBS |
| `age_band` | str | MCBS derived |
| `sex` | int | MCBS |
| `income_band` | int | MCBS derived |
| `n_chronic_conditions` | int | MCBS derived |
| `dual_eligible` | bool | MCBS derived |
| `has_supplemental_insurance` | bool | MCBS derived |

### Forecast Features (`sequences.parquet` + `gold_year/`)

Univariate forecasts (LSTM, Chronos) use **target only** (`Avg_Mdcr_Alowd_Amt` history) plus static group IDs.

Multivariate forecast (V2_13 TFT) adds:
- Static: `provider_type`, `state`, `hcpcs_bucket`
- Time-varying known: `conversion_factor`, `cpi_medical`, `covid_indicator`
- Time-varying unknown: `log_srvcs`, `log_benes`, `Avg_Sbmtd_Chrg`, `Bene_Avg_Risk_Scre`, `srvcs_per_bene`

External covariate tables (`{DRIVE}/external/`):
- `conversion_factors.csv` — CMS published 2013–2026
- `medical_cpi.csv` — BLS medical CPI 2013–2026
- `sequestration_rates.csv` — policy timeline 2013–2026
- `covid_indicators.csv` — binary flag 2020–2021
- `macra_mips.csv` — adjustment factors (used in V2_06 only)

---

## Compute Spend Summary

### Phase 7 V2 (Colab Pro A100 @ 11.2 CU/hr, 300 CU budget)

| Notebook | Duration | CU |
|---|---|---:|
| V2_01 (Stage 1 full) | ~4 hrs | ~45 |
| V2_02 (Stage 1 no-charge) | ~3 hrs | ~34 |
| V2_03 (Stage 1 5-fold ensemble) | ~13 hrs | ~146 |
| V2_04 (CatBoost Monotonic OOP) | ~2 hrs | ~22 |
| V2_05 (Zero-inflated OOP) | ~1.5 hrs | ~17 |
| V2_06 (TFT univariate) | ~3 hrs | ~34 |
| V2_07 (Hierarchical) | ~1 hr | ~11 |
| V2_08 (Comparison) | ~0.5 hrs | ~6 |
| **Phase 7 subtotal** | | **~315 CU** |

Phase 7 ran slightly over the 300 CU budget (see `feedback_compute_estimates.md` in project memory).

### Phase 8 (Apr 11–14, Colab Pro A100)

| Notebook | Duration | CU (est.) |
|---|---|---:|
| V2_09 (Chronos-T5-Large) | ~15 min | ~3 |
| V2_10 (Chronos-Bolt derived features) | ~25 min | ~5 |
| V2_11 (Chronos-Bolt 5 deflation variants) | ~35 min | ~7 |
| V2_12 (LightGBM stacker, self-contained with LSTM retrain) | ~20 min | ~4 |
| V2_13 (Multivariate TFT) | ~60 min | ~12 |
| **Phase 8 subtotal** | | **~31 CU** |

Phase 8 was much cheaper than Phase 7 because Chronos inference is fast and the stacker/TFT trained in minutes rather than hours.

### Lesson Learned (from `feedback_compute_estimates.md`)

Phase 7 training estimates were consistently 3-5× too low. CatBoost 3000 iters on 126M rows took ~100 min vs predicted ~20 min. Ensemble 5-fold took 13.3 hrs vs predicted ~4 hrs. **Always pad compute estimates 3-5× for first-time-on-this-data runs.**

---

## Backlog

Prioritized, with estimated effort and expected outcome:

### High-priority (measurement & deployment hygiene)
1. **Fix `train_lstm_local.py evaluate()` teacher-forcing bug** — add `--eval-mode` flag, report both teacher-forced and autoregressive R². ~1 hour. Prevents future measurement confusion.
2. **Swap Railway API forecast endpoint** to serve `stacker_forecast_2024_2026.parquet` instead of LSTM V1 forecast. ~30 min including smoke test. See `FRONTEND_TODOS.md`.
3. **Update Supabase `lstm_forecasts` table** with stacker forecast rows (same schema, different values). ~30 min.

### Medium-priority (minor follow-ups)
4. **Quantile stacker variant** — 20-line edit to V2_12 Cell 10 to train 3 LightGBM boosters at `alpha={0.1, 0.5, 0.9}` for real P10/P50/P90 bounds. Adds genuine uncertainty UI to the frontend. ~2 hours.
5. **Update web app model comparison page** with V2 full results + Phase 8 forecast track leaderboard. ~2 hours.

### Low-priority (research-level, may never ship)
6. **Fine-tune Chronos-Bolt on Medicare sequences** — zero-shot hit ceiling at R² 0.8576. Even light fine-tuning (LoRA on T5 backbone) might close the gap to the stacker. ~1 day.
7. **Quarterly data ingestion** — the only known lever that could push forecast R² above 0.89. Evaluate whether CMS publishes MUP-PS quarterly aggregates; if yes, rebuild `sequences.parquet` at quarterly resolution (44 timesteps per group instead of 11), which unlocks CNN/TCN/PatchTST architectures and meaningful Chronos fine-tuning. **2-4 weeks of data engineering.** Target: R² 0.91–0.93.
8. **Real MCBS LDS data for Stage 2** — replace synthetic OOP with actual patient-level OOP from the LDS cohort. Would likely revive CatBoost Monotonic V2 because real data may obey the domain constraints synthetic data violates.

---

**Document maintenance:** Update this file whenever a new model is trained, a model is retired, or a production deployment changes. Keep the "Production Models" table at the top in sync with what's actually running in Railway/Vercel.
