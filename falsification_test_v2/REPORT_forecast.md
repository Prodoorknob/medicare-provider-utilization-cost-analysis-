# Forecast Track Falsification: 2024 LSTM forecast vs PFS policy reality

**Date:** 2026-04-27
**Forecast file under test:** `local_pipeline/lstm/forecast_2024_2026.parquet` (LSTM autoregressive forecast at specialty x bucket x state granularity, 62,703 rows). Production stacker output (`stacker_forecast_2024_2026.parquet`) is in Google Drive only and not available locally for testing. The LSTM is the dominant feature in the V2_12 stacker (70.5% feature importance per CLAUDE.md), so this test characterizes ~70% of the production forecast signal.
**Ground truth status:** the actual 2024 CMS PUF (Medicare Physician & Other Practitioners by Provider and Service) is not yet released to the local repo. CMS publishes with a roughly 2-year lag, so a true forecast accuracy test is not possible right now. Instead, this is a **policy-consistency test**: does the forecast track the known 2024 PFS rate policy?

## What changed in the 2024 PFS that the forecast should know about

The Medicare conversion factor (CF) is the single largest driver of nominal payment-rate change year-over-year for codes whose RVUs are stable.

| File                                | CF        | vs 2023      |
| ----------------------------------- | --------- | -----------: |
| 2023 (final)                        | $33.8872  |              |
| 2024A (March 2024 release)          | $32.7442  | -3.37%       |
| **2024B (September 2024 final)**    | $33.2875  | **-1.77%**   |
| 2025                                | $32.3465  | -4.55% vs 2023 |

The 2024A → 2024B revision was a Congressional partial restoration of the cut. Any annual-level forecast for 2024 is fairly compared against the 2024B (final) rate.

**Net effect of 2023 to 2024 PFS policy on a constant code:** payments fell by roughly 1 to 3 percent for active codes, with bucket-specific variation from RVU updates (radiology had additional downward adjustments).

## Test 1: LSTM forecast YoY 2024 vs PFS-implied YoY 2024

I built a **PFS-implied baseline** for each (state, bucket) cell by:
1. Finding the top 80 HCPCS codes by 2023 service volume in that (state, bucket) cell from the silver layer.
2. Pricing each code in 2023 and 2024B at the modifier-aware Global rate, averaged over all the localities in the state.
3. Volume-weighting the per-code 2024/2023 ratio.

Then I compared each LSTM forecast row's `forecast_mean_2024 / last_known_value - 1` against the PFS-implied YoY change for the same (state, bucket).

### Headline numbers (n=15,754 cells, all `last_known_year=2023`)

| metric                              | LSTM forecast YoY 2024 | PFS-implied YoY 2024 |
| ----------------------------------- | ---------------------: | -------------------: |
| **median**                          | **+2.00%**             | **-0.85%**           |
| mean                                | +2.13%                 | -0.97%               |
| p10                                 | -12.07%                | -2.43%               |
| p90                                 | +13.48%                | +0.30%               |
| Pearson r between the two           | **0.0719**             |                      |
| residual (forecast minus PFS), median | **+2.86 percentage points** |                |
| residual MAE                        | 9.71 pp                |                      |
| residual P50                        | 6.63 pp                |                      |
| residual P90                        | 19.13 pp               |                      |

### Per-bucket breakdown

| Bucket           |  n     | LSTM median | PFS median | residual | abs residual |
| ---------------- | -----: | ----------: | ---------: | -------: | -----------: |
| Medicine / E&M   | 4,255  | +2.79%      | -0.74%     | +3.53 pp | 6.27 pp      |
| Surgery          | 3,142  | +2.28%      | -1.39%     | +3.67 pp | 10.34 pp     |
| Radiology        | 2,691  | +1.90%      | -2.43%     | +4.33 pp | 7.59 pp      |
| Lab/Pathology    | 2,191  | +7.93%      | +0.46%     | +7.47 pp | 14.13 pp     |
| HCPCS Level II   | 3,475  | -2.28%      | -0.64%     | -1.64 pp | 12.23 pp     |

### Reading the numbers

**The forecast does not encode the 2024 PFS policy.** Pearson r = 0.07 between forecast YoY change and PFS YoY change. The forecast is essentially uncorrelated with the rate change that drives most of the actual year-over-year movement for fee-schedule codes.

**The forecast is biased upward by ~3 percentage points** vs the policy baseline. This is consistent with the LSTM having learned the 2013-2023 trend, which was dominated by the 2021 E&M reweighting (E&M codes got a large RVU bump) and steady inflation-adjusted growth in many specialties. The model projected continuation of that trend; it did not know about the 2024 conversion-factor cut.

**HCPCS Level II is the only bucket where the forecast directionally matches the policy** (both negative). For E&M, Surgery, and Radiology the forecast points up while the schedule went down. Lab/Pathology has the largest disagreement: forecast +7.9% vs schedule +0.5% (most lab tests are CLFS not PFS, so the PFS-side number is a poor proxy for what the lab-bucket Avg_Mdcr_Alowd_Amt actually does, but the forecast still looks too hot).

## Test 2: 2025 / 2026 autoregressive collapse

**32.2% of (specialty, state, bucket) cells have `forecast_2025 < 0.5 * forecast_2024`.** The LSTM autoregressive rollout is collapsing toward zero for nearly a third of cells in year 2 of the projection. This confirms the bug noted in CLAUDE.md:

> Known bug: modeling/train_lstm_local.py evaluate() (~line 340) uses 1-step teacher-forced prediction, not autoregressive rollout. Reported R² approx 0.886 is inflated by approx 0.017. Cosmetic (V2 stacker is the production forecast model anyway), but should be fixed if anyone re-runs the local LSTM.

Plot pane 4 in `forecast_falsification.png` shows the bimodal distribution: most cells stay near 1.0 (flat or modest change), but a long left tail collapses below 0.5. This isn't a 0.017 R² inflation; it is a one-third-of-cells-broken issue if the LSTM is queried for 2025-2026 directly. The stacker presumably smooths some of this through `last_history_value` (16% feature importance) and `history_mean` (2.7%), but the dominant `lstm_pred` (70.5% importance) is corrupt for years 2 and 3 of the projection.

**Recommendation:** if the LSTM forecast file is used anywhere downstream (for example, surfaced to the `/forecast` route via the stacker output), restrict it to 2024 only. Years 2025 and 2026 in this file are not trustworthy.

## Caveats

1. **No actual 2024 PUF**, so we can't measure true forecast accuracy. We can only verify the forecast is consistent with policy. A +3 pp forecast bias does not necessarily mean the forecast is wrong: realized 2024 Avg_Mdcr_Alowd_Amt could differ from the PFS-implied baseline due to volume mix shifts, modifier mix shifts, or place-of-service shifts that the forecast might be picking up. But the near-zero correlation (r=0.07) between forecast direction and policy direction is harder to explain away.
2. **The PFS-implied baseline is a proxy**, not ground truth. It uses Global modifier rates (not specialty-modifier-aware), volume-weighted by 2023 silver. Lab/Pathology is the worst-served bucket because most lab tests price under CLFS, not PFS.
3. **The LSTM under test is not the production stacker.** Stacker may smooth some of these issues but will not invert them: the stacker is a weighted average of LSTM, last value, and Chronos, and the median forecast direction will be dominated by the LSTM at 70% weight.
4. **CMS adjusts forecasts in mid-year revisions**. The 2024A→2024B revision (Congressional restoration) was unpredictable and no model could have anticipated it. The fair comparison is forecast vs the post-revision 2024B rate, which I used.

## What this says about the forecast track product claim

CLAUDE.md describes the production forecast as **R^2 = 0.8852 on temporal holdout 2022-2023**. That holdout was 2022-2023 evaluated against 2022-2023 actuals. **It says nothing about whether the forecast tracks 2024+ policy changes**, because the holdout window predates the 2024 CF cut.

The fair takeaway:
- **For predicting 2024 absolute Avg_Mdcr_Alowd_Amt**, the LSTM forecast is biased upward by roughly 3 pp on average. Whether that translates into a real forecast error depends on volume/modifier shifts that are unobserved without the 2024 PUF.
- **For predicting 2024 *direction* of change**, the LSTM forecast is essentially uninformative (r=0.07 with the rate change that drives most of the actual movement).
- **For 2025 and 2026**, the LSTM forecast is partially broken (32% of cells collapse). Years 2-3 of the projection are not safe to surface to users.

**A simple "use last year's value, adjusted for the published CF change" baseline** would have outperformed the LSTM directionally for 2024 because the dominant signal is policy, not learned trend. This baseline can be implemented from the same indicators CSVs we used to validate against, with no model required.

## Files

- `falsification_test_v2/forecast_falsification.py`: end-to-end script.
- `falsification_test_v2/forecast_falsification_2024.csv`: 15,754 cells with forecast vs PFS YoY columns.
- `falsification_test_v2/pfs_baseline_state_bucket.csv`: 260 (state, bucket) cells with 2023 / 2024 / 2025 average PFS rates and YoY changes.
- `falsification_test_v2/forecast_falsification.png`: 2x2 plot grid (histograms, scatter, residual boxplots, 2025 collapse demonstration).

## Bottom line

**The forecast track encodes 2013-2023 trend, not 2024 policy.** The single largest signal in 2024 PFS payment-rate change is the conversion-factor cut, which the model could not anticipate from training data alone. As a result the LSTM forecast for 2024 is biased upward (median +2.0% vs policy -0.85%) and uncorrelated with the actual rate-change direction (Pearson 0.07). Years 2025-2026 of the local LSTM file have the documented autoregressive collapse bug in roughly a third of cells. **Recommendation: clamp any forecast surface to 2024 only and frame the prediction as "trend-projection ignoring policy revisions," not as a payment forecast.** A policy-aware baseline (last value times annual CF change) would beat the model directionally for 2024 and is trivially easy to compute from CMS source data we already have.
