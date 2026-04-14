# FRONTEND_TODOS.md — Next Session Deployment Plan

> Purpose: concrete checklist for deploying the Phase 8 forecast track results to the live web app + Railway API + Supabase. Written 2026-04-14 at end of modeling session. Pick up here next session.
>
> Related docs: `MODELING.md` (authoritative results), `PROGRESS.md` Phase 8 (session log), `memory/project_railway_api.md` (backend), `memory/project_v2_spec.md` (web app context)

---

## TL;DR — What's changing

| Component | Current | Target |
|---|---|---|
| Forecast model | LSTM V1 (R² 0.886 reported, 0.8689 honest) | **LGB Stacker V2_12 (R² 0.8852)** |
| Forecast serving | Supabase `lstm_forecasts` table proxied via `/forecast` endpoint | Same endpoint, new data (stacker) |
| Quantile bounds | MC Dropout P10/P50/P90 (LSTM) | Point-only (stacker) → **needs decision** |
| Web app forecast page | Shows LSTM-generated bands | Must show stacker predictions with new uncertainty story |
| Model comparison page | Up through V2_08 (TFT lost to LSTM) | Add V2_09–V2_13 + show stacker wins |

**Production model artifact:** `{DRIVE}/v2_artifacts/predictions/stacker_forecast_2024_2026.parquet` (61,716 rows, 20,572 groups × 3 years, LSTM-compatible schema)

---

## Priority 0 — Must do first (30 min)

### [ ] 0.1 Download stacker forecast parquet from Drive to local
- Source: `Google Drive/MyDrive/AllowanceMap/V2/v2_artifacts/predictions/stacker_forecast_2024_2026.parquet`
- Destination: `local_pipeline/v2_artifacts/stacker_forecast_2024_2026.parquet` (gitignored)
- Verify schema: should have `Rndrng_Prvdr_Type_idx`, `hcpcs_bucket`, `Rndrng_Prvdr_State_Abrvtn_idx`, `forecast_year`, `forecast_mean`, `forecast_std`, `forecast_p10/p50/p90`, `last_known_year`, `last_known_value`, `n_history_years`
- Verify row count: 61,716
- Spot-check 2024 vs 2025 for a known specialty — stacker has a mean-reversion drop 2024 → 2025 (~$10 for aggregate mean) which is expected behavior, not a bug

### [ ] 0.2 Smoke-test stacker forecast values
- Compare 5-10 high-volume (specialty × bucket × state) triples against the existing LSTM forecast in Supabase
- Predictions should be close (both models agree most of the time) but not identical
- Use this notebook or quick Python script:
  ```python
  import pandas as pd
  stacker = pd.read_parquet('local_pipeline/v2_artifacts/stacker_forecast_2024_2026.parquet')
  # Pick a known specialty — e.g. Cardiology (specialty_bucket=...) in CA
  s = stacker[(stacker.Rndrng_Prvdr_Type_idx == 10) & (stacker.Rndrng_Prvdr_State_Abrvtn_idx == 5)]
  print(s.groupby(['hcpcs_bucket', 'forecast_year'])['forecast_mean'].mean())
  ```
- If any `forecast_mean < 0` or absurdly large (>$10K), something went wrong during the stacker refit — re-run V2_12 Cell 10

---

## Priority 1 — Decide how to handle the quantile bounds (1 hour)

The stacker is **point-only** — `forecast_p10 == forecast_p50 == forecast_p90 == forecast_mean`. This breaks the forecast explorer UI which currently draws a confidence band. Three options:

### [ ] 1.A (Recommended, fastest) Synthesize bounds from historical error distribution
- Compute the stacker's OOF prediction error on the 2022-2023 holdout
- Fit a per-percentile error scaling: `p10_lower = pred × (1 + error_quantile_10)`, etc.
- Apply to the 2024-2026 forecasts
- Pros: 1 hour of work, uses real OOF error distribution
- Cons: assumes error distribution is stationary (it's not quite — 2024-2026 predictions are out-of-sample and may have wider tails)

### [ ] 1.B Train a quantile stacker (+2 hours)
- Modify V2_12 Cell 10 to train three LightGBM boosters at `objective='quantile', alpha={0.1, 0.5, 0.9}`
- Rerun the notebook (LSTM retrain is cached; only Cell 10 onward needs to run)
- Re-save forecast parquet with actual p10/p50/p90 values
- Pros: real model-based uncertainty, no stationarity assumption
- Cons: more compute, need to verify quantile coverage on OOF set

### [ ] 1.C Use TFT V2_13 forecasts instead
- V2_13 has native P10/P50/P90 from QuantileLoss + more coherent multi-horizon trajectory
- R² 0.8691 is lower than stacker's 0.8852 but has better forecast shape
- Pros: free uncertainty bounds, smoother 2024-2026 trajectory
- Cons: worse validation R², which is the one empirical signal you actually have

**Recommendation:** 1.A for this sprint (ship it), queue 1.B as a follow-up. Don't do 1.C — the validation R² gap is a real signal you shouldn't override on a speculative shape argument.

---

## Priority 2 — Update Supabase `lstm_forecasts` table (1 hour)

The `/forecast` endpoint queries `client.table("lstm_forecasts")`. Options:

### [ ] 2.A (Recommended) Replace `lstm_forecasts` rows in place
- Keep the table name (no API changes needed)
- Truncate the existing table or UPSERT new rows
- Column mapping from stacker parquet to Supabase schema:
  ```
  Rndrng_Prvdr_Type_idx          → specialty_idx
  Rndrng_Prvdr_State_Abrvtn_idx  → state_idx
  hcpcs_bucket                    → hcpcs_bucket
  forecast_year                   → forecast_year
  forecast_mean                   → forecast_mean
  forecast_std                    → forecast_std
  forecast_p10/p50/p90            → forecast_p10/p50/p90
  last_known_year                 → last_known_year
  last_known_value                → last_known_value
  n_history_years                 → n_history_years
  ```
- Check existing `api/services/supabase.py:56-72` to confirm column names match
- Use Supabase service role key (NOT anon) for the bulk UPSERT

### [ ] 2.B Create separate `stacker_forecasts` table
- Add new table with same schema as `lstm_forecasts`
- Add new `fetch_stacker_forecasts()` function in `api/services/supabase.py`
- Add new `/forecast/stacker` route alongside existing `/forecast`
- Pros: keeps LSTM forecasts for comparison / A/B
- Cons: more API surface to maintain, frontend needs to choose which to call

**Recommendation:** 2.A for cleaner deployment. The LSTM forecast is already documented as a measurement-bugged baseline in MODELING.md — no reason to keep serving it.

### [ ] 2.3 Upload script

Write `scripts/upload_stacker_forecast_to_supabase.py` that:
1. Reads the parquet
2. Renames columns to Supabase schema
3. Batches UPSERTs (Supabase limits ~1000 rows per request, so 62 batches)
4. Validates row count after upload
5. Runs a sanity query to confirm data is retrievable via the same `/forecast` endpoint

---

## Priority 3 — Railway API verification (30 min)

### [ ] 3.1 Smoke-test `/forecast` endpoint after Supabase update
- Target: `medicare-provider-utilization-cost-analysis-production.up.railway.app/forecast?specialty_idx=10&state_idx=5&hcpcs_bucket=2`
- Should return stacker-generated values now
- Verify the response schema matches what the frontend expects (no `specialty_idx` → `Rndrng_Prvdr_Type_idx` mismatch)
- Check `api/routers/forecast.py:10-21` — the endpoint itself doesn't need code changes if the Supabase table schema stays the same

### [ ] 3.2 No Railway deploy needed if Priority 2.A is chosen
- Priority 2.A swaps data only, not code → Railway container doesn't need redeploy
- If Priority 2.B (new table/route) is chosen → `git push` to deploy branch, Railway auto-deploys

---

## Priority 4 — Frontend updates (2 hours)

Next.js app at `https://medicare-provider-utilization-cost-analysis.vercel.app/` (or your Vercel URL).

### [ ] 4.1 Forecast Explorer page — update the story
- **Copy change:** current page says "LSTM 2024-2026 forecasts with MC Dropout confidence bounds." Update to: "LightGBM Stacker forecasts blending LSTM and Chronos-Bolt base models. Uncertainty from historical error distribution." (or TFT if you went with 1.C)
- **Chart change:** confidence band rendering works as-is if quantile bounds exist in the data; no React code change needed
- **Model name badge:** update from "LSTM V1" to "LGB Stacker V2_12"
- **Last updated date:** refresh to 2026-04-14

### [ ] 4.2 Model Comparison page — add Phase 8 results
- Add a new section "Forecast Track (2024-2026 Projections)"
- Include the leaderboard from MODELING.md Phase 8:

  | Model | MAE | RMSE | R² |
  |---|---|---|---|
  | LSTM V1 (fair, AR) | $9.82 | $18.91 | 0.8689 |
  | Chronos-Bolt cpi_cf_deflated | $9.39 | $19.71 | 0.8576 |
  | Multivariate TFT V2_13 | $9.23 | $18.79 | 0.8691 |
  | **LGB Stacker V2_12** 🏆 | **$8.74** | **$17.69** | **0.8852** |

- Add explanatory copy:
  - Why the stacker won (ensemble diversity + history conditioning)
  - Why multivariate TFT didn't help (signal ceiling at annual resolution)
  - Teacher-forcing bug in reported LSTM baseline (transparency)

### [ ] 4.3 About page — update model pipeline description
- Add forecast stacker to the "How it works" section
- Link to `MODELING.md` on GitHub for the full story
- Mention the signal ceiling finding as a research takeaway

### [ ] 4.4 Verify `NEXT_PUBLIC_API_URL` in Vercel
- Per `memory/project_railway_api.md` — this was flagged as "needs to be set"
- Check Vercel dashboard → Project → Settings → Environment Variables
- Should point to `https://medicare-provider-utilization-cost-analysis-production.up.railway.app`
- If missing, set it and redeploy

---

## Priority 5 — Supabase `model_metrics` and `feature_importances` tables (45 min)

Phase 7 Remaining Work section flags:
> - [ ] Update Supabase model_metrics and feature_importances tables

### [ ] 5.1 Update `model_metrics`
- Add rows for V2_09, V2_10, V2_11, V2_12, V2_13
- Columns: `model_name`, `stage`, `mae`, `rmse`, `r2`, `source_notebook`, `status` (production/archived/failed)
- Mark `lgb_stacker_v2_12_colab` as `status='production'` for forecast stage
- Mark `lstm_local` as `status='archived'` with note "teacher-forcing bug in evaluate(), see MODELING.md"

### [ ] 5.2 Update `feature_importances` table with stacker importances
- Insert 13 rows from the V2_12 feature importance table (see MODELING.md)
- `model_name='lgb_stacker_v2_12_colab'`, `stage='forecast'`

---

## Priority 6 — Git hygiene & documentation (30 min)

### [ ] 6.1 Commit V2_12 and V2_13 notebooks + builder scripts
- Already created in this session:
  - `colab/V2_12_stacker_forecast.ipynb`
  - `colab/V2_13_multivariate_tft.ipynb`
  - `scripts/build_v2_12_notebook.py`
  - `scripts/build_v2_13_notebook.py`
- Plus V2_10/V2_11 modifications (already tracked as modified)

### [ ] 6.2 Commit documentation updates
- `PROGRESS.md` (Phase 8 + changelog entry)
- `MODELING.md` (new)
- `FRONTEND_TODOS.md` (this file)

### [ ] 6.3 Add stacker forecast parquet to `.gitignore` (if not already)
- `local_pipeline/v2_artifacts/` should be gitignored
- Verify with `git check-ignore local_pipeline/v2_artifacts/stacker_forecast_2024_2026.parquet`

---

## Priority 7 — Code debt (from MODELING.md backlog)

These are not blockers but worth scheduling.

### [ ] 7.1 Fix `train_lstm_local.py evaluate()` teacher-forcing bug
- Current bug: line 340, `evaluate()` does 1-step-ahead teacher-forced prediction
- Fix: add `--eval-mode {teacher_forced, autoregressive}` CLI flag; default to `autoregressive`
- Use V2_12 Cell 6's batched autoregressive rollout logic as the reference implementation
- Report both metrics in MLflow for transparency
- **Why:** the reported R² 0.886 is silently wrong and has already misled the V2_09–V2_11 analysis once
- Effort: 1 hour

### [ ] 7.2 Add `eval_level` parameter to Stage 1 training scripts
- Stage 1 R² 0.9575 and Forecast R² 0.8852 are NOT comparable (individual rows vs group-year means)
- Add a docstring and MLflow param making this explicit
- Prevents future "why is the forecast worse than the allowed-amount model" confusion

---

## Session wrap-up checklist

When all of Priority 0-4 are done:

- [ ] Stacker forecasts live at `/forecast` endpoint
- [ ] Frontend renders stacker predictions with uncertainty bands
- [ ] Model comparison page updated with Phase 8 results
- [ ] `NEXT_PUBLIC_API_URL` confirmed in Vercel
- [ ] End-to-end smoke test: open Vercel frontend → pick a specialty → see forecast → verify numbers match what's in `stacker_forecast_2024_2026.parquet`
- [ ] Write a short LinkedIn/portfolio post: "Shipped a LightGBM forecast stacker that blends LSTM + Chronos-Bolt, R² 0.8852, after confirming multivariate + foundation models hit a 0.885 ceiling at annual resolution. Next lever: quarterly data."

## Stretch: close the project

If time permits and Priorities 0-6 are done:

- [ ] Update `README.md` with the final model lineup and link to MODELING.md
- [ ] Update `IMPROVEMENTS.md` with Phase 8 findings
- [ ] Archive `PROVIDER_ANOMALY_AGENT_SPEC.md` if not pursuing it
- [ ] Tag a git release `v2.13` marking the end of the forecasting track

---

**Estimated total effort for Priority 0-4 (core deployment):** ~4-5 hours
**Plus Priorities 5-7:** ~2-3 hours
**Total to fully close the project:** ~1 working day
