# AllowanceMap — Technical Summary 3: The Pivot from Cost Estimator to Fraud-Investigation Platform

**Role:** Sole developer — product direction, architecture refactoring, evaluation redesign.
**Client/Context:** One codebase, two products: a consumer-facing Medicare cost estimator (Stages 1–2 + forecast) refactored into a provider fraud-investigation platform, reusing the same data layer, models, and serving stack.
**Status:** Both surfaces live; the investigation agent is the actively maintained product (July 2026).

## What it is
The project began as a cost-estimation tool: predict Medicare allowed amounts and patient out-of-pocket costs, forecast specialty trends. Two findings forced a re-evaluation of where the value actually was. First, an adversarial falsification study showed the flagship Stage 1 model (R² 0.943) was largely re-deriving the published CMS fee schedule — accurate, but administered prices are look-up-able, so prediction adds limited value. Second, the same 11-year, 126.8M-row provider panel that made price prediction easy made *behavioral deviation* detection genuinely novel. The refactor repositioned the asset base — data layer, benchmarks, even the Stage 1 model itself — from "predict the price" to "explain why this provider's billing doesn't look like their peers'," with an agentic workflow producing analyst-ready investigation briefs.

## Architecture of the refactor
**Same medallion silver layer → new consumer (NPI-year profiles) → detection + rules + LLM agent → validation against government sanctions.**

- **Data layer reused, not rebuilt:** the anomaly pipeline reads the existing silver parquet directly; provider profiles, peer benchmarks, specialty scopes, and E&M distributions are new gold-layer artifacts derived from cleaned data that already existed for modeling.
- **The Stage 1 model was repurposed as a fraud signal:** a REVENUE_DEVIATION rule scores every provider's actual-vs-model-expected allowed dollars using the production LightGBM artifact. Because falsification showed the model approximates the fee schedule, its residuals approximate deviation *from the fee schedule* — turning the falsification finding into a feature. (Implemented and unit-scored; not yet exercised in a published brief batch.)
- **The serving stack was extended, not forked:** the same FastAPI service and Next.js app gained `/investigations` routes; specialty canonicalization built for the estimator is shared by the agent.
- **Evaluation was rebuilt for the new problem:** regression metrics (R², MAE) were replaced by ranking metrics against sparse real-world labels — lift@k over sanctioned-provider base rates, lead-time from flag to sanction, FDR-controlled significance.

## Hard problems solved
- **Killed my own headline honestly.** The falsification study (modifier-aware fee-schedule comparator, rebuilt from raw CMS PFS files) collapsed the model's apparent radiology edge from ~$92 to ~$26 MAE and showed near-parity with a lookup on ~65% of volume. Instead of burying it, the finding became the pivot rationale and is documented in-repo.
- **Empirically confirmed the pivot with independent data:** in sanction backtests, fee-schedule-derived metrics were *negatively* predictive of fraud (0.63–0.68× lift) while behavioral intensity metrics over-indexed (2.0–2.2×) — the ground truth agreed that price signals were the wrong axis and behavior was the right one.
- **Redefined success criteria mid-project without inflating them:** the agent is explicitly a prioritization + evidence layer (lift 3.9–6.2×@1000, median 42-month lead-time, known volume confound), not a "fraud detector" — a framing chosen because the sanction label is sparse, lagged, and censored.
- **Managed the transition as reviewable increments:** the pivot shipped as sequenced PRs (anomaly agent phases A–E, validation harness, leakage correction, privacy gating), each independently mergeable, rather than a big-bang rewrite; the estimator remained live throughout.

## Verified outcomes
- Reuse ratio: the agent added ~26 new Python modules while reusing the entire ingestion/silver layer, the production Stage 1 artifact, and the API/frontend deployment stack unchanged.
- Falsification: model-vs-lookup gap on the audit sample = $10.39 vs $17.03 MAE — the quantified basis for the pivot.
- Post-pivot product: 11.52M provider-year profiles, validated lead queue, monthly autonomous validation on Railway (~40s/run).

## Tech stack
Same stack, redirected: Python · pandas/PyArrow medallion pipeline · LightGBM (repurposed as residual detector) · Isolation Forest · Claude API agent · lift/lead-time validation · FastAPI · Next.js · Railway · Vercel
