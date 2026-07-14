# AllowanceMap — Technical Summary 2: Provider Anomaly Investigation Agent

**Role:** Sole developer — detection pipeline, rules engine, LLM agent, ground-truth validation, cloud automation.
**Client/Context:** Fraud-lead generation over the full CMS Medicare provider universe: 11.52M provider-year profiles (1.76M NPIs × 11 years), validated against ~13,105 government-sanctioned providers (OIG LEIE + CMS Revoked Providers).
**Status:** Live — briefs published on the web app; validation loop runs monthly as a Railway cron service (deployed June 2026).

## What it is
An agentic pipeline that surfaces Medicare providers whose billing patterns warrant investigation, then writes the investigation brief for them. It ranks anomalies statistically, checks each candidate against a library of codified fraud-indicator rules, and calls Claude (Sonnet 4.6) to produce an evidence-cited analyst brief. Critically, it is validated as a **lead-prioritization system** — measured by lift and lead-time against real government sanctions — not marketed as a fraud classifier.

## Architecture
**Silver data → NPI-year profiles (22 metrics) → peer benchmarks → 3-method outlier detection → rules engine → LLM brief → analyst queue → monthly validation against government labels.**

- **Detection:** log1p z-scores against specialty/state/national benchmarks + Isolation Forest + temporal spike rules; requiring ≥2 independent methods keeps the composite flag rate at 0.77% of provider-years.
- **Rules engine:** 11 codified fraud indicators (8 evaluable on public data; 3 documented as structurally NOT EVALUABLE because they need per-claim linkage). Includes a CRITICAL statutory override for LEIE-excluded providers still billing, an empirical specialty-scope rule built from what each specialty's population actually bills, and an E&M upcoding rule against 5.94M pre-computed coding distributions.
- **LLM layer:** briefs generated with a 2,584-token cached system prompt and structured evidence packages; 429/529 retry with backoff. Cost ~$0.04/brief — a 100-brief validation batch cost $3.82.
- **Ground truth + validation:** a labels pipeline unifies OIG LEIE and the CMS Revoked Providers dataset (13,105 sanctioned NPIs, 10,624 fraud-relevant after statute filtering) and backtests the ranking point-in-time.
- **Autonomy:** `run_pipeline.py` orchestrates refresh → labels → lead queue → backtest → self-eval drift tracking, containerized and deployed as a Railway cron (monthly, full run in ~40s, ~cents/month).

## Hard problems solved
- **Caught and corrected look-ahead leakage in my own validation.** The first backtest (8.5× lift@50) aggregated each provider's full 2013–2023 panel — including post-sanction years — and its lead-time headline rested on a single true positive. I rebuilt it as a point-in-time, as-of-year ranking with a dev(≤2017)/test(≥2018) holdout, volume stratification, and Benjamini-Hochberg FDR control.
- **Diagnosed a volume confound in the corrected results:** held-out lift@1000 of 3.9–6.2× is statistically real (BH p ≤ 0.0013), but sanction base rates rise 6.5× from lowest to highest volume quintile while within-volume-band lift is only ~1–1.6×. The system's honest description — "surfaces high-volume providers, who get sanctioned more" — is documented alongside the headline.
- **Reframed the metric to fit the label.** At a 0.24% positive base rate with sparse, lagged, censored sanctions, accuracy and ROC-AUC are meaningless; the system is evaluated on lift@k and lead-time (median 42 months from first flag to sanction — the actual value proposition).
- **Let validation redesign the ranking:** severity-mass ranking scored 0× lift (it surfaces chronic high-volume ER/internal-medicine billing); breadth of distinct anomaly signals per provider validated best and became the production rank score. Fee-schedule-derived metrics proved *negatively* predictive (0.63–0.68× lift) and were excluded.
- **Shipped privacy controls for real-person data:** briefs reference real NPIs, so the public route defaults to masked NPIs (build-time masking + runtime redaction toggle with cross-tab sync) and the pages are noindexed.
- **Fixed a silent coverage bug:** LEIE metadata reported 100% NPI coverage; the real figure is 10.4% — materially changing how much of the exclusion list the labels can use.

## Verified outcomes
- Held-out, leakage-corrected lift@1000: 3.9–6.2× over base rate (8–15 true positives, all BH p ≤ 0.0013); median lead-time 42 months.
- 100-brief validation run: 39 CRITICAL / 59 HIGH / 2 MEDIUM at $3.82 total LLM spend.
- Monthly pipeline: end-to-end on Railway in ~40 seconds, compute cost ~cents/month.
- Scale: 11.52M profiles, 86,924-row specialty-scope table (130 specialties), 5.94M E&M distributions.

## Tech stack
Python · pandas · PyArrow · scikit-learn (Isolation Forest) · Claude API (Sonnet 4.6, prompt caching) · statistical validation (lift@k, BH-FDR, point-in-time backtesting) · OIG LEIE / CMS data APIs · Docker · Railway cron · FastAPI · Next.js · Vercel
