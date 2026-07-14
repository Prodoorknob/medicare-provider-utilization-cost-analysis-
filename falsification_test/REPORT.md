# Falsification Test: Stage 1 model vs official PFS vs empirical

**Date:** 2026-04-27
**Model:** `lgbm_v2_no_charge.txt` (1000 trees, no-charge variant — production)
**Sample:** N=50 silver rows, stratified across 11 years (2013–2023) and 15 states, drawn from `local_pipeline/silver/*.parquet` after filtering to PFS-active codes (status A/R/T) where the lookup is even defined.
**Comparator:** state-area-weighted official PFS rate built from CMS `PPRRVU{YY}` + `GPCI{YY}` files for each row's (HCPCS, state, year), using non-facility PE RVU when `Place_Of_Srvc=O` and facility PE RVU when `=F`. State-area-weighted = simple unweighted mean across the localities in that state (no county-area data was used).

## Headline numbers (n=50)

|                        | MAE      | bias       | P50\|err\| | P90\|err\| | Pearson r |
| ---------------------- | -------- | ---------- | ---------- | ---------- | --------- |
| empirical vs official  | $32.69   | −$32.24    | $5.48      | $94.65     | 0.6318    |
| **model vs official**  | **$33.77** | −$32.13  | $5.75      | $86.47     | 0.6469    |
| **model vs empirical** | **$10.39** | +$0.11   | $5.29      | $25.21     | **0.9578** |

**Aggregate verdict:** model ≈ empirical (r=0.96, MAE $10.39, near-zero bias) ≠ official (r=0.65 with both, both ≈ $32 below official). The model has internalized the gap between the unmodified PFS rate and what providers actually realize. **Per the user's decision rule, this is the "legitimate work" branch — not "lookup wins."**

But the aggregate result hides a clean two-regime split that's the more useful answer.

## Per-bucket breakdown

| Bucket           | n  | MAE(model, emp) | MAE(model, off) | MAE(emp, off) | mean(emp / off) |
| ---------------- | -- | --------------: | --------------: | ------------: | --------------: |
| Medicine / E&M   | 29 | $7.35           | $7.33           | $6.55         | 0.94            |
| Radiology        | 14 | **$16.07**      | **$91.95**      | **$87.73**    | **0.55**        |
| Surgery          | 4  | $12.75          | $41.95          | $51.35        | 0.73            |
| HCPCS Level II   | 3  | $10.06          | $6.89           | $3.62         | 0.97            |

E&M and Level II billings settle at ~94–97% of the unmodified fee schedule. **Radiology settles at ~55%.** The aggregate edge the model has over the lookup is almost entirely radiology + a smaller surgery contribution.

## Root cause: the radiology gap is modifier-26 (professional component)

The "official PFS" I computed is the **global** rate. Many imaging codes have `PCTC IND = 1`, meaning the rate splits into a professional component (modifier 26 — the radiologist's read) and a technical component (modifier TC — the equipment/supplies, billed by the facility). For 2023 row-level CMS PPRRVU verification:

| HCPCS | Description           | Global rate | MOD-26 (PC) | MOD-TC (technical) | Empirical (sample) | Model pred |
| ----- | --------------------- | ----------- | ----------- | ------------------ | ------------------ | ---------- |
| 70549 | MRI neck w&wo         | $365.64     | **$86.41**  | $279.23            | **$80.96**         | **$81.36** |
| 74174 | CT abdomen/pelvis w&wo | $401.90    | **$104.71** | $297.19            | **$103.19**        | **$105.86** |
| 73060 | X-ray humerus         | $32.87      | **$8.13**   | $24.74             | **$8.22**          | **$8.21**  |
| 99213 | Office visit (E&M)    | $90.82 (NF) / $66.08 (FAC) | n/a (no PCTC split) | n/a | $90.56 (mean)    | $75.15     |

The radiologist NPIs in our sample bill modifier-26 only — they don't own the scanner. The empirical and the model both track the modifier-26 rate. The "official PFS" comparator I used is the global, which essentially nobody collects.

## Stricter-cut decisive numbers

**Subset where empirical ≥ 90% of unmodified official (n=31, dominated by E&M/Lab/Level-II):**
- MAE(model, empirical) = **$10.09**
- MAE(model, official) = **$9.38**
- **Edge of model over lookup: −$0.70.** When the unmodified PFS already captures the realized rate, the lookup is at least as good as the model and arguably better (parsimony, no train/serve infra).

**Subset where empirical < 60% of unmodified official (n=10, all radiology + 1 surgery + 1 derm):**
- MAE(model, empirical) = **$11.89**
- MAE(model, official) = **$127.13**
- **Edge of model over lookup: +$115.** This is where the model earns its keep.

## What this implies

1. **The model is doing legitimate work.** It has learned to predict realized payment, not the unmodified fee schedule. The largest single signal it captures is the modifier-26 / multi-procedure-discount mix that an unmodified-PFS lookup completely misses.

2. **The bar isn't quite the live tool — it's a "modifier-aware lookup."** A smart consumer of PFS data could look up the HCPCS, see `PCTC IND = 1`, ask "is this a radiologist?" and pull the MOD-26 rate instead of the global. That would close most of the radiology gap. The model is doing this kind of inference implicitly: it sees `Rndrng_Prvdr_Type = Diagnostic Radiology` and an imaging-bucket code and predicts the PC-only rate.

3. **For E&M, Lab, and HCPCS Level II — about 65% of our sample by row count and most of CMS volume — the model is at parity with a naïve lookup.** The MAE-on-empirical of $10 reported in the V2 spec is real, but ~$7 of it is ordinary E&M variance the lookup also has, and ~$3 is whatever NPI-conditional adjustment the model adds. The marginal value of the model over the lookup on these codes is tiny.

4. **Where the model has a clear NPI-conditional edge, that edge is concentrated in imaging-heavy rows.** This suggests two product framings:
   - For an estimator that an *unmodified* PFS lookup could already serve (E&M, basic procedures), promote the lookup; the model is overkill.
   - For radiology / surgery / multi-procedure cases, surface the model and call out *why*: "predicts realized payment after typical modifier mix for this specialty."

5. **Caveats on the test design:**
   - State-area-weighted (simple unweighted mean over localities) underweights the dense urban localities where most NPIs actually practice. A beneficiary-weighted lookup would be a tighter "official" comparator. Probably worsens the lookup further on multi-locality states because NPI/volume concentrates in higher-GPCI metros.
   - n=50 is small for tail behavior; the radiology subsample is n=14. Findings on the broad split are robust; surgery (n=4) is suggestive only.
   - I did not scrape the live PFS Look-Up Tool (it's a JS SPA without a public JSON endpoint). My official PFS is computed from the same PPRRVU + GPCI files the tool consumes, with the same `(Work·PW_GPCI + PE·PE_GPCI + MP·MP_GPCI)·CF` formula CMS publishes. Spot check 99213 AL 2023 NF: my $84.67 matches by construction; for any state with all GPCIs ≈ 1, my number reduces exactly to `(Work + PE + MP)·CF` which is what the tool returns.

## Files produced

- `falsification_test/results.csv` — 50 evaluable rows (year, state, hcpcs, specialty, pos, empirical, official_pfs, model_pred, error columns).
- `falsification_test/all_candidates.csv` — full 80-row sample including 30 not in the PFS active list (status I/N/X/B/C — out of scope for this test).
- `falsification_test/scatter.png` — model-vs-empirical and model-vs-official scatter plots, colored by HCPCS bucket. Shows tight fit on the empirical axis and the radiology cluster falling below the y=x line on the official axis.
- `falsification_test/run_falsification.py` — reproducible end-to-end script.
- `falsification_test/analyze_results.py` — per-bucket breakdown + outlier inspection.

## Bottom line

The model is not just learning the fee schedule. It correctly predicts realized payment for codes with structural modifier-driven discounts (radiology PC/TC split most prominently). On codes without those discounts — which is most CMS volume — the unmodified PFS lookup is approximately as accurate as the model. **The defensible product story is: "predicts payment after typical modifier mix per specialty," not "more accurate than the fee schedule across the board."**
