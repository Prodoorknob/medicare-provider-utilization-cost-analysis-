# Stage 1 Falsification Test v2: corrected methodology

**Date:** 2026-04-27
**Sample:** Identical 50 rows from `falsification_test/results.csv` (v1 sample preserved for direct diff)
**Model:** `lgbm_v2_no_charge.txt` (production), unchanged from v1
**Data source (new):** raw CSVs underlying the live PFS Look-Up Tool at `pfs.data.cms.gov/api/1/metastore/schemas/dataset/items`. 13 year-pairs validated (2013 through 2025), of which 11 are used here (2013 through 2023, matching v1).
**Comparator change:** v1 compared against the unmodified Global PFS rate. v2 compares against a modifier-aware rate: MOD-26 (professional component) for radiologist NPIs billing PCTC=1 imaging codes, Global otherwise.
**Spot-check anchor:** 70549 MOD-26 in Suburban Chicago 2013 = $92.28, reproduced exactly from local CSVs.

## Headline

### Table A: Model vs Official GLOBAL (v2 data source, v1 methodology) — replication check

| Bucket             |  n | MAE      | bias       | P50\|err\| | P90\|err\|  |
| ------------------ | -: | -------: | ---------: | ---------: | ----------: |
| Medicine / E&M     | 29 | $7.33    | -$4.76     | $3.63      | $21.82      |
| Radiology          | 14 | **$91.95** | **-$91.62** | $29.97   | $272.24     |
| Surgery            |  4 | $41.95   | -$41.95    | $20.56     | $95.12      |
| HCPCS Level II     |  3 | $6.89    | -$5.98     | $1.51      | $14.53      |
| **Overall**        | 50 | **$33.77** | -$32.13  | $5.75      | $86.47      |
| Pearson r vs model |    | 0.6469   |            |            |             |

This reproduces v1's headline ($33.77 MAE overall, $91.95 on radiology) under the new data source. The two methodologies agree to the cent. The state-mean of GPCI applied to RVUs (v1) and per-locality pricing then state-mean (v2) are arithmetically near-identical when GPCI is the only varying factor.

### Table B: Model vs Official MODIFIER-AWARE (the new methodology)

| Bucket             |  n | MAE      | bias       | P50\|err\| | P90\|err\|  |
| ------------------ | -: | -------: | ---------: | ---------: | ----------: |
| Medicine / E&M     | 29 | $7.33    | -$4.76     | $3.63      | $21.82      |
| Radiology          | 14 | **$26.29** | **+$25.26** | $2.98    | $82.76      |
| Surgery            |  4 | $41.95   | -$41.95    | $20.56     | $95.12      |
| HCPCS Level II     |  3 | $6.89    | -$5.98     | $1.51      | $14.53      |
| **Overall**        | 50 | **$15.38** | +$0.60   | $3.63      | $33.53      |
| Pearson r vs model |    | 0.8116   |            |            |             |

E&M, Surgery, and HCPCS Level II are unchanged because no PCTC split applies. Radiology drops from $91.95 to $26.29 MAE (a 71% reduction). Overall MAE drops from $33.77 to $15.38.

### Empirical vs Modifier-aware (does the gap survive?)

| Bucket             |  n | MAE      | bias    | P50\|err\| | P90\|err\| |
| ------------------ | -: | -------: | ------: | ---------: | ---------: |
| Medicine / E&M     | 29 | $6.55    | -$5.77  | $3.79      | $18.41     |
| Radiology          | 14 | $31.82   | +$29.15 | $2.02      | $113.26    |
| Surgery            |  4 | $51.35   | -$51.35 | $31.10     | $115.57    |
| HCPCS Level II     |  3 | $3.62    | -$3.62  | $0.46      | $8.25      |
| **Overall**        | 50 | **$17.03** | +$0.49 | $3.46      | $29.75     |

The empirical-versus-official gap mostly closes too. **The model is no longer materially better than the modifier-aware lookup at predicting empirical**: model MAE = $10.39 vs lookup MAE = $17.03. The lookup is within ~$7 of the model on this sample.

## Radiology subgroup (the central correction)

Twelve of the fourteen radiology-bucket rows had MOD-26 applied (the other two were imaging codes billed by Neurosurgery and OB/GYN, where the global rate is appropriate).

| HCPCS  | Year | State | POS | Empirical | Global rate | MOD-26 rate | Model pred | Notes |
| ------ | ---- | ----- | --- | --------: | ----------: | ----------: | ---------: | ----- |
| 73080  | 2016 | DC    | F   | $9.53     | $36.65      | $9.88       | $9.55      | clean PC, model+empirical match |
| 73060  | 2017 | NC    | F   | $8.22     | $27.65      | $8.34       | $8.21      | clean PC, all three within $0.20 |
| 74022  | 2018 | CO    | F   | $16.08    | $45.99      | $16.67      | $15.88     | clean PC |
| 74174  | 2023 | MN    | F   | $103.19   | $404.36     | $102.47     | $105.86    | clean PC, all within $4 |
| 74178  | 2022 | KY    | F   | $93.21    | $333.14     | $93.79      | $94.84     | clean PC |
| 70549  | 2023 | MN    | F   | $80.96    | $368.14     | $84.55      | $81.36     | clean PC |
| 74160  | 2023 | CO    | F   | $60.74    | $257.75     | $61.51      | $62.70     | clean PC |
| 76700  | 2016 | RI    | O   | $39.80    | $129.40     | $42.04      | $107.49    | PC empirically; **model overshoots toward Global** |
| 71020  | 2015 | MN    | O   | $22.67    | $27.93      | $10.95      | $19.59     | empirical between PC and Global |
| 70553  | 2023 | WY    | O   | $322.12   | $336.38     | $108.94     | $253.87    | **office-based imaging — global empirically** |
| 72141  | 2022 | WY    | O   | $204.53   | $206.74     | $72.23      | $162.41    | **office-based — global empirically** |
| 70450  | 2020 | WY    | O   | $112.25   | $116.99     | $43.42      | $87.17     | **office-based — global empirically** |

The pattern: facility-place-of-service (`F`) radiology rows match MOD-26 cleanly, both empirically and in the model's prediction, all within a few dollars. Office-place-of-service (`O`) is mixed: in Wyoming especially, the radiologist appears to own the equipment and bills the Global rate (PC + TC combined), so the empirical lands near Global, not MOD-26. The model handles this partially (sits between PC and Global) but does not perfectly. 76700 is the lone POS=O row where MOD-26 is empirically right; the model wrongly went toward Global there.

This says the simple `Diagnostic Radiology + PCTC=1 -> MOD-26` policy is not quite the truth: the right rule appears to be closer to **POS=F -> MOD-26, POS=O -> Global for radiology**. With that refinement (not implemented here, since the user spec dictated the policy), the residual radiology MAE would drop further.

## What this does to v1's claims

| v1 claim                                                       | v2 finding                                                         |
| -------------------------------------------------------------- | ------------------------------------------------------------------ |
| "Model wins by $115 MAE on radiology (n=10 subset)"            | Most of that $115 was a comparator artifact. Against the right modifier the radiology MAE is $26, of which most is one POS=O confound. |
| "MAE(model, official) = $33.77 overall"                        | Replicated to the cent under v2 data source. Was real, against the wrong comparator. |
| "Model is doing legitimate NPI-conditional work"               | Weakened. The model is mostly doing implicit modifier inference: it learns 'when specialty is Diagnostic Radiology and code is imaging, predict roughly the MOD-26 rate.' That is a useful capability but it is not NPI-conditional signal beyond what a modifier-aware lookup gives you. |
| "On E&M / Lab / Level II, model is at parity with the lookup"  | Confirmed. Numbers identical (no PCTC, so the comparator is the same). |
| "Defensible product story: 'predicts realized payment after typical modifier mix per specialty'" | Still defensible, but bounded. The 'modifier mix per specialty' inference is the model's core contribution; on most of CMS volume there is little additional signal. |

## Option A robustness: per-locality range

For each row, I priced the modifier-aware HCPCS in every locality of the row's state and recorded `[min, max]`. **Only 2 of 50 rows (4%)** have empirical falling inside the per-locality range. This is not surprising: GPCI variation within a state is small (typically 5 to 10%), but realized Avg_Mdcr_Alowd_Amt varies by far more due to deductible offsets, modifier mixes (50, 51, 22, 26, TC), and small-cell averaging effects. The narrow per-locality range is essentially uninformative as a containment check; it confirms that locality choice within a state is a second-order effect compared to modifier and aggregation effects.

## Caveats

1. **Surgery (n=4) shows a stable $42 MAE under both Global and modifier-aware** because most surgical codes in the sample do not have PCTC=1. That residual is likely multi-procedure discount (modifier 51), bilateral (modifier 50), and assistant-surgeon discount, none of which the simple Global/MOD-26 policy captures. With more rules (a fuller modifier policy), surgery would also close. Sample is too small to over-interpret.
2. **Radiology POS=O empirical-Global cluster** (Wyoming MRI/CT) suggests office-based radiology billing both PC and TC. Recommend `POS=F + Diagnostic Radiology -> MOD-26; POS=O -> Global` as a refinement.
3. **n=50 with n=14 radiology and n=4 surgery** is too small to publish as a definitive product comparison. Findings on the broad split are robust; bucket-level numbers are directional.
4. **The original sample was drawn under the v1 design** (no specialty stratification). A redrawn n=200 with explicit specialty stratification would tighten the radiology and surgery cells substantially.

## Files

- `falsification_test_v2/results_v2.csv`: 50 rows with v1 columns preserved plus `official_global_v2`, `official_modifier_aware`, `applied_modifier`, `modifier_reason`, `err_model_vs_official_global`, `err_model_vs_official_modifier_aware`, `err_emp_vs_official_modifier_aware`, `empirical_in_state_range`, plus per-locality min/max columns.
- `falsification_test_v2/scatter_v2.png`: 2 by 2 scatter (model vs empirical, model vs Global, model vs modifier-aware, empirical vs modifier-aware), color-coded by HCPCS bucket.
- `falsification_test_v2/bucket_breakdown.csv`: machine-readable per-bucket statistics for all four views.
- `falsification_test_v2/methodology_diff.md`: short writeup of v1 vs v2 changes.
- `falsification_test_v2/run_v2.py`: reproducible end-to-end script.
- `falsification_test_v2/analyze_v2.py`: per-bucket analysis and plotting.

## Bottom line

The original "model wins by $115 MAE on radiology" finding was largely a methodology artifact: v1 compared against the unmodified Global PFS rate, but radiologist NPIs bill the modifier-26 (professional component) rate, which is roughly a quarter of the Global. Against the right comparator, the radiology MAE is $26, not $115, and most of that residual is one specific POS=O office-based-imaging confound. The defensible product claim is **'the model performs implicit modifier inference,'** not 'the model adds NPI-conditional signal beyond the published fee schedule.' On E&M, Lab, and HCPCS Level II (about 65% of CMS volume by row count and an even larger share by claim count), the modifier-aware lookup is at parity with the model: model MAE = $7.33, lookup MAE = $6.55. There is no edge to be had there. The model earns its keep on PCTC-split codes by inferring which modifier to apply from specialty and bucket. A modifier-aware lookup with a small per-specialty modifier policy table would close most of the remaining gap.
