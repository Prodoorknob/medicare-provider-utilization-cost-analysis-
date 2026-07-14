# v1 to v2 methodology diff

## What changed

**Data source.** v1 derived the official PFS rate by formula from the CMS PPRRVU and GPCI files using `(rvu_work * gpci_work + rvu_pe * gpci_pe + rvu_mp * gpci_mp) * conv_fact`, with state-area-weighted GPCI averaged across localities in the state. v2 uses the raw CSVs underlying the live PFS Look-Up Tool (`pfs.data.cms.gov/api/1/metastore/schemas/dataset/items`), which expose the same RVUs, GPCIs, and conversion factor as separate `indicators{year}.csv` and `localities{year}.csv` files. v2 prices each locality independently and then averages across the localities in a state, which is mathematically equivalent to v1 when the only varying factor is GPCI. The headline numerical agreement (v1 overall MAE $33.77 vs v2 overall MAE on Global $33.77) confirms the two pipelines are computing the same thing.

**Modifier handling.** This is the central correction. v1 always used the unmodified Global rate as the comparator. v2 selects the modifier per row: for rows where `Rndrng_Prvdr_Type == "Diagnostic Radiology"` and the HCPCS has `pctc == 1` in the indicators file (PC/TC split applies), v2 looks up the row with `modifier == "26"` instead, which is the professional-component rate that radiologists bill when they read a study performed on equipment they do not own. For all other rows the Global (no-modifier) row is used. This correctly aligns the comparator with what radiologist NPIs are actually paid for.

## Numerical changes

- Overall MAE(model vs official) drops from **$33.77** (v1, against Global) to **$15.38** (v2, modifier-aware). Bias drops from -$32.13 to +$0.60.
- Radiology MAE(model vs official) drops from **$91.95** to **$26.29** (a 71% reduction).
- Pearson correlation(model vs official) rises from 0.6469 to 0.8116.
- E&M, HCPCS Level II, and Surgery numbers are unchanged: those buckets do not have PCTC splits in the sample, so the modifier-aware lookup equals the Global lookup.
- MAE(model vs empirical) is unchanged at $10.39 (the model's predictions did not change, only the comparator changed).

## What this means for the v1 'model wins by $115 MAE on radiology' claim

That claim is largely a methodology artifact. The $115 was the gap between model predictions (which closely matched the empirical MOD-26-rate-equivalent payment radiologists actually receive) and the Global rate (which represents PC + TC combined, a number nobody actually collects when those components are billed separately). Once the comparator is the MOD-26 rate, the gap collapses to roughly $26 on radiology and most of that residual is concentrated in three POS=O office-based imaging rows where the radiologist appears to own the equipment and bill globally. The model's value-add is best described as **implicit modifier inference**: it learns from training data which specialty + bucket combinations bill PC-only versus globally, and predicts accordingly. That capability is real and useful, but it is replicable by a modifier-aware lookup with a small per-specialty policy table; it is not strong evidence of NPI-conditional signal beyond the published fee schedule.
