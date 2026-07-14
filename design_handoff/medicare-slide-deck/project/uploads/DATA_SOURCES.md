# AllowanceMap — Additional Data Sources & Datasets

> Last updated: 2026-04-19 (added Known Data Quality Issues section; catalog of candidate sources unchanged since 2026-04-08)
> Purpose: Catalog of datasets that could improve AllowanceMap's predictions, coverage, and validity

---

## Currently Integrated Datasets

| Dataset | Years | Records | Used For |
|---|---|---|---|
| CMS Medicare Physician & Other Practitioners (by Provider & Service) | 2013-2023 | ~103M | Stage 1: primary training data for allowed amount prediction |
| CMS Medicare Provider Summary (by Provider) | 2013-2023 | ~10M NPIs | `Bene_Avg_Risk_Scre` (HCC risk score) joined on NPI + year |
| MCBS Cost Supplement PUF | 2018-2023 | ~30K/year | Stage 2: OOP distributions (national aggregate, used for synthetic data generation) |
| MCBS Survey File PUF | 2015-2023 | ~15K/year | Demographic distributions for synthetic data generation |

---

## Known Data Quality Issues

### CMS specialty names are inconsistent across years

The raw `Rndrng_Prvdr_Type` string for a single clinical specialty can change year-to-year in the CMS Physician & Other Practitioners dataset. This is CMS's own data-entry inconsistency, not a schema change — the specialty is the same, but the string differs. When `LabelEncoder` is fit across the 2013–2023 union of strings, each variant becomes a distinct encoded index with disjoint year coverage.

**Known affected specialties (confirmed 2026-04-19):**

- `"Cardiology"` (2013–2015, 2017–2023) vs `"Cardiovascular Disease (Cardiology)"` (2016 only) → idx 16 vs idx 17
- `"Colorectal Surgery (Proctology)"` vs `"Colorectal Surgery (Formerly Proctology)"` → idx 28 vs idx 29
- `"Oral Surgery (Dentist Only)"` vs `"Oral Surgery (Dentists Only)"` → idx 87 vs idx 88

A full audit of the 131 encoded specialty strings for Levenshtein-close duplicates has not been run — more pairs may exist.

**Impact:** Affected specialties have their rows split across two encoded indices, so the sequence builder produces short/partial time series for each index (e.g., idx=17 has a 1-year sequence covering only 2016). Forecast models (LSTM, Chronos, Stacker) trained on this data produce unreliable forecasts for these specialties. The proper fix is a name-canonicalization step in the silver cleaning layer before encoding. Tracked as backlog item #5 in `MODELING.md`.

**Interim workarounds:**
- Frontend `TOP_SPECIALTY_IDXS` points at the richer-history index (e.g., idx=16 for Cardiology).
- Users should not trust forecasts for idx 17, 28 (or 29), 87 (or 88) until the silver-layer fix lands and models are retrained.

---

## Tier 1: Free Public Datasets (High Priority)

### Medicare Physician Fee Schedule (MPFS)

**Source:** CMS.gov — Medicare Physician Fee Schedule Look-Up Tool and downloadable RVU files

**What it contains:**
- Relative Value Units (RVUs) for every HCPCS/CPT code: Work RVU, Practice Expense RVU, Malpractice RVU
- Geographic Practice Cost Index (GPCI) adjustment factors by Medicare locality (~100 localities)
- Conversion Factor (CF): dollar multiplier applied to total RVUs (updated annually, ~$33-37)
- Facility/non-facility differentials for practice expense RVU
- Global surgery indicators, multiple procedure reduction rules, bilateral surgery indicators

**Why it matters:**
Medicare allowed amounts are fundamentally calculated as: `Allowed = [(Work RVU × Work GPCI) + (PE RVU × PE GPCI) + (MP RVU × MP GPCI)] × CF`. This is the exact pricing mechanism. Currently, the model learns this formula implicitly from data. Joining the fee schedule would:
1. Replace `Avg_Sbmtd_Chrg` (the dominant feature at 62% importance) with the actual pricing inputs
2. Transform the model from "learn the fee schedule" to "learn deviations from the fee schedule"
3. Make the model publishable — predicting deviations is a novel contribution; learning the fee schedule is not

**Format:** CSV/Excel, updated quarterly. Historical files available back to 2010.

**Access:** Free, no registration. Download from https://www.cms.gov/medicare/payment/physician-fee-schedule

**Integration path:**
1. Download MPFS RVU files for 2013-2023 (11 annual files)
2. Join on HCPCS code + year → add `work_rvu`, `pe_rvu`, `mp_rvu`, `total_rvu` as features
3. Download GPCI files → join on state/locality → add `gpci_work`, `gpci_pe`, `gpci_mp`
4. Compute `expected_allowed = total_rvu_adjusted × conversion_factor`
5. New target: `deviation = actual_allowed - expected_allowed` (or ratio)

**Estimated size:** ~15MB per year, ~165MB total

---

### Area Health Resources File (AHRF)

**Source:** HRSA (Health Resources and Services Administration)

**What it contains:**
- County-level healthcare supply metrics: physicians per capita (by specialty), hospital beds per capita, dentists per capita
- Health Professional Shortage Area (HPSA) designations
- Hospital characteristics: teaching status, bed count, ownership type
- Population demographics: age distribution, race/ethnicity, poverty rate, insurance coverage
- Medicare-specific: Part A/B enrollment, managed care penetration
- ~6,000 data elements across ~3,200 counties

**Why it matters:**
Healthcare costs are driven by supply and demand. A cardiologist in rural Montana (one of three in the county) operates in a fundamentally different cost environment than one in Manhattan (competing with hundreds). Currently, the only geographic feature is state-level encoding, which is too coarse to capture within-state variation.

**Format:** SAS transport file (.sas7bdat), CSV extract available. Annual release.

**Access:** Free, download from https://data.hrsa.gov/topics/health-workforce/ahrf

**Integration path:**
1. Download AHRF county-level file
2. Map provider ZIP codes (from CMS data) to county FIPS codes
3. Join → add features: `docs_per_capita`, `beds_per_capita`, `is_hpsa`, `ma_penetration_rate`
4. Alternative: aggregate to state level if ZIP-to-county mapping is unavailable

**Estimated size:** ~200MB (single file with all years)

---

### American Community Survey (ACS)

**Source:** U.S. Census Bureau

**What it contains:**
- County and ZIP-level demographics: median household income, poverty rate, educational attainment
- Insurance coverage rates: uninsured %, employer-sponsored %, Medicaid %, Medicare %
- Age distribution by geography
- Race/ethnicity composition
- Housing and economic indicators

**Why it matters:**
Stage 2 OOP prediction currently uses synthetic demographics at the Census region level (4 regions). ACS data at the county level would provide ~3,200 geographic granularity levels instead of 4. Median income and uninsured rate at the county level are strong predictors of healthcare utilization patterns and cost-sharing behavior.

**Format:** CSV via Census API or bulk download. 1-year and 5-year estimates available.

**Access:** Free, API with optional key. https://data.census.gov

**Integration path:**
1. Use 5-year ACS estimates (more stable for small geographies)
2. Join on county FIPS or ZIP code
3. Add features: `county_median_income`, `county_uninsured_rate`, `county_medicare_pct`, `county_poverty_rate`
4. For Stage 2: replace coarse Census region with county-level economic context

**Estimated size:** ~50MB per year for relevant tables

---

### BLS Medical Care CPI

**Source:** Bureau of Labor Statistics

**What it contains:**
- Monthly Consumer Price Index for Medical Care (CPI-M)
- Sub-indices: physician services, hospital services, prescription drugs, dental services
- Available nationally and by Census region
- Historical data back to 1950s

**Why it matters:**
The LSTM forecasting model currently has no explicit inflation input. Medical costs increase 2-5% annually, and this rate varies by service type and region. Adding CPI as a temporal feature would:
1. Give the LSTM a baseline trend to build on (currently it must learn inflation from the target sequences alone)
2. Improve forecast accuracy for 2024-2026 projections
3. Enable inflation-adjusted analysis ("real" cost changes vs. nominal)

**Format:** CSV via BLS API or bulk download

**Access:** Free, no registration. https://www.bls.gov/cpi/

**Integration path:**
1. Download monthly Medical Care CPI for 2013-2023
2. Compute annual average by year
3. For LSTM: add `cpi_medical` as a time-varying feature alongside the target sequence
4. For tree models: add `cpi_year` or `cumulative_inflation_since_2013` as a feature

**Estimated size:** <1MB

---

### NPPES NPI Registry

**Source:** CMS National Plan and Provider Enumeration System

**What it contains:**
- Every NPI in the US: provider name, credential, taxonomy code, practice address, phone
- Organization vs. individual indicator
- Primary and secondary taxonomy codes (more granular than CMS specialty designation)
- Sole proprietor flag
- Enumeration date (proxy for years in practice)

**Why it matters:**
Currently, providers are characterized only by their specialty, state, and risk score. The NPPES data could add:
1. Provider experience (years since NPI enumeration)
2. Multi-specialty classification (providers with multiple taxonomy codes)
3. Organizational affiliation (solo vs. group practice)
4. More precise geocoding (practice ZIP code for county-level features)

**Format:** CSV, updated monthly. Single file ~8GB (full registry)

**Access:** Free, bulk download from https://download.cms.gov/nppes/NPI_Files.html

**Integration path:**
1. Download NPPES monthly file
2. Join on NPI (available in the raw CMS data, dropped during current pipeline — would need to preserve it)
3. Compute `years_since_enumeration`, `n_taxonomy_codes`, `is_organization`
4. Extract ZIP code for geographic feature joins

**Estimated size:** ~8GB raw, ~500MB after filtering to relevant NPIs

---

### CMS Geographic Variation Public Use File

**Source:** CMS Office of Enterprise Data and Analytics

**What it contains:**
- State and county-level Medicare spending per beneficiary (standardized and actual)
- Utilization rates: inpatient days, E&M visits, imaging, procedures per 1000 beneficiaries
- Quality indicators: readmission rates, mortality rates, ER visit rates
- Separate files for Parts A, B, D, and total

**Why it matters:**
This dataset captures the "practice pattern intensity" of a geographic area — a well-studied driver of Medicare cost variation (the Dartmouth Atlas findings). Areas with more hospital beds and specialists per capita tend to have higher utilization and costs, independent of patient health.

**Format:** CSV, annual release

**Access:** Free, https://data.cms.gov/summary-statistics-on-use-and-payments/medicare-geographic-comparisons

**Integration path:**
1. Download state-level file (63 rows/year, 11 years)
2. Join on state abbreviation + year
3. Add features: `state_spending_per_bene`, `state_utilization_rate`, `state_readmission_rate`

**Estimated size:** ~5MB total

---

### CMS Provider of Services (POS) File

**Source:** CMS

**What it contains:**
- Characteristics of every Medicare-certified facility: bed count, teaching status, ownership type (for-profit, non-profit, government), urban/rural classification
- System affiliation
- CMS certification number, accreditation status

**Why it matters:**
For facility-based services (place_of_service = 1), the type of facility matters enormously. A teaching hospital has higher costs than a community hospital. A for-profit ambulatory surgical center has different pricing from a non-profit hospital outpatient department.

**Format:** CSV, quarterly update

**Access:** Free, https://data.cms.gov/provider-characteristics/hospitals-and-other-facilities

**Integration path:**
1. Download POS file
2. Join on provider organization NPI or CMS certification number
3. Add features: `facility_bed_count`, `is_teaching`, `ownership_type`, `is_urban`

**Estimated size:** ~100MB

---

### Medicare Inpatient / Outpatient Hospital Data

**Source:** CMS Provider Utilization and Payment Data

**What it contains:**
- **Inpatient:** DRG-level charges, payments, and discharge volumes by hospital
- **Outpatient:** APC-level charges, payments, and service volumes by hospital
- Hospital-level aggregate statistics

**Why it matters:**
AllowanceMap currently covers only physician/practitioner Part B services. Hospital facility fees (Part A for inpatient, Part B for outpatient) are a separate and often larger component of total cost. Integrating hospital data would:
1. Enable total episode cost estimation (physician fee + facility fee)
2. Cover procedure types that are predominantly hospital-based (inpatient surgery, ER visits)
3. Support geographic cost comparison at the hospital level

**Format:** CSV, annual

**Access:** Free, https://data.cms.gov/provider-summary-by-type-of-service

**Integration path:**
1. Download hospital outpatient file
2. Map APC codes to HCPCS codes (CMS provides a crosswalk)
3. Add hospital facility fee as a companion to physician allowed amount
4. Display in app: "Physician fee: $X + Facility fee: $Y = Total: $Z"

**Estimated size:** ~200MB/year

---

### Medicare Part D Prescriber Data

**Source:** CMS

**What it contains:**
- Drug prescribing patterns by NPI: drug name, generic indicator, claim count, total cost, beneficiary count
- 30-day supply equivalents
- Brand vs. generic utilization rates

**Why it matters:**
Specialty cost profiles are influenced by prescribing patterns. An oncologist who prescribes expensive biologics vs. one who primarily uses generic chemotherapy agents will have different overall cost footprints. Part D data could:
1. Add a prescribing intensity feature per NPI
2. Identify high-cost drug specialties for LSTM forecasting
3. Improve Stage 2 OOP — drug costs are a major OOP driver for Medicare beneficiaries

**Format:** CSV, annual

**Access:** Free, https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers

**Integration path:**
1. Download Part D prescriber files for 2013-2023
2. Aggregate by NPI + year: total drug cost, brand %, unique drugs prescribed
3. Join to gold data on NPI + year (NPI would need to be preserved through the pipeline)

**Estimated size:** ~2GB/year

---

## Tier 2: Free Public Datasets (Medium Priority)

### CDC PLACES / WONDER

**Source:** CDC

**What it contains:**
- PLACES: county and census-tract level chronic disease prevalence estimates (diabetes, hypertension, obesity, etc.)
- WONDER: mortality data, birth data, environmental health data by county

**Why it matters:** Adds population health burden context beyond individual risk scores. A provider in a county with 15% diabetes prevalence operates in a different cost environment than one in a county with 8%.

**Access:** Free, https://www.cdc.gov/places and https://wonder.cdc.gov

**Integration:** Join PLACES on county FIPS. Add `county_diabetes_prevalence`, `county_obesity_rate`, etc.

---

### Dartmouth Atlas

**Source:** Dartmouth Institute for Health Policy & Clinical Practice

**What it contains:**
- Hospital Referral Region (HRR) and Hospital Service Area (HSA) level Medicare utilization and spending
- End-of-life spending patterns
- Surgical variation data

**Why it matters:** Canonical source for understanding geographic variation in Medicare spending. HRR-level features capture referral patterns that state-level data misses.

**Access:** Free, https://www.dartmouthatlas.org

**Integration:** Map provider ZIP to HRR (crosswalk available). Add `hrr_spending_index`, `hrr_surgical_rate`.

---

### Medicare Advantage Plan Data

**Source:** CMS

**What it contains:**
- MA plan enrollment by county
- Star ratings, premium, deductible by plan
- MA penetration rate by county

**Why it matters:** In areas with high MA penetration, fee-for-service Medicare volume drops and provider behavior changes. MA penetration rate is a market competition signal.

**Access:** Free, CMS plan finder data

**Integration:** County-level MA penetration rate as a feature. Already partially available in AHRF.

---

### Hospital Compare / Care Compare

**Source:** CMS

**What it contains:**
- Hospital quality metrics: mortality, readmission, patient experience (HCAHPS), timely care, complications
- Provider quality metrics: MIPS scores (since 2017)

**Why it matters:** Quality metrics could help explain cost variation — high-quality providers may have different cost profiles (either lower costs from efficiency or higher costs from more thorough care).

**Access:** Free, https://data.cms.gov/provider-data

**Integration:** Join hospital quality on facility CMS certification number, provider quality on NPI.

---

## Tier 3: Requires Data Use Agreement or Purchase

### MCBS Limited Data Set (LDS)

**Source:** CMS, via Research Data Assistance Center (ResDAC)

**What it contains:**
- Individual-level linked survey + claims data for ~15K Medicare beneficiaries per year
- Full demographics: age, sex, race, income, education, Census region, MSA
- All Medicare claims: Part A, B, D with diagnosis codes, procedure codes, payments
- Out-of-pocket spending by service type
- Supplemental insurance details: Medigap plan type, employer coverage, Medicaid
- Health status: self-reported health, functional limitations, chronic conditions (detailed)

**Why it matters:** This is the dataset AllowanceMap needs to make Stage 2 real. It provides the actual per-beneficiary linkage between demographics and OOP spending that the synthetic data fabricates. It includes Census region (suppressed in PUF), actual Medigap plan type (not just binary yes/no), and real expenditure breakdowns.

**Cost:** $600 per module per year (Survey Module, Cost Module, Event Module, Topical Module). Full 5-year coverage would be ~$6,000.

**Access:** Data Use Agreement required, 6-8 week processing time. Apply through ResDAC (https://www.resdac.org)

**Integration:** Direct replacement of synthetic_oop.parquet. Run existing pipeline with `--mode puf` on real data. The crosswalk script already handles the real data path.

---

### CMS Virtual Research Data Center (VRDC) / Research Identifiable Files (RIF)

**Source:** CMS, via ResDAC

**What it contains:**
- 100% Medicare claims for all ~60M beneficiaries
- Part A (inpatient), Part B (outpatient/physician), Part D (drugs)
- Beneficiary enrollment and demographics
- Diagnosis codes, procedure codes, provider NPIs, facility IDs
- Exact payment amounts, deductibles, coinsurance

**Why it matters:** The gold standard. Every question AllowanceMap tries to answer could be answered definitively with this data. Individual-level predictions, real OOP, geographic granularity, temporal trends — all from a single source.

**Cost:** Varies by project. Requires institutional affiliation, IRB approval, DUA, and a formal research proposal. VRDC access charges are hourly for compute time.

**Access:** Highly restricted. Typical timeline: 6-12 months from application to data access.

---

### Truven/IBM MarketScan (now Merative)

**Source:** Merative (formerly IBM Watson Health / Truven Health Analytics)

**What it contains:**
- Commercial + Medicare Supplemental claims for ~250M covered lives
- Complete claims: inpatient, outpatient, pharmacy, enrollment
- Actual plan-level detail: deductible, copay, coinsurance amounts
- Employer-sponsored and Medicare Supplemental (Medigap) plan specifics
- Longitudinal: same individuals tracked across years

**Why it matters:** Fills the gap between MCBS (small sample, limited geographic detail) and full Medicare claims (restricted access). MarketScan's Medicare Supplemental database specifically covers Medigap enrollees, which is exactly the population Stage 2 is trying to model.

**Cost:** University license typically $10K-50K/year depending on modules and institution. Some universities have existing licenses.

**Access:** Commercial license through Merative. Many health science universities already have access. Check with your institution's data governance office.

---

### FAIR Health

**Source:** FAIR Health, Inc.

**What it contains:**
- Benchmark allowed amounts and charges by CPT code × geographic area (ZIP-3)
- Commercial and Medicare data
- Cost transparency tools
- Historical trends

**Why it matters:** Independent validation of Stage 1 predictions. FAIR Health's benchmarks are the industry standard for healthcare cost estimation. Comparing AllowanceMap's predictions against FAIR Health benchmarks would validate (or challenge) the model's accuracy.

**Cost:** Paid API, pricing varies. Academic pricing available.

**Access:** https://www.fairhealth.org

---

### OptumLabs Data Warehouse

**Source:** OptumLabs (UnitedHealth Group)

**What it contains:**
- De-identified claims for ~200M commercial and Medicare Advantage lives
- Complete medical and pharmacy claims
- Lab results, electronic health records (in linked version)
- Social determinants of health indicators

**Why it matters:** Largest commercial claims database. MA claims would complement the FFS Medicare data AllowanceMap currently uses, since ~50% of Medicare beneficiaries are now in MA plans.

**Cost:** Collaborative research agreement required. Typically academic partnerships.

**Access:** Apply through OptumLabs partner network. Requires approved research protocol.

---

## Tier 4: Emerging / Specialty Sources

### CMS Transparency in Coverage (TiC) Machine-Readable Files

**Source:** Health insurance issuers (mandated by CMS since 2022)

**What it contains:**
- In-network negotiated rates for every covered item/service by every commercial health plan
- Out-of-network allowed amounts
- Provider-specific negotiated rates by NPI + plan

**Why it matters:** This is the most granular price transparency data ever released. It reveals the actual negotiated rates between insurers and providers, which vary dramatically even within the same geography. Could enable a "commercial comparison" feature: "Medicare allows $X, the average commercial plan pays $Y."

**Challenges:** Files are enormous (terabytes per insurer), poorly structured, and require significant engineering to parse. The Dolthub community has started aggregating these.

**Access:** Free (public mandate), but practically difficult to use at scale.

---

### All-Payer Claims Databases (APCDs)

**Source:** State-level (currently ~20 states have APCDs)

**What it contains:**
- All insurance claims (commercial, Medicare, Medicaid) within a state
- Standardized format across payers
- States with APCDs: CO, CT, MA, MD, ME, MN, NH, NY, OR, RI, UT, VA, VT, WA, and others

**Why it matters:** True all-payer data allows comparing Medicare vs. commercial vs. Medicaid costs for the same service in the same geography. This is unavailable from any single federal source.

**Access:** Varies by state. Most require research application and DUA. Some offer public use files.

---

### HCUP (Healthcare Cost and Utilization Project)

**Source:** AHRQ (Agency for Healthcare Research and Quality)

**What it contains:**
- National and state inpatient/emergency department/ambulatory surgery databases
- Diagnosis and procedure codes, charges, payer, demographics
- Weighted to national estimates

**Why it matters:** Fills the hospital-based care gap. HCUP's NIS (National Inpatient Sample) is the standard for studying inpatient costs and utilization patterns.

**Cost:** $350-$600 per database per year (academic pricing)

**Access:** DUA required, https://hcup-us.ahrq.gov

---

## Integration Priority Matrix

| Priority | Dataset | Impact on AllowanceMap | Effort | Cost |
|---|---|---|---|---|
| 1 | Medicare Fee Schedule (MPFS) | Eliminates charge dependency, adds pricing mechanism features | Medium | Free |
| 2 | AHRF | Adds healthcare supply/demand features by county | Low | Free |
| 3 | ACS Census Demographics | Replaces coarse region encoding with county-level context | Low | Free |
| 4 | Medical CPI (BLS) | Improves LSTM temporal forecasting | Low | Free |
| 5 | CMS Geographic Variation | Adds practice pattern intensity features | Low | Free |
| 6 | NPPES NPI Registry | Adds provider experience, taxonomy, organization features | Medium | Free |
| 7 | MCBS LDS | Validates Stage 2 with real beneficiary OOP | Medium | $600/module |
| 8 | Hospital Outpatient Data | Enables total episode cost (physician + facility) | Medium | Free |
| 9 | Part D Prescriber Data | Adds drug prescribing intensity features | Medium | Free |
| 10 | POS File | Adds facility characteristics for facility-based services | Low | Free |
| 11 | CDC PLACES | Adds county-level disease prevalence | Low | Free |
| 12 | Truven MarketScan | Full plan-level OOP validation | High | $10-50K/yr |
| 13 | CMS VRDC | Gold standard: 100% claims | Very High | Institutional |
| 14 | FAIR Health | Independent benchmark validation | Low | Paid API |
| 15 | Transparency in Coverage MRFs | Commercial rate comparison | Very High | Free (but massive) |

---

## Quick Wins (can integrate in a single session)

1. **MPFS RVU files** — 11 CSV downloads, join on HCPCS + year, adds 4-7 features
2. **Medical CPI** — single API call, 11 data points, one LSTM feature
3. **CMS Geographic Variation** — single CSV download, join on state + year, adds 3 features

These three datasets address the top three model weaknesses (charge dominance, temporal forecasting, geographic granularity) with minimal engineering effort.
