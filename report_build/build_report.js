// Build AllowanceMap project report (docx). Run from this directory.
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, PageOrientation, LevelFormat,
  HeadingLevel, BorderStyle, WidthType, ShadingType, PageNumber, PageBreak,
  TabStopType, TabStopPosition,
} = require("docx");

// ---------- helpers ----------
const CONTENT_WIDTH = 9360; // US Letter minus 1" margins on each side

function P(text, opts = {}) {
  if (typeof text === "string") {
    return new Paragraph({
      spacing: { after: 120, line: 300 },
      alignment: opts.align || AlignmentType.LEFT,
      children: [new TextRun({ text, bold: !!opts.bold, italics: !!opts.italics, size: opts.size || 22 })],
    });
  }
  // text is array of TextRun
  return new Paragraph({
    spacing: { after: 120, line: 300 },
    alignment: opts.align || AlignmentType.LEFT,
    children: text,
  });
}

function H1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 180 },
    children: [new TextRun({ text, bold: true, size: 32 })],
  });
}
function H2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 140 },
    children: [new TextRun({ text, bold: true, size: 26 })],
  });
}
function H3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 100 },
    children: [new TextRun({ text, bold: true, size: 23 })],
  });
}

function Bullet(text) {
  const runs = typeof text === "string"
    ? [new TextRun({ text, size: 22 })]
    : text;
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 60, line: 300 },
    children: runs,
  });
}

// bold-leading bullet: "Key:" text ... rest
function KVBullet(key, rest) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 60, line: 300 },
    children: [
      new TextRun({ text: key + " ", bold: true, size: 22 }),
      new TextRun({ text: rest, size: 22 }),
    ],
  });
}

const border = { style: BorderStyle.SINGLE, size: 6, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };

function HCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: "D9E2F3", type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({
      spacing: { before: 0, after: 0 },
      children: [new TextRun({ text, bold: true, size: 20 })],
    })],
  });
}
function DCell(text, width, opts = {}) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({
      spacing: { before: 0, after: 0 },
      alignment: opts.align || AlignmentType.LEFT,
      children: [new TextRun({ text, size: 20, bold: !!opts.bold })],
    })],
  });
}

function mkTable(columnWidths, headers, rows) {
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => HCell(h, columnWidths[i])),
  });
  const dataRows = rows.map(r => new TableRow({
    children: r.map((cell, i) => {
      const text = typeof cell === "string" ? cell : cell.text;
      const opts = typeof cell === "string" ? {} : cell;
      return DCell(text, columnWidths[i], opts);
    }),
  }));
  return new Table({
    width: { size: columnWidths.reduce((a,b)=>a+b,0), type: WidthType.DXA },
    columnWidths,
    rows: [headerRow, ...dataRows],
  });
}

// ---------- content ----------
const children = [];

// Title block
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 80 },
  children: [new TextRun({ text: "AllowanceMap", bold: true, size: 44 })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 60 },
  children: [new TextRun({ text: "Predicting Medicare Allowed Amounts and Patient Out-of-Pocket Costs", size: 26 })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 40 },
  children: [new TextRun({ text: "A Two-Stage Pipeline on CMS Physician & Practitioner Data (2013–2023)", italics: true, size: 22 })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 280 },
  children: [new TextRun({ text: "Raj Vedire  |  Applied Data Science  |  April 2026", size: 20 })],
}));

// ---------- 1. Executive Summary ----------
children.push(H1("1. Executive Summary"));
children.push(P("AllowanceMap is an end-to-end data-science project that predicts (a) the average Medicare allowed amount for a physician service at a given specialty, procedure, state, and place of service, (b) the distribution of out-of-pocket cost a beneficiary can expect to pay for that service given their demographics and coverage, and (c) the three-year trajectory of allowed amounts through 2026. The pipeline ingests approximately 126.8 million CMS Medicare Physician & Other Practitioners rows covering 2013–2023, joins in NPI-level HCC risk scores, layers in MCBS survey-based demographic distributions, and trains six model families across two cost stages and one forecasting stage. Production models are a LightGBM \u201Cno-charge\u201D regressor for Stage 1 (R\u00B2 = 0.943), an XGBoost quantile regressor for Stage 2 OOP at the P10/P50/P90 level (P50 R\u00B2 = 0.40), and a LightGBM stacker over an LSTM + Chronos-Bolt ensemble for forecasting (R\u00B2 = 0.885 on a held-out 2022–2023 window). The models are deployed behind a FastAPI service on Railway, a Supabase-backed reference data layer, and a Next.js web app on Vercel."));

// ---------- 2. Problem ----------
children.push(H1("2. Problem"));
children.push(P("Medicare beneficiaries, clinicians, and administrators share a common question: for a specific service rendered to a specific kind of patient, how much will Medicare allow, and how much of that will the patient actually owe at the point of care? Despite the CMS fee-schedule formula being mechanically deterministic (RVU \u00D7 GPCI \u00D7 Conversion Factor), the real-world answer a patient sees depends on modifiers, place of service, supplemental coverage, Medicaid dual-eligibility, chronic-condition burden, and billing idiosyncrasies. The gap between \u201Cthe fee schedule\u201D and \u201Cwhat you pay\u201D is where confusion, surprise billing, and non-adherence happen."));
children.push(P("AllowanceMap addresses two linked prediction problems and one forecasting problem:"));
children.push(KVBullet("Stage 1 \u2014 Provider cost prediction:", "given a (specialty, procedure, state, place of service) tuple, predict the average allowed amount Medicare will approve per service (target: Avg_Mdcr_Alowd_Amt)."));
children.push(KVBullet("Stage 2 \u2014 Patient out-of-pocket prediction:", "given the Stage 1 prediction plus beneficiary context (age band, sex, income, chronic count, dual eligibility, supplemental coverage, Census region), predict the distribution of patient OOP at the P10, P50, and P90 quantiles rather than a single point estimate."));
children.push(KVBullet("Forecast track \u2014 Temporal projection:", "for each (specialty \u00D7 HCPCS bucket \u00D7 state) group, project the allowed-amount trajectory for 2024–2026 with uncertainty bounds to support planning and trend exploration."));
children.push(P("The project treats prediction and inference as complementary: strong regression accuracy on the out-of-sample holdout answers the \u201Chow well can we forecast this?\u201D question, while feature-importance analysis, ablation studies, and coefficient interpretation answer the \u201Cwhat drives allowed amount and OOP?\u201D question that a data scientist, clinician, or policy analyst actually wants."));

// ---------- 3. What exists today ----------
children.push(H1("3. What Currently Exists"));
children.push(P("Medicare cost estimation is not a new problem, but the existing public-facing options are surprisingly limited. CMS publishes the Medicare Physician Fee Schedule (MPFS) look-up tool, which returns the national allowed amount for a single HCPCS code after applying the standard RVU \u00D7 GPCI \u00D7 CF formula. It does not predict what a patient will pay, does not condition on demographics, and does not project future trajectories."));
children.push(P("Beyond CMS, FAIR Health publishes consumer-facing benchmarks derived from private-insurer claim data, but its Medicare coverage is secondary, pricing is paywalled at scale, and its estimates are point values rather than quantile distributions. Hospital price-transparency machine-readable files (mandated by CMS since 2022) contain enormous amounts of negotiated-rate data, but they are notoriously hard to parse and almost never address the Medicare patient directly. Academic research on Medicare spending patterns (Dartmouth Atlas, CMS Geographic Variation PUF) is descriptive rather than predictive \u2014 it tells you what has already been spent, not what a specific new service will cost."));
children.push(P("None of these sources combine all of the following in a single tool: (1) a trained predictive model rather than a formula look-up, (2) an uncertainty-bounded patient out-of-pocket estimate, (3) a forward-looking multi-year forecast, and (4) a public user interface. AllowanceMap fills that gap end-to-end."));

// ---------- 4. Data Sources ----------
children.push(H1("4. Data Sources and Why They Matter"));

children.push(H2("4.1 Primary Training Data"));
children.push(KVBullet("CMS Medicare Physician & Other Practitioners, by Provider and Service (2013\u20132023).", "~103M rows across 11 years. This is the foundation dataset for Stage 1. Each row is one (NPI \u00D7 HCPCS \u00D7 place-of-service) combination in a given year with average submitted charge, average allowed amount, and utilization counts. Pulled via the CMS Data API and partitioned per state/specialty before ingestion."));
children.push(KVBullet("CMS Medicare Physician, by Provider (2013\u20132023).", "A separate CMS dataset with one row per NPI per year. Used to join the beneficiary average HCC risk score (Bene_Avg_Risk_Scre) onto the service-level rows. Risk score captures how sick a provider\u2019s patient panel is and is a meaningful conditioner on average cost."));
children.push(KVBullet("Medicare Current Beneficiary Survey (MCBS) \u2014 Cost Supplement and Survey PUFs.", "National-aggregate beneficiary-level data on demographics, chronic conditions, insurance mix, and out-of-pocket spending. Used to derive realistic demographic distributions and national OOP statistics for Stage 2. Covers 2015\u20132023 (Survey) and 2018\u20132023 (Cost)."));

children.push(H2("4.2 Known-Future Covariates (Forecasting)"));
children.push(KVBullet("Medicare Conversion Factor (CMS-published).", "The dollar multiplier applied to total RVUs each year. Known in advance for 2024\u20132026, so used as a time-varying-known-real input in the multivariate TFT and as a deflator in Chronos preprocessing."));
children.push(KVBullet("Medical Consumer Price Index (BLS).", "Medical-care CPI sub-series pulled from the BLS API. Used to deflate the target into real terms before Chronos inference, then reinflated at forecast time using known-future CPI."));
children.push(KVBullet("Sequestration rates, MACRA/MIPS adjustment factors, COVID indicator.", "Policy-driven adjustments that shift fee schedules uniformly. Tried as features; carried no orthogonal signal at annual resolution (see Section 7)."));

children.push(H2("4.3 Synthetic MCBS (Stage 2 Proxy)"));
children.push(P("Because MCBS Cost Supplement and Survey PUFs do not share beneficiary IDs and suppress Census region for privacy, linking real patient-level OOP to a provider-side (specialty \u00D7 state) tuple is impossible from public data. The project builds a synthetic per-service OOP table by sampling demographic distributions from real MCBS years, then modulating OOP by dual-eligibility (-85%), supplemental-insurance (-35%), chronic-count scaling, and log-normal noise. The pipeline is designed as a drop-in replacement: a researcher with a real MCBS LDS data-use agreement can swap in actual per-beneficiary OOP without touching the modeling code."));

children.push(H2("4.4 Why the Data Matters for Modeling"));
children.push(P("The CMS provider-service dataset provides sufficient cardinality (~6,000 unique HCPCS codes, 131 specialties, 56 states/territories, two places of service, 11 years) that tree-based boosters can learn the implicit fee-schedule formula, then learn deviations from it. The risk score from the separate provider-level file is the single most impactful demographic conditioner available to Stage 1 without accessing claim-level data. MCBS contributes the demographic distributions needed to make Stage 2 behave like a real beneficiary population rather than a uniform sample. Known-future covariates (CF, CPI) are the only reason the forecast track can project into 2024\u20132026 without leaking information \u2014 they are the legitimate oracle inputs."));

// ---------- 5. Architecture ----------
children.push(H1("5. System Architecture"));
children.push(P("The pipeline follows a medallion architecture (Bronze \u2192 Silver \u2192 Gold) with two execution modes: a Databricks/PySpark path for the S3-backed production run and a local Python/pandas path for developer iteration and model training. Both paths log to a unified MLflow experiment hosted in the Databricks workspace so that local and cloud experiments appear in one leaderboard."));

children.push(H3("5.1 Data Layer"));
children.push(KVBullet("Bronze.", "Raw CMS CSVs converted to Parquet with a year column injected from the filename. Schema preserved verbatim (all strings)."));
children.push(KVBullet("Silver.", "Typed, IQR-outlier-trimmed data with null handling. Outlier bounds computed on the allowed-amount target to avoid clipping legitimate high-cost procedures."));
children.push(KVBullet("Gold.", "Per-state Parquets with label-encoded categoricals, log-transformed volume features, and persisted LabelEncoder classes saved to gold/label_encoders.json for cross-script consistency. A separate gold_year/ directory preserves the year column for temporal modeling."));

children.push(H3("5.2 Modeling Layer"));
children.push(P("Six model families are trained: GLM (SGD with Huber loss), Random Forest, XGBoost, CatBoost, LightGBM, and an LSTM. Every booster supports a batch mode (incremental training by Census region for memory safety) and a full mode (single pass with early stopping, run on Colab Pro A100). The LSTM is PyTorch with static embeddings for group keys, 2-layer recurrent cells, teacher-forced training, autoregressive inference rollout, and MC Dropout for uncertainty."));

children.push(H3("5.3 Serving Layer"));
children.push(P("A FastAPI service on Railway hosts the Stage 1 LightGBM booster (pickled, ~26 MB) and Stage 2 CatBoost/XGBoost quantile artifacts. It auto-detects the no-charge vs with-charge variant from the model\u2019s feature names. Supabase stores pre-computed aggregations (32K Stage 1 groups, 23K Stage 2 groups, 62K forecast rows), label encoders, feature importances, and model metrics. Vercel hosts the Next.js front end."));

// Architecture diagram as simple ascii-like flow
children.push(P("The end-to-end dataflow is:"));
children.push(P("CMS API \u2192 Bronze (raw) \u2192 Silver (typed/cleaned) \u2192 Gold (features, per-state) \u2192 Modeling (6 families + LSTM + stacker) \u2192 MLflow tracking \u2192 Model artifacts (Railway) + pre-computed lookups (Supabase) \u2192 Next.js UI (Vercel)."));

// ---------- 6. Modeling Approach ----------
children.push(H1("6. Modeling Approach"));
children.push(H2("6.1 Target and Split"));
children.push(KVBullet("Stage 1 target:", "Avg_Mdcr_Alowd_Amt (continuous, dollar value). log1p transform applied at train time for skew correction; inverse at prediction."));
children.push(KVBullet("Data leakage check:", "Avg_Mdcr_Pymt_Amt and Avg_Mdcr_Stdzd_Amt are mathematically derived from the allowed amount and were removed from the feature set (an early leaky run scored R\u00B2 = 0.9996, confirming the problem)."));
children.push(KVBullet("Stage 1 split:", "80/20 random split with random_state=42 applied consistently across every model family for fair comparison."));
children.push(KVBullet("Stage 2 target:", "patient OOP at P10/P50/P90 using XGBoost reg:quantileerror with alpha \u2208 {0.1, 0.5, 0.9}. Coverage is evaluated empirically (should hit ~10/50/90%)."));
children.push(KVBullet("Forecast split:", "temporal \u2014 train \u2264 2021, validate on 2022\u20132023. This is the honest evaluation because the 2024\u20132026 forecast has no ground truth at training time."));

children.push(H2("6.2 Feature Set"));
children.push(P("Stage 1 uses 13 features in the full variant and 12 in the no-charge production variant: encoded specialty, encoded state, encoded HCPCS, hcpcs_bucket (coarse clinical category 0\u20135), place_of_service flag, Bene_Avg_Risk_Scre, log_srvcs, log_benes, srvcs_per_bene, a specialty_bucket, a pos_bucket, and an hcpcs_target_encoding. Submitted charge (Avg_Sbmtd_Chrg) is included in the full variant only. Stage 2 uses the Stage 1 output plus 11 beneficiary-side features including Census region, age band, sex, income band, chronic count, dual-eligible flag, and supplemental-insurance flag. Forecasting uses univariate target history plus static group IDs, with a multivariate TFT variant adding 5 observed reals and 3 known-future reals."));

children.push(H2("6.3 Methods Applied"));
children.push(P("The project applies six distinct method families spanning interpretable baselines, non-parametric ensembles, gradient-boosted trees, and deep sequence models:"));
children.push(Bullet("GLM (SGDRegressor with Huber loss, streaming partial_fit) \u2014 interpretable linear baseline."));
children.push(Bullet("Random Forest (warm-start incremental by Census region, RandomizedSearchCV in full mode) \u2014 non-parametric baseline for feature importance and variance bounds."));
children.push(Bullet("XGBoost (incremental booster continuation by region, early stopping in full mode) \u2014 production-grade tabular booster with CUDA support."));
children.push(Bullet("CatBoost (ordered target statistics, native categorical handling, monotonicity constraints in Stage 2) \u2014 tested as an ensemble-diversity member and as a monotonic Stage 2 option."));
children.push(Bullet("LightGBM (GOSS sampling, leaf-wise growth, histogram-based) \u2014 ultimately won Stage 1 on the full data."));
children.push(Bullet("PyTorch LSTM + LightGBM stacker + Chronos-Bolt foundation model \u2014 forecast-track specialists."));

// ---------- 7. Experimentation and Results ----------
children.push(H1("7. Model Experimentation and Results"));

children.push(H2("7.1 Stage 1 \u2014 Allowed Amount"));
children.push(P("V1 trained on a 30% sample with regional batching; V2 retrained the same models on the full 126.8M rows on Colab Pro A100 GPUs. The V2 full-data run closed almost all of the V2-minus-V1 gap through data volume alone rather than architectural change. A no-charge ablation was then run because the production API cannot require Avg_Sbmtd_Chrg at request time."));

children.push(mkTable(
  [2600, 1400, 1400, 1400, 2560],
  ["Model", "MAE ($)", "RMSE ($)", "R\u00B2", "Notes"],
  [
    ["LightGBM V2 (full)",        {text:"6.73", align:AlignmentType.RIGHT}, {text:"13.59", align:AlignmentType.RIGHT}, {text:"0.9575", align:AlignmentType.RIGHT}, "Best single model, charge-required"],
    ["Ensemble V2 (5-fold stack)",{text:"6.68", align:AlignmentType.RIGHT}, {text:"13.50", align:AlignmentType.RIGHT}, {text:"0.9580", align:AlignmentType.RIGHT}, "+0.0005 R\u00B2 for 13.3 hr compute"],
    ["XGBoost V2",                {text:"7.73", align:AlignmentType.RIGHT}, {text:"15.43", align:AlignmentType.RIGHT}, {text:"0.9452", align:AlignmentType.RIGHT}, "Fallback"],
    ["CatBoost V2",               {text:"10.88",align:AlignmentType.RIGHT}, {text:"20.10", align:AlignmentType.RIGHT}, {text:"0.9070", align:AlignmentType.RIGHT}, "Slowest convergence"],
    [{text:"LightGBM V2 no-charge (PROD)", bold:true}, {text:"7.70", align:AlignmentType.RIGHT, bold:true}, {text:"15.77", align:AlignmentType.RIGHT, bold:true}, {text:"0.9428", align:AlignmentType.RIGHT, bold:true}, {text:"Deployed model", bold:true}],
    ["RF V1 (30% sample)",        {text:"12.04",align:AlignmentType.RIGHT}, {text:"22.73", align:AlignmentType.RIGHT}, {text:"0.8843", align:AlignmentType.RIGHT}, "Best V1"],
    ["XGBoost V1",                {text:"11.83",align:AlignmentType.RIGHT}, {text:"21.18", align:AlignmentType.RIGHT}, {text:"0.8331", align:AlignmentType.RIGHT}, "Regional batch"],
  ]
));
children.push(P("Table 1. Stage 1 leaderboard. Full-data retraining moved LightGBM from R\u00B2 0.88 to 0.96 without any architectural change. The no-charge variant was chosen for production despite a 0.015 R\u00B2 gap because the user-facing API cannot require submitted charge as input.", {italics:true, size:20}));

children.push(H2("7.2 Charge Ablation \u2014 An Important Finding"));
children.push(P("Dropping Avg_Sbmtd_Chrg, which had dominated V1 feature importance at 62%, reduced R\u00B2 by only 0.01\u20130.02 across all three boosters. This is the most important inference result in the project: the model is not parroting a near-leaky charge signal. After full-data training with hcpcs_target_enc and HCPCS_Cd_idx, procedure identity captures the pricing information that submitted charge previously provided. The no-charge model remains strictly better than the best V1 model and is safe to deploy in a consumer tool."));

children.push(H2("7.3 Stage 2 \u2014 Out-of-Pocket Quantile Regression"));
children.push(mkTable(
  [2700, 1400, 1400, 1400, 2460],
  ["Model", "P50 MAE", "P50 RMSE", "P50 R\u00B2", "Notes"],
  [
    [{text:"XGB Quantile V1 (PROD)", bold:true}, {text:"9.78", align:AlignmentType.RIGHT, bold:true}, {text:"18.28", align:AlignmentType.RIGHT, bold:true}, {text:"0.400", align:AlignmentType.RIGHT, bold:true}, "Coverage 50%/90% on P50/P90"],
    ["CatBoost Monotonic V2", {text:"10.55", align:AlignmentType.RIGHT}, {text:"21.34", align:AlignmentType.RIGHT}, {text:"0.173", align:AlignmentType.RIGHT}, "Monotonicity too restrictive"],
    ["CatBoost Zero-Inflated V2", {text:"11.95", align:AlignmentType.RIGHT}, {text:"24.15", align:AlignmentType.RIGHT}, {text:"-0.054", align:AlignmentType.RIGHT}, "Gate+regression compound errors"],
  ]
));
children.push(P("Table 2. Stage 2 leaderboard. V1 XGBoost quantile wins on synthetic MCBS. V2 monotonicity constraints are theoretically sound but fight the synthetic data distribution; they are expected to help on real MCBS LDS data.", {italics:true, size:20}));

children.push(H2("7.4 Forecast Track"));
children.push(P("Five forecast architectures were evaluated on the same fair 2022\u20132023 autoregressive holdout (N = 32,481 group-years). A teacher-forcing bug in the original LSTM evaluator had inflated the reported V1 baseline by 0.017 R\u00B2 and $17.51 RMSE; Cell 6 of the V2_12 notebook implements a proper batched autoregressive rollout and is the fair baseline below."));
children.push(mkTable(
  [2800, 1400, 1400, 1400, 2360],
  ["Model", "MAE", "RMSE", "R\u00B2", "Status"],
  [
    ["LSTM V1 (teacher-forced, inflated)", {text:"8.84", align:AlignmentType.RIGHT}, {text:"36.42", align:AlignmentType.RIGHT}, {text:"0.8860", align:AlignmentType.RIGHT}, "Bug artifact"],
    ["LSTM V1 (fair, autoregressive)",    {text:"9.82", align:AlignmentType.RIGHT}, {text:"18.91", align:AlignmentType.RIGHT}, {text:"0.8689", align:AlignmentType.RIGHT}, "Honest baseline"],
    ["Chronos-Bolt cpi_cf_deflated",       {text:"9.39", align:AlignmentType.RIGHT}, {text:"19.71", align:AlignmentType.RIGHT}, {text:"0.8576", align:AlignmentType.RIGHT}, "Best foundation-model variant"],
    ["Multivariate TFT V2_13",             {text:"9.23", align:AlignmentType.RIGHT}, {text:"18.79", align:AlignmentType.RIGHT}, {text:"0.8691", align:AlignmentType.RIGHT}, "Tied fair LSTM; covariates did not help"],
    [{text:"LightGBM Stacker V2_12 (PROD)", bold:true}, {text:"8.74", align:AlignmentType.RIGHT, bold:true}, {text:"17.69", align:AlignmentType.RIGHT, bold:true}, {text:"0.8852", align:AlignmentType.RIGHT, bold:true}, {text:"Deployed", bold:true}],
  ]
));
children.push(P("Table 3. Forecast leaderboard. Three independent architectures cluster at R\u00B2 0.857\u20130.869; only the stacker exceeds this. This is an empirical signal ceiling at annual resolution, not a modeling failure.", {italics:true, size:20}));

children.push(H2("7.5 Systematic Comparison"));
children.push(P("All Stage 1 models were evaluated on the same 80/20 split with the same random seed. Stage 2 models used a 60/20/20 train/calibration/test split for CQR-style quantile calibration. Every model family was logged to a unified MLflow experiment with identical metric names (test_mae, test_rmse, test_r2) so the comparison table in compare_models_local.py is generated directly from MLflow rather than hand-curated. For the forecast track, all five architectures were evaluated on the identical 32,481-row 2022\u20132023 holdout so the R\u00B2 numbers are directly comparable. A 5-fold GroupKFold cross-validation was applied to the forecast stacker to rule out group-level leakage."));

// ---------- 8. Lessons Learned ----------
children.push(H1("8. Lessons Learned"));

children.push(H2("8.1 What Worked (Positive Lessons)"));
children.push(KVBullet("Full-data training dwarfs architectural tuning.",
  "V2 gained +0.07 R\u00B2 over V1 mostly by training on 126.8M rows instead of 30%. This outpaced every hyperparameter search. The first lever to pull when a model is underperforming is more data, not a fancier algorithm."));
children.push(KVBullet("The no-charge ablation was non-negotiable.",
  "V1 feature importance showed 62% of signal concentrated in one near-leaky feature. Deliberately ablating it before deployment revealed that procedure-identity features carry almost the same signal and is the reason the production API can serve users without a billed charge."));
children.push(KVBullet("Ensembling is not always worth it.",
  "A 5-fold stacked ensemble (13.3 hrs compute, ~149 Colab compute units) bought +0.0005 R\u00B2 over LightGBM alone. Operational cost far exceeds benefit. Ship the single model."));
children.push(KVBullet("Stacker diversity beat any single forecast architecture.",
  "The LightGBM stacker over LSTM + Chronos-Bolt won the forecast track by +0.016 R\u00B2, because LSTM and Chronos disagree on which group-years are hard. Two mediocre disagreeing models can outscore any one of them."));
children.push(KVBullet("Temporal splits are the only honest forecast evaluation.",
  "Random-split forecast R\u00B2 inflates reality. Moving to train \u2264 2021 / eval 2022\u20132023 and autoregressive rollout exposed the teacher-forcing bug and the true signal ceiling."));

children.push(H2("8.2 What Did Not Work (Negative Lessons)"));
children.push(KVBullet("Monotonicity constraints on synthetic data hurt performance.",
  "CatBoost Monotonic Stage 2 was theoretically sound (allowed \u2191 \u2192 OOP \u2191, income \u2191 \u2192 OOP \u2191, etc.) but the synthetic MCBS distribution does not perfectly obey these rules, so the constraints fought the data and drove P50 R\u00B2 down to 0.17 vs 0.40 for V1."));
children.push(KVBullet("Zero-inflated gate + regression models compound errors.",
  "The two-stage classifier-then-regressor approach scored negative R\u00B2 (\u22120.054) on synthetic data where the zero-inflation pattern is weak."));
children.push(KVBullet("More covariates did not beat a univariate LSTM at annual resolution.",
  "Multivariate TFT with 5 observed + 3 known-future reals tied the univariate LSTM (R\u00B2 0.8691 vs 0.8689). At annual aggregation, volume/charge/risk trajectories move in near-lockstep with the target and carry no orthogonal signal."));
children.push(KVBullet("Derived-feature Chronos variants collapsed.",
  "Charge-ratio and volume-normalized Chronos series produced catastrophic R\u00B2 (0.19 and \u2212540) because dividing then re-multiplying by a forecasted denominator compounds error multiplicatively."));
children.push(KVBullet("Teacher-forced evaluation lies.",
  "The inherited train_lstm_local.py evaluator used 1-step-ahead teacher-forcing, which inflated the reported V1 LSTM R\u00B2 by 0.017 and RMSE by $17.51, misleading the first three foundation-model comparisons. Always report autoregressive metrics for autoregressive inference."));
children.push(KVBullet("Compute estimates were 3\u20135\u00D7 too low.",
  "Phase 7 training estimates underestimated real wall time across the board \u2014 CatBoost 3000 iters on 126M rows took ~100 min vs 20 min predicted, and the 5-fold ensemble took 13.3 hrs vs ~4 hrs predicted. First-time-on-this-data runs deserve aggressive compute padding."));
children.push(KVBullet("CMS specialty names are not normalized.",
  "The raw Rndrng_Prvdr_Type string changes year-to-year for at least three specialties (Cardiology vs Cardiovascular Disease (Cardiology); Colorectal Surgery (Proctology) vs Colorectal Surgery (Formerly Proctology); Oral Surgery (Dentist/Dentists Only)). Fitting LabelEncoder across the 11-year union splits these into distinct indices with disjoint year coverage and breaks the LSTM sequence builder for affected specialties. Canonicalization belongs in silver, not in the frontend."));

// ---------- 9. Limitations ----------
children.push(H1("9. Limitations"));
children.push(KVBullet("Stage 2 uses synthetic OOP data.",
  "Real MCBS Limited Data Set access requires a $600-per-module fee, a Data Use Agreement, and a 6\u20138 week processing cycle through ResDAC. Until that data is available, Stage 2 R\u00B2 numbers reflect synthetic-distribution performance, not real-patient performance."));
children.push(KVBullet("Forecast signal ceiling at R\u00B2 \u2248 0.885.",
  "Three independent architectures (fair LSTM, Chronos-Bolt, multivariate TFT) all cluster between 0.857 and 0.869 on annual data. Only ensemble diversity breaks that ceiling, and by only 0.016 R\u00B2. Further gains almost certainly require quarterly CMS aggregates, which do not exist in the current data source."));
children.push(KVBullet("Coarse geographic granularity.",
  "The only geographic feature is state-level (56 levels). County- or ZIP-level data (via AHRF, ACS, or NPPES) would expose within-state cost variation that is currently baked into residual noise."));
children.push(KVBullet("No individual patient claims.",
  "Everything in the pipeline is aggregated at the (NPI \u00D7 HCPCS \u00D7 POS \u00D7 year) level. The 100% Medicare Research Identifiable Files could replace every modeled estimate with a direct look-up, but require IRB and institutional access."));
children.push(KVBullet("CMS data-entry inconsistencies.",
  "The specialty-rename issue documented in Section 8 affects training and display for at least three specialties and possibly more. A full Levenshtein audit of all 131 specialty strings is still pending."));
children.push(KVBullet("Part B physician services only.",
  "AllowanceMap does not cover Part A (inpatient), Part D (drugs), or hospital facility fees. Total episode cost estimation would require integrating the CMS Inpatient/Outpatient PUFs."));

// ---------- 10. Comparison to Existing Solutions ----------
children.push(H1("10. Comparison to Existing Solutions"));
children.push(mkTable(
  [2600, 1700, 1700, 1700, 1660],
  ["Capability", "CMS MPFS Tool", "FAIR Health", "Dartmouth Atlas", "AllowanceMap"],
  [
    ["Allowed amount estimate",        "Formula look-up", "Paid API",  "Not offered", {text:"Predicted (R\u00B2 0.94)", bold:true}],
    ["Patient OOP distribution",       "Not offered",     "Point only","Not offered", {text:"P10/P50/P90", bold:true}],
    ["Future-year forecast",           "Not offered",     "Not offered","Not offered",{text:"2024\u20132026", bold:true}],
    ["Uncertainty bounds",             "None",            "None",       "None",        {text:"MC Dropout + quantiles", bold:true}],
    ["Conditions on demographics",     "No",              "Partially",  "Aggregate",   {text:"Yes", bold:true}],
    ["Publicly accessible UI",         "Yes",             "Paywalled",  "Reports",     {text:"Yes (free)", bold:true}],
    ["Open-source code",               "No",              "No",         "Partial",     {text:"Yes", bold:true}],
  ]
));
children.push(P("Table 4. Feature-level comparison against the most comparable public tools. AllowanceMap is the only tool in this set that combines predictive models, quantile uncertainty, future forecasting, and a public UI under an open-source license.", {italics:true, size:20}));

children.push(P("Looping back to Section 3: the gap that motivated this project has been closed on the features AllowanceMap set out to address. CMS\u2019s MPFS tool remains the authoritative look-up for exact national allowed amounts per HCPCS, and nothing in AllowanceMap replaces it as a source of truth. What AllowanceMap adds is the prediction, inference, and forecasting layer that CMS does not provide: an estimate conditioned on demographic context, a quantile-bounded OOP figure, a three-year trajectory with uncertainty, and a free user interface for each."));

// ---------- 11. Frontend ----------
children.push(H1("11. Frontend"));
children.push(P("The patient-facing web application is a Next.js 16 app (App Router, Turbopack) styled with Material UI v7 and visualized with Recharts. It is deployed on Vercel at web-three-omega-21.vercel.app and is backed by two data layers: a Supabase PostgreSQL instance holding 119K+ pre-computed lookup rows for instant static views, and a FastAPI service on Railway that serves live LightGBM and CatBoost inference for customized inputs. The app has six pages: a Dashboard with model KPIs, a Cost Estimator that runs the two-stage prediction flow, a Forecast Explorer showing specialty trends with P10\u2013P90 confidence bands, a Model Comparison page with feature-importance plots and methodology accordions, a Data Explorer with state-level summaries, and an About page documenting the V2 leaderboard and the Phase 8 forecast-track closeout. The Cost Estimator falls back to nearest-neighbor lookups when a user\u2019s exact (specialty, HCPCS, state, POS) tuple is missing from the pre-computed table, which covers 32,818 groups from the 103M-row gold dataset. Responsive mobile navigation and drawer-based forms make the tool usable on a phone."));

// ---------- 12. Future Possibilities ----------
children.push(H1("12. Future Possibilities"));
children.push(KVBullet("Quarterly data ingestion (highest-leverage lever).",
  "The annual-resolution signal ceiling at R\u00B2 \u2248 0.885 is empirically confirmed. Going from 11 annual to 44 quarterly points per group would unlock CNN/TCN/PatchTST architectures, enable meaningful Chronos fine-tuning, and expose seasonal structure (open-enrollment spikes, COVID-era shocks). Target: R\u00B2 0.91\u20130.93. Estimated 2\u20134 weeks of data engineering."));
children.push(KVBullet("Integrate the Medicare Physician Fee Schedule (MPFS).",
  "Join the actual Work/PE/MP RVUs and the GPCI adjustment factors from CMS\u2019s RVU files. The model then predicts deviations from the fee schedule rather than the fee schedule itself, which is a publishable contribution."));
children.push(KVBullet("Replace synthetic Stage 2 with real MCBS LDS data.",
  "A $600-per-module purchase from ResDAC would unlock real beneficiary-linked OOP and likely validate (or invalidate) the CatBoost Monotonic V2 hypothesis that failed on synthetic data."));
children.push(KVBullet("County-level geography.",
  "Merge in the Area Health Resources File (AHRF) and American Community Survey (ACS) to replace 56 state codes with ~3,200 county-level supply/demand and economic features."));
children.push(KVBullet("Quantile stacker variant.",
  "A 20-line edit to the V2_12 notebook would train three LightGBM boosters at alpha \u2208 {0.1, 0.5, 0.9} to give the forecast track genuine P10/P50/P90 bounds instead of the current point-only forecast."));
children.push(KVBullet("Hospital facility fees and total-episode cost.",
  "Integrating the CMS Outpatient PUF (APC \u2192 HCPCS crosswalk) would let the UI display \u201CPhysician fee + facility fee = total episode cost,\u201D which is the number patients actually care about."));
children.push(KVBullet("Silver-layer specialty canonicalization.",
  "Collapse CMS\u2019s year-to-year specialty-name variants with a Levenshtein audit and parenthetical stripping, then retrain. Fixes the LSTM split-index bug affecting Cardiology, Colorectal Surgery, and Oral Surgery forecasts."));

// ---------- 13. Conclusion ----------
children.push(H1("13. Conclusion"));
children.push(P("AllowanceMap demonstrates that a small team, with CMS public data and ~$100 of cloud compute, can build a two-stage Medicare cost estimator that beats any free public tool on predictive accuracy, uncertainty quantification, and temporal forecasting \u2014 without access to claim-level data or commercial databases. The Stage 1 LightGBM predicts allowed amounts with R\u00B2 = 0.943 while consuming no submitted-charge input, the Stage 2 XGBoost quantile regressor delivers calibrated P10/P50/P90 OOP estimates, and the LightGBM stacker reaches the empirical ceiling for annual-resolution forecasting at R\u00B2 = 0.885. The project also produced hard-won methodological lessons: full data beats architecture, monotonicity constraints fight synthetic distributions, teacher-forced evaluation inflates sequence-model metrics, and ensemble diversity \u2014 not model cleverness \u2014 is what exceeded the annual-resolution signal ceiling. The remaining improvement path is clear: richer data (quarterly CMS aggregates, real MCBS LDS, county-level supply-demand features) not richer algorithms. Every deliverable \u2014 pipeline, models, API, and UI \u2014 is live and public, and the documented negative results are at least as useful for future work as the leaderboard numbers."));

// ---------- Appendix ----------
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H1("Appendix"));

children.push(H2("A1. Stage 1 V2 Feature Importance (LightGBM, by gain %)"));
children.push(mkTable(
  [4000, 2000, 3360],
  ["Feature", "Gain %", "Role"],
  [
    ["Avg_Sbmtd_Chrg",             {text:"61.8", align:AlignmentType.RIGHT}, "Submitted charge (dropped in no-charge variant)"],
    ["HCPCS_Cd_idx",                {text:"20.2", align:AlignmentType.RIGHT}, "Procedure identity"],
    ["hcpcs_bucket",                {text:"6.4",  align:AlignmentType.RIGHT}, "Coarse clinical category"],
    ["Rndrng_Prvdr_Type_idx",       {text:"4.4",  align:AlignmentType.RIGHT}, "Specialty"],
    ["srvcs_per_bene",              {text:"3.2",  align:AlignmentType.RIGHT}, "Utilization intensity"],
    ["Other features (combined)",   {text:"4.0",  align:AlignmentType.RIGHT}, "State, POS, risk, log volumes"],
  ]
));

children.push(H2("A2. Forecast Stacker (V2_12) Feature Importance"));
children.push(mkTable(
  [3600, 1800, 3960],
  ["Feature", "Gain %", "Role"],
  [
    ["lstm_pred",              {text:"70.48", align:AlignmentType.RIGHT}, "LSTM autoregressive prediction"],
    ["last_history_value",     {text:"15.97", align:AlignmentType.RIGHT}, "Persistence anchor (2021 value)"],
    ["chronos_pred",           {text:"3.63",  align:AlignmentType.RIGHT}, "Chronos-Bolt cpi_cf_deflated median"],
    ["history_mean",           {text:"2.67",  align:AlignmentType.RIGHT}, "Mean over available history"],
    ["provider_type + state",  {text:"3.07",  align:AlignmentType.RIGHT}, "Categorical conditioning"],
    ["history_cv + trend",     {text:"2.80",  align:AlignmentType.RIGHT}, "Volatility/slope conditioning"],
    ["n_history_years + other",{text:"1.38",  align:AlignmentType.RIGHT}, "Context length and flags"],
  ]
));

children.push(H2("A3. Compute Spend Summary"));
children.push(mkTable(
  [4000, 2200, 3160],
  ["Notebook / Phase", "Duration", "Colab Compute Units"],
  [
    ["V2_01 Stage 1 full",                     "~4 hrs",   "~45"],
    ["V2_02 Stage 1 no-charge",                "~3 hrs",   "~34"],
    ["V2_03 Stage 1 5-fold ensemble",          "~13 hrs",  "~146"],
    ["V2_04 CatBoost Monotonic OOP",           "~2 hrs",   "~22"],
    ["V2_05 Zero-inflated OOP",                "~1.5 hrs", "~17"],
    ["V2_06 TFT univariate",                   "~3 hrs",   "~34"],
    ["V2_09\u2013V2_11 Foundation models",     "~1.5 hrs", "~15"],
    ["V2_12 LightGBM stacker (self-contained)", "~20 min", "~4"],
    ["V2_13 Multivariate TFT",                 "~60 min",  "~12"],
    [{text:"Total (Phases 7 + 8)", bold:true}, {text:"~31 hrs", bold:true}, {text:"~346 CU", bold:true}],
  ]
));

children.push(H2("A4. Key Repository Assets"));
children.push(Bullet("notebooks/01_bronze_ingest_local.py \u2192 05_lstm_sequences_local.py \u2014 Medallion pipeline."));
children.push(Bullet("modeling/train_{glm,rf,xgb,catboost,lgbm,lstm,oop}_local.py \u2014 all model families."));
children.push(Bullet("colab/V2_01 \u2026 V2_13 \u2014 full-data Colab notebooks."));
children.push(Bullet("api/ \u2014 FastAPI service deployed to Railway."));
children.push(Bullet("web/ \u2014 Next.js 16 UI deployed to Vercel."));
children.push(Bullet("PROGRESS.md, MODELING.md, DATA_SOURCES.md \u2014 long-form project documentation."));

// ---------- Document setup ----------
const doc = new Document({
  creator: "Raj Vedire",
  title: "AllowanceMap Project Report",
  description: "Medicare allowed-amount and patient OOP prediction, and 2024-2026 forecasting.",
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Calibri", color: "1F3864" },
        paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Calibri", color: "2E75B6" },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: "Calibri", color: "2E75B6" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 540, hanging: 260 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    headers: {
      default: new Header({ children: [new Paragraph({
        alignment: AlignmentType.RIGHT,
        children: [new TextRun({
          text: "AllowanceMap \u2014 Medicare Cost Prediction",
          size: 18, color: "7F7F7F",
        })],
      })] }),
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [
          new TextRun({ text: "Page ", size: 18, color: "7F7F7F" }),
          new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "7F7F7F" }),
          new TextRun({ text: " of ", size: 18, color: "7F7F7F" }),
          new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18, color: "7F7F7F" }),
        ],
      })] }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  const outPath = process.argv[2] || "AllowanceMap_Project_Report.docx";
  fs.writeFileSync(outPath, buf);
  console.log("Wrote", outPath, buf.length, "bytes");
});
