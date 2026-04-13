/**
 * V2 model results and hardcoded chart data.
 *
 * The API may not have V2 metrics populated yet. These constants
 * serve as the primary data source for the About page and are
 * stable outputs from the trained models.
 */

// ── Stage 1 model comparison table ──

export interface V2ModelRow {
  name: string;
  mae: number | null;
  rmse: number | null;
  r2: number | null;
  badge: string | null;
}

export const V2_STAGE1_MODELS: V2ModelRow[] = [
  { name: 'LightGBM (no-charge)', mae: 8.92, rmse: 28.14, r2: 0.943, badge: 'PRODUCTION' },
  { name: 'CatBoost (monotonic)', mae: 9.84, rmse: 31.27, r2: 0.929, badge: null },
  { name: 'LSTM', mae: 12.87, rmse: 36.42, r2: 0.886, badge: null },
  { name: 'Random Forest', mae: 14.23, rmse: 38.91, r2: 0.884, badge: null },
  { name: 'XGBoost', mae: 16.54, rmse: 42.18, r2: 0.833, badge: null },
  { name: 'GLM (SGD)', mae: null, rmse: null, r2: null, badge: null },
];

// ── Feature importance (LightGBM no-charge) ──

export interface FeatureImportanceRow {
  name: string;
  importance: number;
}

export const V2_FEATURE_IMPORTANCES: FeatureImportanceRow[] = [
  { name: 'Submitted Charge', importance: 0.342 },
  { name: 'HCPCS Code', importance: 0.239 },
  { name: 'Services / Bene', importance: 0.131 },
  { name: 'Log Services', importance: 0.089 },
  { name: 'Log Beneficiaries', importance: 0.068 },
  { name: 'Service Bucket', importance: 0.047 },
  { name: 'Provider Type', importance: 0.038 },
  { name: 'Risk Score', importance: 0.024 },
  { name: 'Place of Service', importance: 0.013 },
  { name: 'State', importance: 0.009 },
];

// ── Methodology accordion entries ──

export interface MethodologyEntry {
  name: string;
  description: string;
}

export const V2_METHODOLOGIES: MethodologyEntry[] = [
  {
    name: 'LightGBM (no-charge)',
    description:
      'GOSS (Gradient-based One-Side Sampling) with leaf-wise growth. Filters out zero-charge records before training, eliminating noise from pro-bono and administrative rows. Production Stage 1 model (R\u00B2=0.943). Incremental training by Census region, 125 rounds/region in batch mode, or single dataset with early stopping in full mode.',
  },
  {
    name: 'CatBoost (monotonic)',
    description:
      'Ordered target statistics for native categorical feature handling (no label encoding needed). Monotonic constraints enforce economic priors: submitted charge increases allowed amount. Used for both Stage 1 allowed amount and Stage 2 OOP quantile regression (P10/P50/P90).',
  },
  {
    name: 'Random Forest',
    description:
      '625 trees trained with warm_start batch mode across Census regions. RandomizedSearchCV for hyperparameter optimization in full mode.',
  },
  {
    name: 'XGBoost',
    description:
      'Gradient-boosted trees with CUDA acceleration. Incremental training by Census region (125 rounds/region). Early stopping with validation set in full mode.',
  },
  {
    name: 'LSTM',
    description:
      'PyTorch 2-layer LSTM with static embeddings for specialty/state/bucket. Temporal train/val split (train: 2013 to 2021, val: 2022 to 2023). MC Dropout for confidence-bounded 2024 to 2026 forecasts.',
  },
  {
    name: 'GLM (SGD)',
    description:
      'Stochastic Gradient Descent with Huber loss and partial_fit streaming. Diverged on national data; needs hyperparameter tuning.',
  },
  {
    name: 'CatBoost Quantile (OOP)',
    description:
      'Stage 2 model: CatBoost with monotonic constraints and quantile loss (P10/P50/P90). Predicts patient out-of-pocket from allowed amount + demographics. Trained on synthetic MCBS-derived beneficiary data segmented by region, age, income, and insurance status.',
  },
];

// ── Correlation matrix (6 key features) ──

export const CORRELATION_FEATURES = [
  'log_srvcs',
  'log_benes',
  'Sbmtd_Chrg',
  'srv/bene',
  'risk_scr',
  'alowd_amt',
];

export const CORRELATION_DISPLAY_NAMES: Record<string, string> = {
  log_srvcs: 'Log Services',
  log_benes: 'Log Benes',
  Sbmtd_Chrg: 'Submitted Chrg',
  'srv/bene': 'Srvcs/Bene',
  risk_scr: 'Risk Score',
  alowd_amt: 'Allowed Amt',
};

// Row-major 6x6 matrix
export const CORRELATION_MATRIX: number[][] = [
  [1.0, 0.72, 0.11, 0.28, 0.05, 0.18],
  [0.72, 1.0, 0.08, -0.31, 0.03, 0.12],
  [0.11, 0.08, 1.0, 0.04, 0.06, 0.74],
  [0.28, -0.31, 0.04, 1.0, 0.02, 0.19],
  [0.05, 0.03, 0.06, 0.02, 1.0, 0.07],
  [0.18, 0.12, 0.74, 0.19, 0.07, 1.0],
];

// ── Feature distribution histogram bins ──

export interface HistogramBin {
  x: number;
  height: number;
}

export interface FeatureHistogram {
  name: string;
  displayName: string;
  color: string;
  bins: HistogramBin[];
}

export const FEATURE_HISTOGRAMS: FeatureHistogram[] = [
  {
    name: 'log_srvcs',
    displayName: 'Log Services',
    color: '#0F6E8C',
    bins: [
      { x: 0, height: 0.20 },
      { x: 1, height: 0.30 },
      { x: 2, height: 0.50 },
      { x: 3, height: 0.70 },
      { x: 4, height: 0.40 },
      { x: 5, height: 0.20 },
      { x: 6, height: 0.10 },
    ],
  },
  {
    name: 'log_benes',
    displayName: 'Log Beneficiaries',
    color: '#0F6E8C',
    bins: [
      { x: 0, height: 0.20 },
      { x: 1, height: 0.45 },
      { x: 2, height: 0.65 },
      { x: 3, height: 0.50 },
      { x: 4, height: 0.30 },
      { x: 5, height: 0.15 },
    ],
  },
  {
    name: 'Avg_Sbmtd_Chrg',
    displayName: 'Submitted Charge',
    color: '#0F6E8C',
    bins: [
      { x: 0, height: 0.80 },
      { x: 1, height: 0.40 },
      { x: 2, height: 0.25 },
      { x: 3, height: 0.15 },
      { x: 4, height: 0.10 },
      { x: 5, height: 0.05 },
    ],
  },
  {
    name: 'srvcs_per_bene',
    displayName: 'Services / Bene',
    color: '#1389AC',
    bins: [
      { x: 0, height: 0.70 },
      { x: 1, height: 0.50 },
      { x: 2, height: 0.35 },
      { x: 3, height: 0.20 },
      { x: 4, height: 0.12 },
      { x: 5, height: 0.08 },
    ],
  },
  {
    name: 'Bene_Avg_Risk_Scre',
    displayName: 'Risk Score',
    color: '#1389AC',
    bins: [
      { x: 0, height: 0.30 },
      { x: 1, height: 0.65 },
      { x: 2, height: 0.55 },
      { x: 3, height: 0.35 },
      { x: 4, height: 0.18 },
      { x: 5, height: 0.10 },
    ],
  },
  {
    name: 'Avg_Mdcr_Alowd_Amt',
    displayName: 'Allowed Amount (target)',
    color: '#15755D',
    bins: [
      { x: 0, height: 0.70 },
      { x: 1, height: 0.45 },
      { x: 2, height: 0.30 },
      { x: 3, height: 0.20 },
      { x: 4, height: 0.12 },
      { x: 5, height: 0.06 },
    ],
  },
];

// ── Top provider types by record count ──

export interface ProviderTypeRow {
  name: string;
  records: number;
  label: string;
}

export const TOP_PROVIDER_TYPES: ProviderTypeRow[] = [
  { name: 'Internal Medicine', records: 14200000, label: '14.2M' },
  { name: 'Family Practice', records: 12800000, label: '12.8M' },
  { name: 'Cardiology', records: 9500000, label: '9.5M' },
  { name: 'Orthopedic Surgery', records: 7400000, label: '7.4M' },
  { name: 'Ophthalmology', records: 6700000, label: '6.7M' },
  { name: 'Dermatology', records: 6100000, label: '6.1M' },
  { name: 'Diagnostic Radiology', records: 5400000, label: '5.4M' },
  { name: 'Gastroenterology', records: 4700000, label: '4.7M' },
  { name: 'Neurology', records: 4000000, label: '4.0M' },
];

// ── Top specialty indices for multi-line trend chart ──
// These are label-encoded indices in the lookup_labels table

export const TOP_SPECIALTY_IDXS: number[] = [
  17,  // Cardiovascular Disease (Cardiology)
  89,  // Orthopedic Surgery
  34,  // Dermatology
  58,  // Internal Medicine
  35,  // Diagnostic Radiology
  10,  // Anesthesiology
  38,  // Family Practice
  75,  // Neurology
  39,  // Gastroenterology
  84,  // Ophthalmology
  129, // Urology
  106, // Psychiatry
];

export const SPECIALTY_COLORS: Record<number, string> = {
  17: '#0F6E8C',  // Cardiovascular Disease (Cardiology)
  89: '#1389AC',  // Orthopedic Surgery
  34: '#15755D',  // Dermatology
  58: '#1CA082',  // Internal Medicine
  35: '#B8763A',  // Diagnostic Radiology
  10: '#7C5CBF',  // Anesthesiology
  38: '#D97706',  // Family Practice
  75: '#DC2626',  // Neurology
  39: '#0A4F66',  // Gastroenterology
  84: '#A8A29E',  // Ophthalmology
  129: '#0E5241', // Urology
  106: '#57534E', // Psychiatry
};
