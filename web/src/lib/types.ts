export interface LookupLabel {
  id: number;
  category: string;
  idx: number;
  label: string;
}

export interface Stage1Estimate {
  specialty_idx: number;
  hcpcs_bucket: number;
  state_idx: number;
  place_of_service: number;
  n_records: number;
  mean_allowed: number;
  median_allowed: number;
  p10_allowed: number;
  p90_allowed: number;
  mean_charge: number | null;
  mean_risk_score: number | null;
}

export interface Stage2Estimate {
  specialty_idx: number;
  hcpcs_bucket: number;
  census_region: number;
  dual_eligible: number;
  has_supplemental: number;
  age_group: number;
  income_bracket: number;
  n_records: number;
  oop_p10: number;
  oop_p50: number;
  oop_p90: number;
  mean_allowed: number | null;
}

export interface LstmForecast {
  specialty_idx: number;
  hcpcs_bucket: number;
  state_idx: number;
  forecast_year: number;
  forecast_mean: number;
  forecast_std: number | null;
  forecast_p10: number;
  forecast_p50: number;
  forecast_p90: number;
  last_known_year: number | null;
  last_known_value: number | null;
  n_history_years: number | null;
}

export interface StateSummary {
  state_abbrev: string;
  state_idx: number;
  mean_allowed: number;
  median_allowed: number;
  n_records: number;
}

export interface ModelMetric {
  model_name: string;
  stage: number;
  metric_name: string;
  metric_value: number;
}

export interface FeatureImportance {
  model_name: string;
  feature_name: string;
  importance: number;
  rank: number | null;
}
