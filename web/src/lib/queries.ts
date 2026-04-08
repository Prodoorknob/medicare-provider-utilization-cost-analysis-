import { supabase } from './supabase';
import type {
  LookupLabel, Stage1Estimate, Stage2Estimate,
  LstmForecast, StateSummary, ModelMetric, FeatureImportance,
} from './types';

export async function getLabels(category: string): Promise<LookupLabel[]> {
  const { data, error } = await supabase
    .from('lookup_labels')
    .select('*')
    .eq('category', category)
    .order('idx');
  if (error) throw error;
  return data ?? [];
}

export async function getStage1Estimate(
  specialtyIdx: number,
  hcpcsBucket: number,
  stateIdx: number,
  placeOfService: number,
): Promise<Stage1Estimate | null> {
  // Try exact match first
  let { data, error } = await supabase
    .from('stage1_allowed_amounts')
    .select('*')
    .eq('specialty_idx', specialtyIdx)
    .eq('hcpcs_bucket', hcpcsBucket)
    .eq('state_idx', stateIdx)
    .eq('place_of_service', placeOfService)
    .limit(1)
    .maybeSingle();
  if (error) throw error;
  if (data) return data;

  // Fallback: drop POS
  ({ data, error } = await supabase
    .from('stage1_allowed_amounts')
    .select('*')
    .eq('specialty_idx', specialtyIdx)
    .eq('hcpcs_bucket', hcpcsBucket)
    .eq('state_idx', stateIdx)
    .limit(1)
    .maybeSingle());
  if (error) throw error;
  return data;
}

export async function getStage2Estimate(
  specialtyIdx: number,
  hcpcsBucket: number,
  censusRegion: number,
  dualEligible: number,
  hasSupplemental: number,
  ageGroup: number,
  incomeBracket: number,
): Promise<Stage2Estimate | null> {
  // Exact match
  let { data, error } = await supabase
    .from('stage2_oop_estimates')
    .select('*')
    .eq('specialty_idx', specialtyIdx)
    .eq('hcpcs_bucket', hcpcsBucket)
    .eq('census_region', censusRegion)
    .eq('dual_eligible', dualEligible)
    .eq('has_supplemental', hasSupplemental)
    .eq('age_group', ageGroup)
    .eq('income_bracket', incomeBracket)
    .limit(1)
    .maybeSingle();
  if (error) throw error;
  if (data) return data;

  // Fallback: relax age and income
  ({ data, error } = await supabase
    .from('stage2_oop_estimates')
    .select('*')
    .eq('specialty_idx', specialtyIdx)
    .eq('hcpcs_bucket', hcpcsBucket)
    .eq('census_region', censusRegion)
    .eq('dual_eligible', dualEligible)
    .eq('has_supplemental', hasSupplemental)
    .order('n_records', { ascending: false })
    .limit(1)
    .maybeSingle());
  if (error) throw error;
  return data;
}

export async function getForecasts(
  specialtyIdx: number,
  stateIdx?: number,
  hcpcsBucket?: number,
): Promise<LstmForecast[]> {
  let query = supabase
    .from('lstm_forecasts')
    .select('*')
    .eq('specialty_idx', specialtyIdx);

  if (stateIdx !== undefined) query = query.eq('state_idx', stateIdx);
  if (hcpcsBucket !== undefined) query = query.eq('hcpcs_bucket', hcpcsBucket);

  const { data, error } = await query.order('forecast_year');
  if (error) throw error;
  return data ?? [];
}

export async function getStateSummary(): Promise<StateSummary[]> {
  const { data, error } = await supabase
    .from('state_summary')
    .select('*')
    .order('state_abbrev');
  if (error) throw error;
  return data ?? [];
}

export async function getModelMetrics(): Promise<ModelMetric[]> {
  const { data, error } = await supabase
    .from('model_metrics')
    .select('*');
  if (error) throw error;
  return data ?? [];
}

export async function getFeatureImportances(): Promise<FeatureImportance[]> {
  const { data, error } = await supabase
    .from('feature_importances')
    .select('*')
    .order('rank');
  if (error) throw error;
  return data ?? [];
}
