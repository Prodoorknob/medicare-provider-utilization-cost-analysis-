import type {
  LookupLabel, FullPredictionResponse,
  LstmForecast, SpecialtyYearlyAvg, StateSummary, ModelMetric, FeatureImportance,
} from './types';

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`API error ${res.status}: ${detail}`);
  }
  return res.json();
}

// -- Reference data --

export async function getLabels(category: string): Promise<LookupLabel[]> {
  return apiFetch<LookupLabel[]>(`/labels?category=${encodeURIComponent(category)}`);
}

export async function getStateSummary(): Promise<StateSummary[]> {
  return apiFetch<StateSummary[]>('/state-summary');
}

export async function getModelMetrics(): Promise<ModelMetric[]> {
  return apiFetch<ModelMetric[]>('/model-metrics');
}

export async function getFeatureImportances(): Promise<FeatureImportance[]> {
  return apiFetch<FeatureImportance[]>('/feature-importances');
}

export async function getSpecialtyHistory(specialtyIdx: number): Promise<SpecialtyYearlyAvg[]> {
  return apiFetch<SpecialtyYearlyAvg[]>(`/specialty-history?specialty_idx=${specialtyIdx}`);
}

// -- Forecasts --

export async function getForecasts(
  specialtyIdx: number,
  stateIdx?: number,
  hcpcsBucket?: number,
): Promise<LstmForecast[]> {
  const params = new URLSearchParams({ specialty_idx: String(specialtyIdx) });
  if (stateIdx !== undefined) params.set('state_idx', String(stateIdx));
  if (hcpcsBucket !== undefined) params.set('hcpcs_bucket', String(hcpcsBucket));
  return apiFetch<LstmForecast[]>(`/forecast?${params}`);
}

// -- Predictions (real-time model inference) --

export async function getFullPrediction(params: {
  provider_type: string;
  state: string;
  hcpcs_bucket: number;
  place_of_service: number;
  age: number;
  sex: number;
  income: number;
  chronic_count: number;
  dual_eligible: number;
  has_supplemental: number;
}): Promise<FullPredictionResponse> {
  return apiFetch<FullPredictionResponse>('/predict/full', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}
