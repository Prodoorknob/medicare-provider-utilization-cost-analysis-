// Client-side helpers for fetching investigation briefs synced to
// web/public/data/investigations/ by scripts/sync-briefs.mjs.

export type RiskClassification = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
export type RuleStatus = 'TRIGGERED' | 'NOT_TRIGGERED' | 'NOT_EVALUABLE';

export interface RuleEntry {
  rule_id: string;
  status: RuleStatus;
  evidence: string;
}

export interface RuleSummary {
  triggered: number;
  not_triggered: number;
  not_evaluable: number;
  rules: RuleEntry[];
}

export interface BriefIndexEntry {
  npi: string;
  year: number;
  specialty: string;
  state: string;
  risk_classification: RiskClassification;
  risk_score: number;
  generated_at: string;
  model_version: string;
  rule_summary: RuleSummary;
}

export interface BriefIndex {
  count: number;
  briefs: BriefIndexEntry[];
}

export interface BriefDetail extends BriefIndexEntry {
  executive_summary: string;
  statistical_findings: string;
  contextual_interpretation: string;
  rule_check_results: string;
  data_limitations: string;
  recommended_actions: string[];
  evidence_summary: { usage?: Record<string, number> };
}

const BASE = '/data/investigations';

export async function fetchIndex(): Promise<BriefIndex> {
  const res = await fetch(`${BASE}/index.json`, { cache: 'no-store' });
  if (!res.ok) throw new Error(`Failed to load brief index (${res.status})`);
  return res.json();
}

export async function fetchBrief(npi: string, year: number): Promise<BriefDetail> {
  const res = await fetch(`${BASE}/${npi}_${year}.json`, { cache: 'no-store' });
  if (!res.ok) throw new Error(`Brief not found: ${npi}_${year}`);
  const data = await res.json();
  // Re-derive rule_summary since the synced per-brief JSON does not include it
  data.rule_summary = extractRuleSummary(data.rule_check_results ?? '');
  return data;
}

export function extractRuleSummary(text: string): RuleSummary {
  const statusRe = /(TRIGGERED|NOT[\s_]+TRIGGERED|NOT[\s_]+EVALUABLE)/i;
  const rules: RuleEntry[] = [];
  for (const line of text.split('\n')) {
    const idMatch = line.match(/^\s*-\s+\*\*([A-Z_]+)(?::)?\*\*/);
    if (!idMatch) continue;
    const id = idMatch[1];
    const sm = line.match(statusRe);
    if (!sm) continue;
    const status = sm[1].toUpperCase().replace(/\s+/g, '_') as RuleStatus;
    const rest = line
      .slice(line.indexOf(sm[0]) + sm[0].length)
      .replace(/^[\s\-\u2014:()]+/, '')
      .trim();
    rules.push({ rule_id: id, status, evidence: rest });
  }
  return {
    triggered:     rules.filter((r) => r.status === 'TRIGGERED').length,
    not_triggered: rules.filter((r) => r.status === 'NOT_TRIGGERED').length,
    not_evaluable: rules.filter((r) => r.status === 'NOT_EVALUABLE').length,
    rules,
  };
}

export const RISK_COLORS: Record<RiskClassification, { bg: string; fg: string; border: string }> = {
  LOW:      { bg: '#F0FAF7', fg: '#0E5241', border: '#1CA082' },
  MEDIUM:   { bg: '#FDF4EA', fg: '#78401C', border: '#B8763A' },
  HIGH:     { bg: '#FEF1ED', fg: '#9A3412', border: '#EA580C' },
  CRITICAL: { bg: '#FEE2E2', fg: '#7F1D1D', border: '#DC2626' },
};

export const RULE_STATUS_COLORS: Record<RuleStatus, { bg: string; fg: string }> = {
  TRIGGERED:     { bg: '#FEE2E2', fg: '#7F1D1D' },
  NOT_TRIGGERED: { bg: '#E8F7F3', fg: '#0E5241' },
  NOT_EVALUABLE: { bg: '#F5F5F4', fg: '#57534E' },
};
