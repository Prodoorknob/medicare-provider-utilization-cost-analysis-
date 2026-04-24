// Client-side helpers for fetching investigation briefs synced to
// web/public/data/investigations/ by scripts/sync-briefs.mjs.

import { useEffect, useState, useCallback } from 'react';

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

// --- NPI redaction ---------------------------------------------------------
// NPIs are public identifiers via NPPES, but naming them in a live demo or
// public deploy can feel invasive. The redaction helpers mask the middle
// digits so list/detail cross-reference still works but no full NPI appears
// on screen or in shared screenshots. Format: 1033****74 (first 4 + last 2).

const NPI_RE = /\b(\d{4})(\d{4})(\d{2})\b/g;

export function maskNpi(npi: string | null | undefined): string {
  if (!npi) return '';
  const digits = String(npi).trim();
  if (!/^\d{10}$/.test(digits)) return digits;
  return `${digits.slice(0, 4)}****${digits.slice(8)}`;
}

/**
 * Mask every 10-digit NPI-shaped token inside an arbitrary string
 * (e.g. executive summary text that names the provider inline).
 */
export function maskNpisInText(text: string): string {
  return text.replace(NPI_RE, (_m, p1: string, _p2: string, p3: string) => `${p1}****${p3}`);
}

const REDACT_KEY = 'inv:redactNpis';

/** Raw read of the redaction pref from localStorage. SSR-safe. */
export function getRedactPref(): boolean {
  if (typeof window === 'undefined') return false;
  // Allow an env-baked default for public deploys: NEXT_PUBLIC_REDACT_NPIS=1
  if (process.env.NEXT_PUBLIC_REDACT_NPIS === '1') return true;
  try {
    return window.localStorage.getItem(REDACT_KEY) === '1';
  } catch {
    return false;
  }
}

export function setRedactPref(value: boolean): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(REDACT_KEY, value ? '1' : '0');
    // Notify other tabs / components in the same tab
    window.dispatchEvent(new CustomEvent('inv:redactChanged', { detail: value }));
  } catch {}
}

/**
 * Display helper: returns the NPI as-is or masked, based on the toggle.
 * Keep pure so server-rendered paths can call it too.
 */
export function displayNpi(npi: string, redact: boolean): string {
  return redact ? maskNpi(npi) : npi;
}

export function displayText(text: string, redact: boolean): string {
  return redact ? maskNpisInText(text) : text;
}

/**
 * Reactive redaction state. Reads localStorage on mount, subscribes to
 * cross-component change events, exposes a setter that persists.
 */
export function useRedactNpis(): [boolean, (v: boolean) => void] {
  const [redact, setRedact] = useState(false);

  useEffect(() => {
    setRedact(getRedactPref());
    const onChange = (e: Event) => {
      const detail = (e as CustomEvent<boolean>).detail;
      if (typeof detail === 'boolean') setRedact(detail);
      else setRedact(getRedactPref());
    };
    window.addEventListener('inv:redactChanged', onChange);
    window.addEventListener('storage', onChange);
    return () => {
      window.removeEventListener('inv:redactChanged', onChange);
      window.removeEventListener('storage', onChange);
    };
  }, []);

  const update = useCallback((v: boolean) => {
    setRedact(v);
    setRedactPref(v);
  }, []);

  return [redact, update];
}

export const REDACT_LOCKED = process.env.NEXT_PUBLIC_REDACT_NPIS === '1';

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
