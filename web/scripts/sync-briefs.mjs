#!/usr/bin/env node
/**
 * sync-briefs.mjs -- copy anomaly investigation briefs into web/public/data.
 *
 * Reads local_pipeline/anomaly/briefs/{NPI}_{YEAR}.{md,json} plus summary.json
 * and writes them under web/public/data/investigations/ so the Next.js app can
 * fetch them as static assets (works in dev + on Vercel).
 *
 * We also generate index.json, a lightweight list of all briefs with just the
 * fields needed for the review table (npi, year, specialty, state,
 * risk_classification, risk_score, rule_summary, generated_at).
 *
 * Run from web/ or repo root:
 *   node web/scripts/sync-briefs.mjs
 *   npm --prefix web run sync-briefs
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const BRIEFS_SRC = path.join(REPO_ROOT, 'local_pipeline', 'anomaly', 'briefs');
const BRIEFS_DST = path.join(REPO_ROOT, 'web', 'public', 'data', 'investigations');

// --- NPI redaction helpers (mirror web/src/lib/investigations.ts) ----------
// When --mask-npis is set, rewrite every NPI-shaped token in the synced JSONs
// so deployed builds never expose real identifiers, even via devtools.
const NPI_RE = /\b(\d{4})(\d{4})(\d{2})\b/g;
const maskNpi = (npi) => {
  const d = String(npi || '').trim();
  if (!/^\d{10}$/.test(d)) return d;
  return `${d.slice(0, 4)}****${d.slice(8)}`;
};
const maskText = (t) =>
  typeof t === 'string' ? t.replace(NPI_RE, (_, p1, _p2, p3) => `${p1}****${p3}`) : t;

function deepMask(value) {
  if (value == null) return value;
  if (typeof value === 'string') return maskText(value);
  if (Array.isArray(value)) return value.map(deepMask);
  if (typeof value === 'object') {
    const out = {};
    for (const [k, v] of Object.entries(value)) out[k] = deepMask(v);
    return out;
  }
  return value;
}

const MASK = process.argv.includes('--mask-npis')
          || process.env.MASK_NPIS === '1';

function extractRuleSummary(brief) {
  // Parse the rule_check_results markdown block into structured flags. Claude
  // emits two slightly different bullet formats; we accept both:
  //   "- **RULE_ID:** TRIGGERED - evidence"
  //   "- **RULE_ID** - TRIGGERED (severity): evidence"
  const text = brief.rule_check_results ?? '';
  const statusRe = /(TRIGGERED|NOT[\s_]+TRIGGERED|NOT[\s_]+EVALUABLE)/i;
  const rules = [];
  for (const line of text.split('\n')) {
    const idMatch = line.match(/^\s*-\s+\*\*([A-Z_]+)(?::)?\*\*/);
    if (!idMatch) continue;
    const id = idMatch[1];
    const statusMatch = line.match(statusRe);
    if (!statusMatch) continue;
    const status = statusMatch[1].toUpperCase().replace(/\s+/g, '_');
    const rest = line.slice(line.indexOf(statusMatch[0]) + statusMatch[0].length)
                     .replace(/^[\s\-\u2014:()]+/, '').trim();
    rules.push({ rule_id: id, status, evidence: rest });
  }
  return {
    triggered:     rules.filter(r => r.status === 'TRIGGERED').length,
    not_triggered: rules.filter(r => r.status === 'NOT_TRIGGERED').length,
    not_evaluable: rules.filter(r => r.status === 'NOT_EVALUABLE').length,
    rules,
  };
}

function main() {
  if (!fs.existsSync(BRIEFS_SRC)) {
    console.error(`[sync-briefs] source not found: ${BRIEFS_SRC}`);
    process.exit(1);
  }
  fs.mkdirSync(BRIEFS_DST, { recursive: true });

  const files = fs.readdirSync(BRIEFS_SRC).filter(f => f.endsWith('.json') && f !== 'summary.json');
  const index = [];

  for (const f of files) {
    const srcJsonPath = path.join(BRIEFS_SRC, f);
    const raw = fs.readFileSync(srcJsonPath, 'utf8');
    let brief;
    try {
      brief = JSON.parse(raw);
    } catch (e) {
      console.warn(`[sync-briefs] skipping ${f}: invalid JSON (${e.message})`);
      continue;
    }

    // Strip the full_markdown blob from the copied file -- the Next.js viewer
    // renders from structured fields directly. Keep usage metadata for debug.
    let copy = { ...brief };
    if (copy.evidence_summary && copy.evidence_summary.full_markdown) {
      const { full_markdown, ...rest } = copy.evidence_summary;
      copy.evidence_summary = rest;
    }
    if (MASK) {
      copy = deepMask(copy);
      copy.npi = maskNpi(brief.npi); // top-level exact mask
    }
    const dstJsonPath = path.join(BRIEFS_DST, f);
    fs.writeFileSync(dstJsonPath, JSON.stringify(copy, null, 2));

    const ruleSummary = extractRuleSummary(brief);
    index.push({
      npi:                 MASK ? maskNpi(brief.npi) : brief.npi,
      year:                brief.year,
      specialty:           brief.specialty,
      state:               brief.state,
      risk_classification: brief.risk_classification,
      risk_score:          brief.risk_score,
      generated_at:        brief.generated_at,
      model_version:       brief.model_version,
      rule_summary:        MASK ? deepMask(ruleSummary) : ruleSummary,
    });
  }

  // Sort by risk score desc, then npi
  index.sort((a, b) => (b.risk_score - a.risk_score) || a.npi.localeCompare(b.npi));

  fs.writeFileSync(
    path.join(BRIEFS_DST, 'index.json'),
    JSON.stringify({ count: index.length, briefs: index }, null, 2),
  );

  // Copy summary.json if it exists (mask NPIs if requested)
  const summarySrc = path.join(BRIEFS_SRC, 'summary.json');
  if (fs.existsSync(summarySrc)) {
    if (MASK) {
      const s = JSON.parse(fs.readFileSync(summarySrc, 'utf8'));
      fs.writeFileSync(path.join(BRIEFS_DST, 'summary.json'), JSON.stringify(deepMask(s), null, 2));
    } else {
      fs.copyFileSync(summarySrc, path.join(BRIEFS_DST, 'summary.json'));
    }
  }

  console.log(`[sync-briefs]${MASK ? ' [MASKED]' : ''} wrote ${index.length} briefs to ${BRIEFS_DST}`);
}

main();
