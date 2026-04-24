'use client';

import { use, useEffect, useState } from 'react';
import Link from 'next/link';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
import Divider from '@mui/material/Divider';
import Button from '@mui/material/Button';
import ButtonGroup from '@mui/material/ButtonGroup';
import Alert from '@mui/material/Alert';
import CircularProgress from '@mui/material/CircularProgress';
import Snackbar from '@mui/material/Snackbar';
import TextField from '@mui/material/TextField';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import FlagIcon from '@mui/icons-material/Flag';
import {
  fetchBrief,
  RISK_COLORS,
  RULE_STATUS_COLORS,
  type BriefDetail,
  type RuleStatus,
} from '@/lib/investigations';

type Decision = 'approve' | 'dismiss' | 'escalate';

const DECISION_KEY = (npi: string, year: number) => `inv:decision:${npi}:${year}`;

export default function BriefDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [npiStr, yearStr] = id.split('_');
  const year = Number(yearStr);

  const [brief, setBrief] = useState<BriefDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [decision, setDecision] = useState<Decision | null>(null);
  const [notes, setNotes] = useState('');
  const [toast, setToast] = useState('');

  useEffect(() => {
    if (!npiStr || !Number.isFinite(year)) {
      setError('Invalid brief id');
      setLoading(false);
      return;
    }
    fetchBrief(npiStr, year)
      .then(setBrief)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
    // Load any prior decision from localStorage (captured feedback per spec §8.2)
    try {
      const raw = typeof window !== 'undefined' && window.localStorage.getItem(DECISION_KEY(npiStr, year));
      if (raw) {
        const parsed = JSON.parse(raw);
        setDecision(parsed.decision);
        setNotes(parsed.notes || '');
      }
    } catch {}
  }, [npiStr, year]);

  const saveDecision = (d: Decision) => {
    setDecision(d);
    try {
      window.localStorage.setItem(
        DECISION_KEY(npiStr, year),
        JSON.stringify({ decision: d, notes, reviewed_at: new Date().toISOString() }),
      );
      setToast(`Marked as ${d}`);
    } catch {}
  };

  const saveNotes = () => {
    try {
      const prev = window.localStorage.getItem(DECISION_KEY(npiStr, year));
      const base = prev ? JSON.parse(prev) : { decision: null };
      window.localStorage.setItem(
        DECISION_KEY(npiStr, year),
        JSON.stringify({ ...base, notes, reviewed_at: new Date().toISOString() }),
      );
      setToast('Notes saved');
    } catch {}
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
        <CircularProgress />
      </Box>
    );
  }
  if (error || !brief) {
    return (
      <Box sx={{ py: 3 }}>
        <Button component={Link} href="/investigations" startIcon={<ArrowBackIcon />} sx={{ mb: 2 }}>
          All investigations
        </Button>
        <Alert severity="error">{error || 'Brief not found'}</Alert>
      </Box>
    );
  }

  const risk = RISK_COLORS[brief.risk_classification];

  return (
    <Box sx={{ py: 2, maxWidth: 1100 }}>
      <Button component={Link} href="/investigations" startIcon={<ArrowBackIcon />} sx={{ mb: 2 }}>
        All investigations
      </Button>

      {/* Header */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} sx={{ mb: 2, alignItems: { md: 'center' } }}>
            <Box sx={{ flex: 1 }}>
              <Typography variant="overline" color="text.secondary">
                Investigation Brief · {brief.year}
              </Typography>
              <Typography variant="h4" component="h1" sx={{ fontFamily: 'monospace', fontWeight: 700 }}>
                NPI {brief.npi}
              </Typography>
              <Typography color="text.secondary">
                {brief.specialty} · {brief.state}
              </Typography>
            </Box>
            <Box sx={{
              border: `2px solid ${risk.border}`,
              bgcolor: risk.bg,
              borderRadius: 2,
              px: 2.5, py: 1.5,
              textAlign: 'center',
              minWidth: 180,
            }}>
              <Typography variant="overline" sx={{ color: risk.fg }}>Risk classification</Typography>
              <Typography variant="h5" sx={{ color: risk.fg, fontWeight: 800 }}>
                {brief.risk_classification}
              </Typography>
              <Typography variant="body2" sx={{ color: risk.fg }}>
                Composite score <strong>{brief.risk_score.toFixed(0)}</strong> / 100
              </Typography>
            </Box>
          </Stack>

          <Divider sx={{ my: 2 }} />

          {/* Review actions (spec §8.2) */}
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ alignItems: { sm: 'center' } }}>
            <Typography variant="subtitle2" sx={{ minWidth: 140 }}>Analyst decision</Typography>
            <ButtonGroup size="small" variant="outlined">
              <Button
                startIcon={<CheckCircleIcon />}
                color={decision === 'approve' ? 'success' : 'inherit'}
                variant={decision === 'approve' ? 'contained' : 'outlined'}
                onClick={() => saveDecision('approve')}
              >
                Approve
              </Button>
              <Button
                startIcon={<FlagIcon />}
                color={decision === 'escalate' ? 'warning' : 'inherit'}
                variant={decision === 'escalate' ? 'contained' : 'outlined'}
                onClick={() => saveDecision('escalate')}
              >
                Escalate
              </Button>
              <Button
                startIcon={<CancelIcon />}
                color={decision === 'dismiss' ? 'error' : 'inherit'}
                variant={decision === 'dismiss' ? 'contained' : 'outlined'}
                onClick={() => saveDecision('dismiss')}
              >
                Dismiss
              </Button>
            </ButtonGroup>
            {decision && (
              <Typography variant="body2" color="text.secondary">
                Saved locally · {decision}
              </Typography>
            )}
          </Stack>
        </CardContent>
      </Card>

      {/* Executive summary */}
      <Section title="Executive Summary">
        <Typography>{brief.executive_summary}</Typography>
      </Section>

      {/* Statistical findings */}
      <Section title="Statistical Findings">
        <BulletList text={brief.statistical_findings} />
      </Section>

      {/* Contextual interpretation */}
      <Section title="Contextual Interpretation">
        <Typography>{brief.contextual_interpretation}</Typography>
      </Section>

      {/* Rule checks */}
      <Section title="Rule Check Results">
        <Stack spacing={1.25}>
          {brief.rule_summary.rules.map((r) => (
            <RuleCheckRow key={r.rule_id} id={r.rule_id} status={r.status} evidence={r.evidence} />
          ))}
          {brief.rule_summary.rules.length === 0 && (
            <BulletList text={brief.rule_check_results} />
          )}
        </Stack>
      </Section>

      {/* Data limitations */}
      <Section title="Data Limitations">
        <BulletList text={brief.data_limitations} />
      </Section>

      {/* Recommended actions */}
      <Section title="Recommended Actions">
        <Box component="ol" sx={{ pl: 3, m: 0 }}>
          {brief.recommended_actions.map((a, i) => (
            <Box component="li" key={i} sx={{ mb: 1 }}>
              <InlineMarkdown text={a} />
            </Box>
          ))}
        </Box>
      </Section>

      {/* Notes */}
      <Section title="Analyst Notes">
        <TextField
          multiline
          rows={4}
          fullWidth
          placeholder="Add context, cross-references, or next steps. Saved locally to this browser."
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
        />
        <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
          <Button variant="outlined" size="small" onClick={saveNotes}>Save notes</Button>
        </Stack>
      </Section>

      {/* Footer */}
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 3 }}>
        Generated {new Date(brief.generated_at).toLocaleString()} · {brief.model_version}
      </Typography>

      <Snackbar
        open={!!toast}
        autoHideDuration={2500}
        onClose={() => setToast('')}
        message={toast}
      />
    </Box>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="overline" color="text.secondary">{title}</Typography>
        <Box sx={{ mt: 1 }}>{children}</Box>
      </CardContent>
    </Card>
  );
}

function BulletList({ text }: { text: string }) {
  const items = text
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l.startsWith('-') || l.startsWith('*'))
    .map((l) => l.replace(/^[-*]\s*/, ''));
  if (items.length === 0) return <Typography>{text}</Typography>;
  return (
    <Box component="ul" sx={{ pl: 3, m: 0 }}>
      {items.map((it, i) => (
        <Box component="li" key={i} sx={{ mb: 0.75 }}>
          <InlineMarkdown text={it} />
        </Box>
      ))}
    </Box>
  );
}

function InlineMarkdown({ text }: { text: string }) {
  // Minimal inline bold renderer: **foo** -> <strong>foo</strong>
  const parts: React.ReactNode[] = [];
  const re = /\*\*([^*]+)\*\*/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    parts.push(<strong key={`b${m.index}`}>{m[1]}</strong>);
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return <span>{parts}</span>;
}

function RuleCheckRow({ id, status, evidence }: { id: string; status: RuleStatus; evidence: string }) {
  const c = RULE_STATUS_COLORS[status];
  const label = status.replace('_', ' ');
  return (
    <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'flex-start' }}>
      <Chip
        size="small"
        label={label}
        sx={{
          bgcolor: c.bg,
          color: c.fg,
          fontWeight: 700,
          minWidth: 120,
          flexShrink: 0,
        }}
      />
      <Box sx={{ flex: 1 }}>
        <Typography variant="body2" sx={{ fontWeight: 700, fontFamily: 'monospace' }}>
          {id}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          <InlineMarkdown text={evidence} />
        </Typography>
      </Box>
    </Box>
  );
}
