'use client';

import { useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Chip from '@mui/material/Chip';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import TableSortLabel from '@mui/material/TableSortLabel';
import TextField from '@mui/material/TextField';
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import Stack from '@mui/material/Stack';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import FormControlLabel from '@mui/material/FormControlLabel';
import Switch from '@mui/material/Switch';
import Tooltip from '@mui/material/Tooltip';
import ShieldOutlinedIcon from '@mui/icons-material/ShieldOutlined';
import ReportProblemOutlinedIcon from '@mui/icons-material/ReportProblemOutlined';
import HelpOutlineOutlinedIcon from '@mui/icons-material/HelpOutlineOutlined';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutlineOutlined';
import {
  fetchIndex,
  RISK_COLORS,
  displayNpi,
  useRedactNpis,
  REDACT_LOCKED,
  type BriefIndex,
  type BriefIndexEntry,
  type RiskClassification,
} from '@/lib/investigations';

type SortKey = 'risk_score' | 'npi' | 'year' | 'specialty' | 'state' | 'triggered';
type Order = 'asc' | 'desc';

const CLASSIFICATIONS: (RiskClassification | 'ALL')[] = ['ALL', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'];

export default function InvestigationsPage() {
  const [data, setData] = useState<BriefIndex | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [filter, setFilter] = useState<RiskClassification | 'ALL'>('ALL');
  const [query, setQuery] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('risk_score');
  const [order, setOrder] = useState<Order>('desc');
  const [redact, setRedact] = useRedactNpis();

  useEffect(() => {
    fetchIndex()
      .then((d) => setData(d))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const rows = useMemo(() => {
    if (!data) return [];
    let rs = data.briefs.slice();
    if (filter !== 'ALL') rs = rs.filter((b) => b.risk_classification === filter);
    if (query.trim()) {
      const q = query.trim().toLowerCase();
      rs = rs.filter(
        (b) =>
          b.npi.includes(q) ||
          b.specialty.toLowerCase().includes(q) ||
          b.state.toLowerCase().includes(q),
      );
    }
    const mul = order === 'asc' ? 1 : -1;
    rs.sort((a, b) => {
      switch (sortKey) {
        case 'risk_score': return (a.risk_score - b.risk_score) * mul;
        case 'npi':        return a.npi.localeCompare(b.npi) * mul;
        case 'year':       return (a.year - b.year) * mul;
        case 'specialty':  return a.specialty.localeCompare(b.specialty) * mul;
        case 'state':      return a.state.localeCompare(b.state) * mul;
        case 'triggered':  return (a.rule_summary.triggered - b.rule_summary.triggered) * mul;
      }
    });
    return rs;
  }, [data, filter, query, sortKey, order]);

  const totals = useMemo(() => {
    if (!data) return { total: 0, critical: 0, high: 0, medium: 0, low: 0, triggered: 0 };
    const t = { total: data.briefs.length, critical: 0, high: 0, medium: 0, low: 0, triggered: 0 };
    for (const b of data.briefs) {
      t.triggered += b.rule_summary.triggered;
      if (b.risk_classification === 'CRITICAL') t.critical++;
      else if (b.risk_classification === 'HIGH') t.high++;
      else if (b.risk_classification === 'MEDIUM') t.medium++;
      else if (b.risk_classification === 'LOW') t.low++;
    }
    return t;
  }, [data]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setOrder(order === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setOrder(key === 'npi' || key === 'specialty' || key === 'state' ? 'asc' : 'desc'); }
  };

  return (
    <Box sx={{ py: 2 }}>
      <Stack direction="row" spacing={1.5} sx={{ mb: 1, alignItems: 'center' }}>
        <ShieldOutlinedIcon sx={{ color: 'primary.main', fontSize: 32 }} />
        <Typography variant="h4" component="h1" sx={{ flex: 1 }}>Provider Investigations</Typography>
        <Tooltip title={
          REDACT_LOCKED
            ? 'NPI redaction is enforced for this deployment and cannot be disabled'
            : 'Mask NPIs (e.g. 1033****74) for live demos and screenshots. Saved locally.'
        }>
          <FormControlLabel
            control={<Switch size="small" checked={redact} disabled={REDACT_LOCKED} onChange={(e) => setRedact(e.target.checked)} />}
            label="Redact NPIs"
            sx={{ m: 0, '& .MuiFormControlLabel-label': { fontSize: '0.8125rem', color: 'text.secondary' } }}
          />
        </Tooltip>
      </Stack>
      <Typography color="text.secondary" sx={{ mb: 3, maxWidth: 800 }}>
        Claude-generated investigation briefs for providers that triggered anomaly detection. Each brief
        combines statistical flags, rule-based fraud indicators, and contextual reasoning. Approve, dismiss,
        or escalate to guide detection calibration.
      </Typography>

      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5} sx={{ mb: 3, flexWrap: 'wrap' }}>
        {[
          { label: 'Briefs', value: totals.total, color: 'text.primary' },
          { label: 'Critical', value: totals.critical, color: RISK_COLORS.CRITICAL.border },
          { label: 'High', value: totals.high, color: RISK_COLORS.HIGH.border },
          { label: 'Medium', value: totals.medium, color: RISK_COLORS.MEDIUM.border },
          { label: 'Rules triggered', value: totals.triggered, color: 'primary.main' },
        ].map((s) => (
          <Card key={s.label} sx={{ minWidth: 140, flex: { sm: '0 0 auto' } }}>
            <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
              <Typography variant="overline" color="text.secondary">{s.label}</Typography>
              <Typography variant="h5" sx={{ color: s.color as string, fontWeight: 700 }}>
                {s.value}
              </Typography>
            </CardContent>
          </Card>
        ))}
      </Stack>

      <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} sx={{ mb: 2 }}>
        <ToggleButtonGroup
          exclusive
          size="small"
          value={filter}
          onChange={(_, v) => v && setFilter(v)}
        >
          {CLASSIFICATIONS.map((c) => (
            <ToggleButton key={c} value={c}>{c}</ToggleButton>
          ))}
        </ToggleButtonGroup>
        <TextField
          size="small"
          placeholder={redact ? 'Filter by specialty or state' : 'Filter by NPI, specialty, or state'}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          sx={{ minWidth: 300 }}
        />
      </Stack>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 6 }}>
          <CircularProgress />
        </Box>
      )}
      {error && <Alert severity="error">{error}</Alert>}

      {!loading && !error && data && (
        <Card>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow sx={{ '& th': { fontWeight: 700, bgcolor: 'background.default' } }}>
                  <TableCell>
                    <TableSortLabel active={sortKey === 'risk_score'} direction={order} onClick={() => handleSort('risk_score')}>
                      Risk
                    </TableSortLabel>
                  </TableCell>
                  <TableCell>
                    <TableSortLabel active={sortKey === 'npi'} direction={order} onClick={() => handleSort('npi')}>
                      NPI
                    </TableSortLabel>
                  </TableCell>
                  <TableCell>
                    <TableSortLabel active={sortKey === 'year'} direction={order} onClick={() => handleSort('year')}>
                      Year
                    </TableSortLabel>
                  </TableCell>
                  <TableCell>
                    <TableSortLabel active={sortKey === 'specialty'} direction={order} onClick={() => handleSort('specialty')}>
                      Specialty
                    </TableSortLabel>
                  </TableCell>
                  <TableCell>
                    <TableSortLabel active={sortKey === 'state'} direction={order} onClick={() => handleSort('state')}>
                      State
                    </TableSortLabel>
                  </TableCell>
                  <TableCell align="right">
                    <TableSortLabel active={sortKey === 'triggered'} direction={order} onClick={() => handleSort('triggered')}>
                      Rules
                    </TableSortLabel>
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {rows.map((b) => <InvestigationRow key={`${b.npi}_${b.year}`} b={b} redact={redact} />)}
                {rows.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={6} align="center" sx={{ py: 4, color: 'text.secondary' }}>
                      No briefs match this filter.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </Card>
      )}
    </Box>
  );
}

function InvestigationRow({ b, redact }: { b: BriefIndexEntry; redact: boolean }) {
  const c = RISK_COLORS[b.risk_classification];
  const router = useRouter();
  return (
    <TableRow
      hover
      onClick={() => router.push(`/investigations/${b.npi}_${b.year}`)}
      sx={{ cursor: 'pointer', '& td': { color: 'text.primary' } }}
    >
      <TableCell sx={{ width: 170 }}>
        <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
          <Chip
            size="small"
            label={b.risk_classification}
            sx={{
              bgcolor: c.bg,
              color: c.fg,
              border: `1px solid ${c.border}`,
              fontWeight: 700,
              minWidth: 80,
            }}
          />
          <Typography variant="body2" sx={{ fontWeight: 700 }}>{b.risk_score.toFixed(0)}</Typography>
        </Stack>
      </TableCell>
      <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.85rem' }}>{displayNpi(b.npi, redact)}</TableCell>
      <TableCell>{b.year}</TableCell>
      <TableCell>{b.specialty}</TableCell>
      <TableCell>{b.state}</TableCell>
      <TableCell align="right">
        <Stack direction="row" spacing={0.5} sx={{ justifyContent: 'flex-end' }}>
          {b.rule_summary.triggered > 0 && (
            <Chip size="small" icon={<ReportProblemOutlinedIcon />} label={b.rule_summary.triggered}
              sx={{ bgcolor: '#FEE2E2', color: '#7F1D1D', fontWeight: 600, '& .MuiChip-icon': { color: '#7F1D1D' } }} />
          )}
          {b.rule_summary.not_triggered > 0 && (
            <Chip size="small" icon={<CheckCircleOutlineIcon />} label={b.rule_summary.not_triggered}
              sx={{ bgcolor: '#E8F7F3', color: '#0E5241', fontWeight: 600, '& .MuiChip-icon': { color: '#0E5241' } }} />
          )}
          {b.rule_summary.not_evaluable > 0 && (
            <Chip size="small" icon={<HelpOutlineOutlinedIcon />} label={b.rule_summary.not_evaluable}
              sx={{ bgcolor: '#F5F5F4', color: '#57534E', fontWeight: 600, '& .MuiChip-icon': { color: '#57534E' } }} />
          )}
        </Stack>
      </TableCell>
    </TableRow>
  );
}
