'use client';

import { useState, useEffect, useMemo } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Autocomplete from '@mui/material/Autocomplete';
import TextField from '@mui/material/TextField';
import CircularProgress from '@mui/material/CircularProgress';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Divider from '@mui/material/Divider';
import {
  ResponsiveContainer, ComposedChart, LineChart, Line, Area, Bar, BarChart,
  XAxis, YAxis, Tooltip, CartesianGrid, Legend, ReferenceLine, ErrorBar,
} from 'recharts';
import { getLabels, getForecasts, getSpecialtyHistory } from '@/lib/queries';
import { formatDollars, formatPercent } from '@/lib/formatters';
import { HCPCS_BUCKET_NAMES } from '@/lib/constants';
import type { LookupLabel, LstmForecast, SpecialtyYearlyAvg } from '@/lib/types';
import {
  TOP_SPECIALTY_IDXS, SPECIALTY_COLORS,
} from '@/lib/v2-model-data';

export default function ForecastPage() {
  const [specialties, setSpecialties] = useState<LookupLabel[]>([]);
  const [selectedSpecialty, setSelectedSpecialty] = useState<LookupLabel | null>(null);
  const [forecasts, setForecasts] = useState<LstmForecast[]>([]);
  const [history, setHistory] = useState<SpecialtyYearlyAvg[]>([]);
  const [loading, setLoading] = useState(false);

  // Multi-specialty data for trend charts
  const [allForecasts, setAllForecasts] = useState<Record<number, LstmForecast[]>>({});
  const [trendsLoading, setTrendsLoading] = useState(false);

  useEffect(() => {
    getLabels('specialty').then(setSpecialties);
  }, []);

  // Fetch top specialties for trend charts on mount
  useEffect(() => {
    if (specialties.length === 0) return;
    setTrendsLoading(true);
    Promise.all(
      TOP_SPECIALTY_IDXS.map((idx) =>
        getForecasts(idx).then((data) => [idx, data] as const)
      )
    )
      .then((results) => {
        const map: Record<number, LstmForecast[]> = {};
        for (const [idx, data] of results) map[idx] = data;
        setAllForecasts(map);
      })
      .catch(() => {})
      .finally(() => setTrendsLoading(false));
  }, [specialties]);

  // Fetch selected specialty forecasts + history
  useEffect(() => {
    if (!selectedSpecialty) { setForecasts([]); setHistory([]); return; }
    setLoading(true);
    Promise.all([
      getForecasts(selectedSpecialty.idx),
      getSpecialtyHistory(selectedSpecialty.idx),
    ])
      .then(([fc, hist]) => { setForecasts(fc); setHistory(hist); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [selectedSpecialty]);

  // ── Chart data for selected specialty (historical + forecast) ──
  const chartData = useMemo(() => {
    if (!forecasts.length) return [];

    // Historical years from specialty_yearly_avg
    const points: { year: number; mean: number; p10: number | null; p90: number | null; isHistorical: boolean }[] = [];
    for (const h of history) {
      points.push({ year: h.year, mean: h.mean_allowed, p10: null, p90: null, isHistorical: true });
    }

    // Forecast years (aggregated across states/buckets)
    const byYear: Record<number, { means: number[]; p10s: number[]; p90s: number[] }> = {};
    for (const f of forecasts) {
      if (!byYear[f.forecast_year]) byYear[f.forecast_year] = { means: [], p10s: [], p90s: [] };
      byYear[f.forecast_year].means.push(f.forecast_mean);
      byYear[f.forecast_year].p10s.push(f.forecast_p10);
      byYear[f.forecast_year].p90s.push(f.forecast_p90);
    }

    // If no history data, use the single anchor point from forecasts
    if (history.length === 0) {
      const lastKnown = forecasts.find((f) => f.last_known_year && f.last_known_value);
      if (lastKnown?.last_known_year && lastKnown?.last_known_value) {
        points.push({ year: lastKnown.last_known_year, mean: lastKnown.last_known_value, p10: null, p90: null, isHistorical: true });
      }
    }

    const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
    for (const [yr, vals] of Object.entries(byYear)) {
      points.push({
        year: Number(yr),
        mean: avg(vals.means),
        p10: avg(vals.p10s),
        p90: avg(vals.p90s),
        isHistorical: false,
      });
    }

    // Deduplicate by year (prefer historical for overlapping years)
    const yearMap = new Map<number, typeof points[0]>();
    for (const p of points) {
      if (!yearMap.has(p.year) || p.isHistorical) yearMap.set(p.year, p);
    }
    return Array.from(yearMap.values()).sort((a, b) => a.year - b.year);
  }, [forecasts, history]);

  // ── Summary stats ──
  const historicalPoints = chartData.filter((d) => d.isHistorical);
  const lastHistorical = historicalPoints.length > 0 ? historicalPoints[historicalPoints.length - 1] : null;
  const lastKnownValue = lastHistorical?.mean ?? null;
  const lastKnownYear = lastHistorical?.year ?? null;
  const p50_2026 = chartData.find((d) => d.year === 2026)?.mean ?? null;
  const growthPct = lastKnownValue && p50_2026
    ? ((p50_2026 / lastKnownValue - 1) * 100)
    : null;

  // ── Forecast detail table data ──
  const tableRows = useMemo(() => {
    if (!chartData.length) return [];
    const rows = chartData.map((d, i) => ({
      year: d.year,
      p10: d.isHistorical ? null : d.p10,
      p50: d.mean,
      p90: d.isHistorical ? null : d.p90,
      isHistorical: d.isHistorical,
      yoyChange: i > 0 && chartData[i - 1]
        ? ((d.mean / chartData[i - 1].mean - 1) * 100)
        : null,
    }));
    return rows;
  }, [chartData]);

  // ── Multi-specialty trend chart data ──
  const trendChartData = useMemo(() => {
    if (Object.keys(allForecasts).length === 0) return [];
    const yearMap: Record<number, Record<string, number>> = {};

    for (const [idxStr, data] of Object.entries(allForecasts)) {
      const idx = Number(idxStr);
      const specLabel = specialties.find((s) => s.idx === idx)?.label ?? `Specialty ${idx}`;

      // Add historical anchor
      const anchor = data.find((f) => f.last_known_year && f.last_known_value);
      if (anchor?.last_known_year && anchor?.last_known_value) {
        if (!yearMap[anchor.last_known_year]) yearMap[anchor.last_known_year] = {};
        yearMap[anchor.last_known_year][specLabel] = anchor.last_known_value;
      }

      // Add forecast years (average across states/buckets)
      const byYear: Record<number, number[]> = {};
      for (const f of data) {
        if (!byYear[f.forecast_year]) byYear[f.forecast_year] = [];
        byYear[f.forecast_year].push(f.forecast_mean);
      }
      for (const [yr, vals] of Object.entries(byYear)) {
        const y = Number(yr);
        if (!yearMap[y]) yearMap[y] = {};
        yearMap[y][specLabel] = vals.reduce((a, b) => a + b, 0) / vals.length;
      }
    }

    return Object.entries(yearMap)
      .map(([yr, vals]) => ({ year: Number(yr), ...vals }))
      .sort((a, b) => a.year - b.year);
  }, [allForecasts, specialties]);

  // ── Distribution data (2026 by bucket) ──
  const distributionData = useMemo(() => {
    if (Object.keys(allForecasts).length === 0) return [];
    const all2026: LstmForecast[] = [];
    for (const data of Object.values(allForecasts)) {
      all2026.push(...data.filter((f) => f.forecast_year === 2026));
    }

    const bucketMap: Record<number, { p10s: number[]; means: number[]; p90s: number[] }> = {};
    for (const f of all2026) {
      if (!bucketMap[f.hcpcs_bucket]) bucketMap[f.hcpcs_bucket] = { p10s: [], means: [], p90s: [] };
      bucketMap[f.hcpcs_bucket].p10s.push(f.forecast_p10);
      bucketMap[f.hcpcs_bucket].means.push(f.forecast_mean);
      bucketMap[f.hcpcs_bucket].p90s.push(f.forecast_p90);
    }

    const avg = (arr: number[]) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
    return Object.entries(bucketMap)
      .map(([bucket, vals]) => {
        const p10 = avg(vals.p10s);
        const p50 = avg(vals.means);
        const p90 = avg(vals.p90s);
        return {
          bucket: Number(bucket),
          name: HCPCS_BUCKET_NAMES[Number(bucket)] ?? `Bucket ${bucket}`,
          p10,
          p50,
          p90,
          range: [p10, p90] as [number, number],
          errorLow: p50 - p10,
          errorHigh: p90 - p50,
        };
      })
      .sort((a, b) => a.bucket - b.bucket);
  }, [allForecasts]);

  // ── Growth data ──
  const growthData = useMemo(() => {
    if (Object.keys(allForecasts).length === 0) return [];
    const rows: { name: string; cagr: number; idx: number }[] = [];

    for (const [idxStr, data] of Object.entries(allForecasts)) {
      const idx = Number(idxStr);
      const specLabel = specialties.find((s) => s.idx === idx)?.label ?? `Specialty ${idx}`;
      const anchor = data.find((f) => f.last_known_year && f.last_known_value);
      if (!anchor?.last_known_value) continue;

      const forecast2026 = data.filter((f) => f.forecast_year === 2026);
      if (!forecast2026.length) continue;
      const avg2026 = forecast2026.reduce((a, b) => a + b.forecast_mean, 0) / forecast2026.length;

      const cagr = (Math.pow(avg2026 / anchor.last_known_value, 1 / 3) - 1) * 100;
      rows.push({ name: specLabel, cagr, idx });
    }

    return rows.sort((a, b) => b.cagr - a.cagr).slice(0, 8);
  }, [allForecasts, specialties]);

  // Specialty names for trend chart
  const trendSpecNames = useMemo(() => {
    return TOP_SPECIALTY_IDXS.map((idx) => ({
      idx,
      label: specialties.find((s) => s.idx === idx)?.label ?? `Specialty ${idx}`,
      color: SPECIALTY_COLORS[idx] ?? '#A8A29E',
    }));
  }, [specialties]);

  return (
    <Box>
      {/* Page header */}
      <Box sx={{ pb: 3, mb: 4, borderBottom: '1px solid', borderColor: 'divider' }}>
        <Typography variant="h4" sx={{ fontWeight: 800, letterSpacing: '-0.02em' }} gutterBottom>
          Forecast Explorer
        </Typography>
        <Typography variant="body1" color="text.secondary">
          LSTM-powered Medicare cost forecasts through 2026 with confidence bounds. Select a specialty to see projected trends.
        </Typography>
      </Box>

      {/* Specialty selector */}
      <Autocomplete
        options={specialties}
        getOptionLabel={(o) => o.label}
        value={selectedSpecialty}
        onChange={(_, v) => setSelectedSpecialty(v)}
        renderInput={(params) => <TextField {...params} label="Select Specialty" />}
        sx={{ maxWidth: 500, mb: 4 }}
      />

      {loading && <CircularProgress sx={{ display: 'block', mx: 'auto', my: 4 }} />}

      {/* ── Summary stat cards ── */}
      {chartData.length > 0 && (
        <Grid container spacing={2} sx={{ mb: 4 }}>
          <Grid size={{ xs: 12, sm: 4 }}>
            <Card>
              <CardContent>
                <Typography variant="overline" color="text.disabled" sx={{ display: 'block', letterSpacing: '0.06em' }}>
                  {lastKnownYear} Avg Allowed
                </Typography>
                <Typography sx={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: 24, fontWeight: 700, color: 'primary.main' }}>
                  {lastKnownValue != null ? formatDollars(lastKnownValue) : 'N/A'}
                </Typography>
                <Typography variant="body2" color="text.disabled" sx={{ fontSize: 12 }}>Last observed year</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid size={{ xs: 12, sm: 4 }}>
            <Card>
              <CardContent>
                <Typography variant="overline" color="text.disabled" sx={{ display: 'block', letterSpacing: '0.06em' }}>
                  2026 Forecast (P50)
                </Typography>
                <Typography sx={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: 24, fontWeight: 700, color: 'primary.main' }}>
                  {p50_2026 != null ? formatDollars(p50_2026) : 'N/A'}
                </Typography>
                <Typography variant="body2" color="text.disabled" sx={{ fontSize: 12 }}>Median projection</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid size={{ xs: 12, sm: 4 }}>
            <Card>
              <CardContent>
                <Typography variant="overline" color="text.disabled" sx={{ display: 'block', letterSpacing: '0.06em' }}>
                  Projected Growth
                </Typography>
                <Typography sx={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: 24, fontWeight: 700, color: 'secondary.main' }}>
                  {growthPct != null ? `${growthPct >= 0 ? '+' : ''}${growthPct.toFixed(1)}%` : 'N/A'}
                </Typography>
                <Typography variant="body2" color="text.disabled" sx={{ fontSize: 12 }}>3-year CAGR</Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* ── Forecast chart ── */}
      {chartData.length > 0 && (
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h6" sx={{ fontWeight: 700 }} gutterBottom>
              {selectedSpecialty?.label}: Cost Forecast
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Historical ({history.length > 0 ? `${history[0].year} to ${lastKnownYear}` : lastKnownYear}) + projected 2024 to 2026 with P10/P90 confidence bounds
            </Typography>
            <ResponsiveContainer width="100%" height={400}>
              <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                <XAxis dataKey="year" />
                <YAxis tickFormatter={(v) => `$${Number(v).toFixed(0)}`} />
                <Tooltip formatter={(v) => formatDollars(Number(v))} />
                {lastKnownYear && (
                  <ReferenceLine
                    x={lastKnownYear}
                    stroke="#0F6E8C"
                    strokeDasharray="4 4"
                    strokeOpacity={0.3}
                    label={{ value: 'FORECAST', position: 'insideBottomRight', fill: '#0F6E8C', fontSize: 10, fontWeight: 600, opacity: 0.4, offset: 8 }}
                  />
                )}
                <Area type="monotone" dataKey="p90" stroke="none" fill="#0F6E8C" fillOpacity={0.08} name="P90 Bound" />
                <Area type="monotone" dataKey="p10" stroke="none" fill="#FAFAF8" fillOpacity={1} name="P10 Bound" />
                <Line type="monotone" dataKey="p90" stroke="#0F6E8C" strokeWidth={1.5} strokeDasharray="6 4" dot={false} opacity={0.4} name="P90" />
                <Line type="monotone" dataKey="p10" stroke="#0F6E8C" strokeWidth={1.5} strokeDasharray="6 4" dot={false} opacity={0.4} name="P10" />
                <Line type="monotone" dataKey="mean" stroke="#0F6E8C" strokeWidth={3} dot={{ r: 5, fill: '#0F6E8C', stroke: '#fff', strokeWidth: 2 }} name="Forecast Mean" />
              </ComposedChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* ── Forecast detail table ── */}
      {tableRows.length > 0 && (
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h6" sx={{ fontWeight: 700, mb: 2 }}>
              Forecast Detail: {selectedSpecialty?.label}
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell><strong>Year</strong></TableCell>
                    <TableCell align="right"><strong>P10 (Low)</strong></TableCell>
                    <TableCell align="right"><strong>P50 (Median)</strong></TableCell>
                    <TableCell align="right"><strong>P90 (High)</strong></TableCell>
                    <TableCell align="right"><strong>YoY Change</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {tableRows.map((row) => (
                    <TableRow key={row.year} hover>
                      <TableCell>
                        <strong>{row.year}</strong>
                        {row.isHistorical && (
                          <Typography component="span" sx={{ fontSize: 11, color: 'text.disabled', ml: 0.5 }}>(observed)</Typography>
                        )}
                      </TableCell>
                      <TableCell align="right" sx={{ fontFamily: '"IBM Plex Mono", monospace' }}>
                        {row.p10 != null ? formatDollars(row.p10) : '-'}
                      </TableCell>
                      <TableCell align="right" sx={{ fontFamily: '"IBM Plex Mono", monospace', fontWeight: 600 }}>
                        {formatDollars(row.p50)}
                      </TableCell>
                      <TableCell align="right" sx={{ fontFamily: '"IBM Plex Mono", monospace' }}>
                        {row.p90 != null ? formatDollars(row.p90) : '-'}
                      </TableCell>
                      <TableCell align="right" sx={{ color: row.yoyChange != null ? (row.yoyChange >= 0 ? 'secondary.main' : 'error.main') : 'text.primary', fontWeight: 600 }}>
                        {row.yoyChange != null && !row.isHistorical ? `${row.yoyChange >= 0 ? '+' : ''}${row.yoyChange.toFixed(1)}%` : '-'}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}

      {/* ── Overall Trends section ── */}
      <Divider sx={{ my: 4 }} />
      <Typography variant="h5" sx={{ fontWeight: 700, mb: 3 }}>Overall Trends</Typography>

      {trendsLoading && <CircularProgress sx={{ display: 'block', mx: 'auto', my: 4 }} />}

      {/* Multi-line specialty trends chart */}
      {trendChartData.length > 0 && (
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h6" sx={{ fontWeight: 700 }} gutterBottom>
              Top Specialties: Historical + Forecast
            </Typography>
            <ResponsiveContainer width="100%" height={420}>
              <LineChart data={trendChartData} margin={{ top: 10, right: 30, left: 20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                <XAxis dataKey="year" />
                <YAxis tickFormatter={(v) => `$${Number(v).toFixed(0)}`} />
                <Tooltip formatter={(v) => formatDollars(Number(v))} />
                {trendSpecNames.map((spec) => (
                  <Line
                    key={spec.idx}
                    type="monotone"
                    dataKey={spec.label}
                    stroke={spec.color}
                    strokeWidth={2}
                    dot={false}
                    connectNulls
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
            {/* Legend */}
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mt: 2, pt: 2, borderTop: '1px solid', borderColor: 'divider' }}>
              {trendSpecNames.map((spec) => (
                <Box key={spec.idx} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: spec.color }} />
                  <Typography variant="body2" color="text.secondary" sx={{ fontSize: 12 }}>{spec.label}</Typography>
                </Box>
              ))}
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Two-column: Distribution + Growth */}
      <Grid container spacing={3}>
        {/* 2026 Forecast Distribution by bucket */}
        {distributionData.length > 0 && (
          <Grid size={{ xs: 12, md: 6 }}>
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ fontWeight: 700 }} gutterBottom>
                  2026 Forecast Distribution
                </Typography>
                <ResponsiveContainer width="100%" height={320}>
                  <ComposedChart data={distributionData} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                    <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                    <YAxis tickFormatter={(v) => `$${Number(v).toFixed(0)}`} />
                    <Tooltip formatter={(v) => formatDollars(Number(v))} />
                    <Bar dataKey="p50" fill="#0F6E8C" fillOpacity={0.15} stroke="#0F6E8C" strokeWidth={1.5} radius={[3, 3, 0, 0]} name="Median (P50)">
                      <ErrorBar dataKey="errorHigh" direction="y" width={8} stroke="#0F6E8C" strokeWidth={1.5} />
                    </Bar>
                    <Line type="monotone" dataKey="p10" stroke="#0F6E8C" strokeDasharray="4 4" strokeWidth={1} dot={{ r: 3, fill: '#0F6E8C' }} name="P10" />
                    <Line type="monotone" dataKey="p90" stroke="#0F6E8C" strokeDasharray="4 4" strokeWidth={1} dot={{ r: 3, fill: '#0F6E8C' }} name="P90" />
                  </ComposedChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Top Growth Specialties */}
        {growthData.length > 0 && (
          <Grid size={{ xs: 12, md: 6 }}>
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ fontWeight: 700 }} gutterBottom>
                  Top Growth Specialties (3yr CAGR)
                </Typography>
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={growthData} layout="vertical" margin={{ top: 0, right: 60, left: 10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                    <XAxis type="number" tickFormatter={(v) => `${Number(v).toFixed(0)}%`} />
                    <YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 12 }} />
                    <Tooltip formatter={(v) => `+${Number(v).toFixed(1)}%`} />
                    <Bar dataKey="cagr" fill="#15755D" radius={[0, 4, 4, 0]} name="3yr CAGR" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );
}
