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
import {
  ResponsiveContainer, ComposedChart, Line, Area,
  XAxis, YAxis, Tooltip, CartesianGrid, Legend,
} from 'recharts';
import { getLabels, getForecasts } from '@/lib/queries';
import { formatDollars } from '@/lib/formatters';
import type { LookupLabel, LstmForecast } from '@/lib/types';
import Image from 'next/image';

export default function ForecastPage() {
  const [specialties, setSpecialties] = useState<LookupLabel[]>([]);
  const [selectedSpecialty, setSelectedSpecialty] = useState<LookupLabel | null>(null);
  const [forecasts, setForecasts] = useState<LstmForecast[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getLabels('specialty').then(setSpecialties);
  }, []);

  useEffect(() => {
    if (!selectedSpecialty) { setForecasts([]); return; }
    setLoading(true);
    getForecasts(selectedSpecialty.idx)
      .then(setForecasts)
      .finally(() => setLoading(false));
  }, [selectedSpecialty]);

  const chartData = useMemo(() => {
    if (!forecasts.length) return [];
    const byYear: Record<number, { means: number[]; p10s: number[]; p90s: number[] }> = {};
    for (const f of forecasts) {
      if (!byYear[f.forecast_year]) byYear[f.forecast_year] = { means: [], p10s: [], p90s: [] };
      byYear[f.forecast_year].means.push(f.forecast_mean);
      byYear[f.forecast_year].p10s.push(f.forecast_p10);
      byYear[f.forecast_year].p90s.push(f.forecast_p90);
    }

    // Add last known historical point as anchor
    const lastKnown = forecasts.find((f) => f.last_known_year && f.last_known_value);
    const points = [];
    if (lastKnown?.last_known_year && lastKnown?.last_known_value) {
      points.push({
        year: lastKnown.last_known_year,
        mean: lastKnown.last_known_value,
        p10: lastKnown.last_known_value,
        p90: lastKnown.last_known_value,
      });
    }

    for (const [yr, vals] of Object.entries(byYear)) {
      const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
      points.push({
        year: Number(yr),
        mean: avg(vals.means),
        p10: avg(vals.p10s),
        p90: avg(vals.p90s),
      });
    }
    return points.sort((a, b) => a.year - b.year);
  }, [forecasts]);

  return (
    <Box>
      <Typography variant="h4" gutterBottom>Forecast Explorer</Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        LSTM-powered Medicare cost forecasts through 2026 with confidence bounds.
      </Typography>

      <Autocomplete
        options={specialties}
        getOptionLabel={(o) => o.label}
        value={selectedSpecialty}
        onChange={(_, v) => setSelectedSpecialty(v)}
        renderInput={(params) => <TextField {...params} label="Select Specialty" />}
        sx={{ maxWidth: 500, mb: 4 }}
      />

      {loading && <CircularProgress sx={{ display: 'block', mx: 'auto', my: 4 }} />}

      {chartData.length > 0 && (
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Forecast: {selectedSpecialty?.label}
            </Typography>
            <ResponsiveContainer width="100%" height={400}>
              <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis tickFormatter={(v) => `$${Number(v).toFixed(0)}`} />
                <Tooltip formatter={(v) => formatDollars(Number(v))} />
                <Legend />
                <Area type="monotone" dataKey="p90" stackId="1" stroke="none" fill="#1565c0" fillOpacity={0.1} name="P90" />
                <Area type="monotone" dataKey="p10" stackId="2" stroke="none" fill="#ffffff" fillOpacity={0} name="P10" />
                <Line type="monotone" dataKey="mean" stroke="#1565c0" strokeWidth={3} dot={{ r: 5 }} name="Forecast Mean" />
                <Line type="monotone" dataKey="p10" stroke="#1565c0" strokeWidth={1} strokeDasharray="5 5" dot={false} name="P10 Bound" />
                <Line type="monotone" dataKey="p90" stroke="#1565c0" strokeWidth={1} strokeDasharray="5 5" dot={false} name="P90 Bound" />
              </ComposedChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Pre-generated LSTM visualizations */}
      <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>Overall Trends</Typography>
      <Grid container spacing={3}>
        <Grid size={{ xs: 12 }}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Top 12 Specialties: Historical + Forecast</Typography>
              <Image src="/plots/specialty_trends.png" alt="Specialty trends" width={1000} height={750} style={{ width: '100%', height: 'auto' }} />
            </CardContent>
          </Card>
        </Grid>
        <Grid size={{ xs: 12, md: 6 }}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>2026 Forecast Distribution</Typography>
              <Image src="/plots/forecast_distribution.png" alt="Forecast distribution" width={600} height={360} style={{ width: '100%', height: 'auto' }} />
            </CardContent>
          </Card>
        </Grid>
        <Grid size={{ xs: 12, md: 6 }}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Top Growth Specialties</Typography>
              <Image src="/plots/top_growth_specialties.png" alt="Top growth specialties" width={600} height={480} style={{ width: '100%', height: 'auto' }} />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
