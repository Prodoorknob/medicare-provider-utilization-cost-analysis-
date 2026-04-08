'use client';

import { useState, useEffect } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import TableSortLabel from '@mui/material/TableSortLabel';
import TextField from '@mui/material/TextField';
import Image from 'next/image';
import { getStateSummary } from '@/lib/queries';
import { formatDollars, formatNumber } from '@/lib/formatters';
import type { StateSummary } from '@/lib/types';

type SortKey = 'state_abbrev' | 'mean_allowed' | 'median_allowed' | 'n_records';

export default function ExplorerPage() {
  const [states, setStates] = useState<StateSummary[]>([]);
  const [sortKey, setSortKey] = useState<SortKey>('mean_allowed');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [filter, setFilter] = useState('');

  useEffect(() => {
    getStateSummary().then(setStates);
  }, []);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('desc'); }
  };

  const filtered = states
    .filter((s) => s.state_abbrev.toLowerCase().includes(filter.toLowerCase()))
    .sort((a, b) => {
      const m = sortDir === 'asc' ? 1 : -1;
      if (sortKey === 'state_abbrev') return m * a.state_abbrev.localeCompare(b.state_abbrev);
      return m * ((a[sortKey] as number) - (b[sortKey] as number));
    });

  return (
    <Box>
      <Typography variant="h4" gutterBottom>Data Explorer</Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Explore feature distributions, correlations, and state-level Medicare cost summaries.
      </Typography>

      {/* EDA Plots */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid size={{ xs: 12, md: 6 }}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Feature Distributions</Typography>
              <Image src="/plots/01_distributions.png" alt="Feature distributions" width={600} height={400} style={{ width: '100%', height: 'auto' }} />
            </CardContent>
          </Card>
        </Grid>
        <Grid size={{ xs: 12, md: 6 }}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Correlation Heatmap</Typography>
              <Image src="/plots/02_correlation_heatmap.png" alt="Correlation heatmap" width={600} height={500} style={{ width: '100%', height: 'auto' }} />
            </CardContent>
          </Card>
        </Grid>
        <Grid size={{ xs: 12 }}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Top Provider Types</Typography>
              <Image src="/plots/03_top_provider_types.png" alt="Top provider types" width={800} height={500} style={{ width: '100%', height: 'auto' }} />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* State Summary Table */}
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">State-Level Summary</Typography>
            <TextField
              size="small"
              placeholder="Filter states..."
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              sx={{ width: 200 }}
            />
          </Box>
          <TableContainer sx={{ maxHeight: 500 }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  {([
                    ['state_abbrev', 'State'],
                    ['mean_allowed', 'Mean Allowed'],
                    ['median_allowed', 'Median Allowed'],
                    ['n_records', 'Records'],
                  ] as [SortKey, string][]).map(([key, label]) => (
                    <TableCell key={key} align={key === 'state_abbrev' ? 'left' : 'right'}>
                      <TableSortLabel active={sortKey === key} direction={sortKey === key ? sortDir : 'asc'} onClick={() => handleSort(key)}>
                        {label}
                      </TableSortLabel>
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {filtered.map((s) => (
                  <TableRow key={s.state_abbrev} hover>
                    <TableCell><strong>{s.state_abbrev}</strong></TableCell>
                    <TableCell align="right">{formatDollars(s.mean_allowed)}</TableCell>
                    <TableCell align="right">{formatDollars(s.median_allowed)}</TableCell>
                    <TableCell align="right">{formatNumber(s.n_records)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </Box>
  );
}
