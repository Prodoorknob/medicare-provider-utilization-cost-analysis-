'use client';

import { useState, useEffect } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Link from '@mui/material/Link';
import Divider from '@mui/material/Divider';
import Chip from '@mui/material/Chip';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import TableSortLabel from '@mui/material/TableSortLabel';
import TextField from '@mui/material/TextField';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
} from 'recharts';
import { getStateSummary } from '@/lib/queries';
import { formatDollars, formatNumber } from '@/lib/formatters';
import type { StateSummary } from '@/lib/types';
import {
  V2_STAGE1_MODELS,
  V2_FEATURE_IMPORTANCES,
  V2_METHODOLOGIES,
  CORRELATION_FEATURES,
  CORRELATION_DISPLAY_NAMES,
  CORRELATION_MATRIX,
  FEATURE_HISTOGRAMS,
  TOP_PROVIDER_TYPES,
} from '@/lib/v2-model-data';

type SortKey = 'state_abbrev' | 'mean_allowed' | 'median_allowed' | 'n_records';

// ── Correlation Heatmap (custom SVG) ──

function CorrelationHeatmap() {
  const n = CORRELATION_FEATURES.length;
  const cellSize = 56;
  const labelWidth = 90;
  const topPad = 85;
  const w = labelWidth + n * cellSize;
  const h = topPad + n * cellSize;

  const getColor = (val: number) => {
    const abs = Math.abs(val);
    if (val < 0) return `rgba(220, 38, 38, ${abs * 0.9})`;
    return `rgba(15, 110, 140, ${abs * 0.9})`;
  };
  const textColor = (val: number) => Math.abs(val) > 0.4 ? '#fff' : '#57534E';

  return (
    <Box sx={{ width: '100%', overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', maxWidth: w }}>
        {/* Column headers (rotated) */}
        {CORRELATION_FEATURES.map((feat, i) => (
          <text
            key={`col-${feat}`}
            x={labelWidth + i * cellSize + cellSize / 2}
            y={topPad - 12}
            textAnchor="middle"
            fontSize="9"
            fill="#57534E"
            fontFamily="Inter, sans-serif"
            transform={`rotate(-40, ${labelWidth + i * cellSize + cellSize / 2}, ${topPad - 12})`}
          >
            {CORRELATION_DISPLAY_NAMES[feat]}
          </text>
        ))}
        {/* Rows */}
        {CORRELATION_FEATURES.map((rowFeat, r) => (
          <g key={`row-${rowFeat}`}>
            <text
              x={labelWidth - 8}
              y={topPad + r * cellSize + cellSize / 2 + 4}
              textAnchor="end"
              fontSize="10"
              fill="#57534E"
              fontFamily="Inter, sans-serif"
            >
              {CORRELATION_DISPLAY_NAMES[rowFeat]}
            </text>
            {CORRELATION_FEATURES.map((colFeat, c) => {
              const val = CORRELATION_MATRIX[r][c];
              return (
                <g key={`cell-${r}-${c}`}>
                  <rect
                    x={labelWidth + c * cellSize + 2}
                    y={topPad + r * cellSize + 2}
                    width={cellSize - 4}
                    height={cellSize - 4}
                    rx={3}
                    fill={getColor(val)}
                  />
                  <text
                    x={labelWidth + c * cellSize + cellSize / 2}
                    y={topPad + r * cellSize + cellSize / 2 + 4}
                    textAnchor="middle"
                    fontSize="10"
                    fill={textColor(val)}
                    fontFamily="IBM Plex Mono, monospace"
                  >
                    {val === 1 ? '1.00' : val.toFixed(2)}
                  </text>
                </g>
              );
            })}
          </g>
        ))}
      </svg>
    </Box>
  );
}

export default function AboutPage() {
  const [tab, setTab] = useState(0);
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

  const filteredStates = states
    .filter((s) => s.state_abbrev.toLowerCase().includes(filter.toLowerCase()))
    .sort((a, b) => {
      const m = sortDir === 'asc' ? 1 : -1;
      if (sortKey === 'state_abbrev') return m * a.state_abbrev.localeCompare(b.state_abbrev);
      return m * ((a[sortKey] as number) - (b[sortKey] as number));
    });

  // Feature importance chart data
  const fiChart = V2_FEATURE_IMPORTANCES.map((x) => ({
    name: x.name,
    importance: x.importance,
  }));

  return (
    <Box>
      <Box sx={{ pb: 3, mb: 0, borderBottom: '1px solid', borderColor: 'divider' }}>
        <Typography variant="h4" sx={{ fontWeight: 800, letterSpacing: '-0.02em' }} gutterBottom>
          About &amp; Methodology
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Data sources, model performance, pipeline architecture, and project attribution.
        </Typography>
      </Box>

      <Tabs
        value={tab}
        onChange={(_, v) => setTab(v)}
        indicatorColor="primary"
        textColor="primary"
        sx={{ borderBottom: '1px solid', borderColor: 'divider', mb: 4 }}
      >
        <Tab label="Overview" />
        <Tab label="Data" />
        <Tab label="Models" />
        <Tab label="Pipeline" />
      </Tabs>

      {/* ════ TAB 0: Overview ════ */}
      {tab === 0 && (
        <Grid container spacing={4}>
          <Grid size={{ xs: 12, md: 8 }}>
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>Overview</Typography>
                <Typography variant="body1" sx={{ mb: 1.5 }}>
                  An end-to-end data science pipeline predicting Medicare provider costs and patient out-of-pocket expenses. Processes over 103 million provider-service records from the CMS Medicare Physician &amp; Other Practitioners dataset spanning 2013 to 2023.
                </Typography>
                <Typography variant="body1">
                  The project implements a two-stage prediction architecture: Stage 1 predicts what Medicare allows for a service; Stage 2 estimates what the patient pays out of pocket based on demographic and insurance profile.
                </Typography>
              </CardContent>
            </Card>

            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>Data Sources</Typography>
                <Typography variant="body1" sx={{ mb: 1.5 }}>
                  <strong>CMS Medicare Physician &amp; Other Practitioners (2013 to 2023)</strong>: Provider-level utilization and payment data for Part B services. Over 10 million records per year across all 50 states and territories.
                </Typography>
                <Typography variant="body1" sx={{ mb: 1.5 }}>
                  <strong>CMS Medicare Current Beneficiary Survey (MCBS)</strong>: Public Use Files for beneficiary demographics, insurance coverage, and cost data. Used for Stage 2 patient cost modeling.
                </Typography>
                <Typography variant="body1">
                  <strong>CMS Provider Summary (by Provider)</strong>: NPI-level HCC risk scores for beneficiary health burden adjustment.
                </Typography>
              </CardContent>
            </Card>

            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom color="primary">Stage 1: Medicare Allowed Amount</Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
                  LightGBM no-charge (R²=0.943), CatBoost monotonic, XGBoost, Random Forest, LSTM
                </Typography>
                <Divider sx={{ my: 1.5 }} />
                <Typography variant="h6" gutterBottom color="secondary" sx={{ mt: 1.5 }}>Stage 2: Patient Out-of-Pocket</Typography>
                <Typography variant="body2" color="text.secondary">
                  CatBoost monotonic quantile regression (P10/P50/P90) trained on synthetic MCBS-derived beneficiary data segmented by region, age, income, and insurance status.
                </Typography>
              </CardContent>
            </Card>

            <Box sx={{ bgcolor: '#FDF4EA', borderLeft: '3px solid #B8763A', borderRadius: 1, p: 2 }}>
              <Typography variant="body2" color="#B8763A" sx={{ fontWeight: 600, mb: 1 }}>Disclaimer</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.7 }}>
                This is an academic research project. Estimates are based on aggregate historical data and should not be used for medical billing decisions. Stage 2 out-of-pocket estimates use synthetic beneficiary data modeled after MCBS distributions. Actual patient costs depend on specific plan details, deductibles, and coverage terms not captured in this model.
              </Typography>
            </Box>
          </Grid>

          <Grid size={{ xs: 12, md: 4 }}>
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>Author</Typography>
                <Typography variant="body1"><strong>Raj Vedire</strong></Typography>
                <Typography variant="body2" color="text.secondary">Indiana University</Typography>
                <Typography variant="body2" color="text.secondary">rvedire@iu.edu</Typography>
              </CardContent>
            </Card>

            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>Links</Typography>
                <Typography variant="body2" sx={{ mb: 1.5 }}>
                  <Link href="https://github.com/Prodoorknob/medicare-provider-utilization-cost-analysis-" target="_blank" rel="noopener">
                    GitHub Repository
                  </Link>
                </Typography>
                <Typography variant="body2">
                  <Link href="https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners" target="_blank" rel="noopener">
                    CMS Data Source
                  </Link>
                </Typography>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>Tech Stack</Typography>
                <Typography variant="body2">Python &middot; pandas &middot; PyArrow &middot; scikit-learn</Typography>
                <Typography variant="body2">XGBoost &middot; CatBoost &middot; LightGBM &middot; PyTorch</Typography>
                <Typography variant="body2">MLflow &middot; Databricks</Typography>
                <Divider sx={{ my: 1.5 }} />
                <Typography variant="body2">Next.js &middot; Material UI &middot; Recharts</Typography>
                <Typography variant="body2">Supabase (PostgreSQL) &middot; Railway &middot; Vercel</Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* ════ TAB 1: Data ════ */}
      {tab === 1 && (
        <Box>
          {/* Feature distributions + Correlation heatmap */}
          <Grid container spacing={3} sx={{ mb: 4, alignItems: 'stretch' }}>
            <Grid size={{ xs: 12, md: 6 }} sx={{ display: 'flex' }}>
              <Card sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                  <Typography variant="h6" gutterBottom>Feature Distributions</Typography>
                  <Grid container spacing={1} sx={{ flex: 1, alignContent: 'center' }}>
                    {FEATURE_HISTOGRAMS.map((feat) => (
                      <Grid key={feat.name} size={{ xs: 4 }}>
                        <Typography variant="body2" sx={{ fontSize: 10, fontWeight: 600, color: feat.color, mb: 0.5 }}>
                          {feat.displayName}
                        </Typography>
                        <ResponsiveContainer width="100%" height={70}>
                          <BarChart data={feat.bins} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                            <Bar dataKey="height" fill={feat.color} fillOpacity={0.6} radius={[2, 2, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </Grid>
                    ))}
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, md: 6 }} sx={{ display: 'flex' }}>
              <Card sx={{ flex: 1 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>Correlation Heatmap</Typography>
                  <CorrelationHeatmap />
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Top provider types */}
          <Card sx={{ mb: 4 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>Top Provider Types by Record Count</Typography>
              <ResponsiveContainer width="100%" height={360}>
                <BarChart data={TOP_PROVIDER_TYPES} layout="vertical" margin={{ left: 140, right: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                  <XAxis type="number" tickFormatter={(v) => `${(Number(v) / 1_000_000).toFixed(0)}M`} />
                  <YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 12 }} />
                  <Tooltip formatter={(v) => `${(Number(v) / 1_000_000).toFixed(1)}M records`} />
                  <Bar dataKey="records" fill="#0F6E8C" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* State-level summary table */}
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
                    {filteredStates.map((s) => (
                      <TableRow key={s.state_abbrev} hover>
                        <TableCell><strong>{s.state_abbrev}</strong></TableCell>
                        <TableCell align="right" sx={{ fontFamily: '"IBM Plex Mono", monospace' }}>{formatDollars(s.mean_allowed)}</TableCell>
                        <TableCell align="right" sx={{ fontFamily: '"IBM Plex Mono", monospace' }}>{formatDollars(s.median_allowed)}</TableCell>
                        <TableCell align="right" sx={{ fontFamily: '"IBM Plex Mono", monospace' }}>{formatNumber(s.n_records)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Box>
      )}

      {/* ════ TAB 2: Models ════ */}
      {tab === 2 && (
        <Box>
          {/* Model performance table */}
          <Card sx={{ mb: 4 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>Stage 1: Medicare Allowed Amount</Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell><strong>Model</strong></TableCell>
                      <TableCell align="right"><strong>Test MAE</strong></TableCell>
                      <TableCell align="right"><strong>Test RMSE</strong></TableCell>
                      <TableCell align="right"><strong>Test R²</strong></TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {V2_STAGE1_MODELS.map((m) => {
                      const isBest = m.badge === 'PRODUCTION';
                      return (
                        <TableRow key={m.name} sx={isBest ? { bgcolor: 'primary.50' } : {}}>
                          <TableCell sx={isBest ? { fontWeight: 700 } : {}}>
                            {m.name}
                            {m.badge && (
                              <Chip label={m.badge} size="small" color="primary" variant="outlined" sx={{ ml: 1, height: 20, fontSize: 10, fontWeight: 600 }} />
                            )}
                          </TableCell>
                          <TableCell align="right" sx={{ fontFamily: '"IBM Plex Mono", monospace' }}>
                            {m.mae != null ? formatDollars(m.mae) : 'N/A'}
                          </TableCell>
                          <TableCell align="right" sx={{ fontFamily: '"IBM Plex Mono", monospace' }}>
                            {m.rmse != null ? formatDollars(m.rmse) : 'N/A'}
                          </TableCell>
                          <TableCell align="right" sx={{ fontFamily: '"IBM Plex Mono", monospace', ...(isBest ? { fontWeight: 700, color: 'primary.main' } : {}) }}>
                            {m.r2 != null ? m.r2.toFixed(4) : 'diverged'}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>

          {/* Feature importance */}
          <Card sx={{ mb: 4 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>Feature Importance: LightGBM (no-charge)</Typography>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={fiChart} layout="vertical" margin={{ left: 140, right: 30 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                  <XAxis type="number" tickFormatter={(v) => `${(Number(v) * 100).toFixed(0)}%`} />
                  <YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 12 }} />
                  <Tooltip formatter={(v) => (Number(v) * 100).toFixed(1) + '%'} />
                  <Bar dataKey="importance" fill="#0F6E8C" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Methodology */}
          <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>Methodology</Typography>
          {V2_METHODOLOGIES.map((m) => (
            <Accordion key={m.name} sx={{ mb: 1 }}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography sx={{ fontWeight: 600 }}>{m.name}</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography variant="body2" color="text.secondary">{m.description}</Typography>
              </AccordionDetails>
            </Accordion>
          ))}
        </Box>
      )}

      {/* ════ TAB 3: Pipeline ════ */}
      {tab === 3 && (
        <Grid container spacing={3}>
          <Grid size={{ xs: 12, md: 8 }}>
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>Medallion Architecture</Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
                  Bronze, Silver, Gold pipeline with two execution modes: PySpark + Delta Lake on Databricks for production, pandas + pyarrow locally for development. All models log to Databricks MLflow.
                </Typography>
                <Box component="pre" sx={{
                  fontFamily: '"IBM Plex Mono", monospace', fontSize: 12,
                  bgcolor: 'grey.50', borderRadius: 2, p: 2, overflowX: 'auto',
                  color: 'text.secondary', lineHeight: 1.8, whiteSpace: 'pre-wrap',
                }}>
{`CMS API (by Provider & Service)
  > pull_medicare_data.py
  > partition_medicare_data.py  (injects year)
  > csv_to_parquet.py

Bronze  >  Silver  >  Gold  >  EDA + Modeling
         (typed,    (features +      |
          cleaned)   encoding)   LSTM sequences

CMS MCBS PUF
  > 06_mcbs_bronze > 07_mcbs_silver
  > 08_mcbs_crosswalk
         |
  generate_synthetic_mcbs.py
  > Stage 2 OOP training data`}
                </Box>
              </CardContent>
            </Card>

            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>Feature Set (10 Features)</Typography>
                <Grid container spacing={1}>
                  {[
                    ['Rndrng_Prvdr_Type_idx', 'Provider specialty (encoded)'],
                    ['Rndrng_Prvdr_State_Abrvtn_idx', 'State (encoded)'],
                    ['HCPCS_Cd_idx', 'HCPCS procedure code (~6K unique)'],
                    ['hcpcs_bucket', 'Clinical category (Anesthesia to HCPCS II)'],
                    ['place_of_srvc_flag', 'Binary: facility (1) or office (0)'],
                    ['Bene_Avg_Risk_Scre', 'NPI-level HCC risk score'],
                    ['log_srvcs', 'log1p(Total services)'],
                    ['log_benes', 'log1p(Total beneficiaries)'],
                    ['Avg_Sbmtd_Chrg', 'Submitted charge amount'],
                    ['srvcs_per_bene', 'Services per beneficiary ratio'],
                  ].map(([name, desc]) => (
                    <Grid key={name} size={{ xs: 12, sm: 6 }}>
                      <Box sx={{ p: 1.5, bgcolor: 'grey.50', borderRadius: 1 }}>
                        <Typography variant="body2" sx={{ fontFamily: '"IBM Plex Mono", monospace', color: 'primary.main', fontSize: 11 }}>{name}</Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ fontSize: 12 }}>{desc}</Typography>
                      </Box>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>Census Regions (Batch Training)</Typography>
                <Grid container spacing={1}>
                  {[
                    ['Northeast', 'CT ME MA NH RI VT NJ NY PA'],
                    ['South', 'DE FL GA MD NC SC VA DC WV AL KY MS TN AR LA OK TX'],
                    ['Midwest', 'IL IN MI OH WI IA KS MN MO NE ND SD'],
                    ['West', 'AZ CO ID MT NV NM UT WY AK CA HI OR WA'],
                  ].map(([region, regionStates]) => (
                    <Grid key={region} size={{ xs: 12, sm: 6 }}>
                      <Box sx={{ p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
                        <Typography variant="body2" color="primary.main" sx={{ fontWeight: 600 }}>{region}</Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ fontSize: 11, fontFamily: '"IBM Plex Mono", monospace' }}>{regionStates}</Typography>
                      </Box>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          <Grid size={{ xs: 12, md: 4 }}>
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>LSTM Forecasting</Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
                  2-layer PyTorch LSTM with static embeddings for specialty, state, and service bucket.
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
                  <strong>Train:</strong> 2013 to 2021 &middot; <strong>Val:</strong> 2022 to 2023
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  MC Dropout produces confidence-bounded (P10/P90) forecasts for 2024 to 2026 by specialty.
                </Typography>
              </CardContent>
            </Card>

            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>Training Modes</Typography>
                <Typography variant="body2" color="primary.main" sx={{ fontWeight: 600 }}>batch (default)</Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
                  Incremental by Census region. Memory-efficient. XGBoost: 125 rounds/region. RF: warm_start adds 125 trees/region.
                </Typography>
                <Typography variant="body2" color="primary.main" sx={{ fontWeight: 600 }}>full</Typography>
                <Typography variant="body2" color="text.secondary">
                  Single pass with RandomizedSearchCV (RF) or early stopping (XGBoost). Use <code>--sample 0.5</code> to limit RAM.
                </Typography>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>Target Variable</Typography>
                <Typography variant="body2" sx={{ fontFamily: '"IBM Plex Mono", monospace', color: 'primary.main', mb: 1, fontSize: 12 }}>
                  Avg_Mdcr_Alowd_Amt
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
                  Medicare allowed amount per service, what Medicare pays the provider (Stage 1).
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ fontSize: 12 }}>
                  Leakage removed: Avg_Mdcr_Pymt_Amt and Avg_Mdcr_Stdzd_Amt are derived from the target and excluded from all models.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  );
}
