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
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import TableSortLabel from '@mui/material/TableSortLabel';
import TextField from '@mui/material/TextField';
import Image from 'next/image';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { getStateSummary, getModelMetrics, getFeatureImportances } from '@/lib/queries';
import { formatDollars, formatNumber } from '@/lib/formatters';
import { FEATURE_DISPLAY_NAMES } from '@/lib/constants';
import type { StateSummary, ModelMetric, FeatureImportance } from '@/lib/types';

type SortKey = 'state_abbrev' | 'mean_allowed' | 'median_allowed' | 'n_records';

const METHODOLOGIES = [
  { name: 'Random Forest', desc: '625 trees trained with warm_start batch mode across Census regions. RandomizedSearchCV for hyperparameter optimization. Best overall Stage 1 performer.' },
  { name: 'XGBoost', desc: 'Gradient-boosted trees with CUDA acceleration. Incremental training by Census region (125 rounds/region). Early stopping with validation set.' },
  { name: 'LSTM', desc: 'PyTorch 2-layer LSTM with static embeddings for specialty/state/bucket. Temporal train/val split (train: 2013–2021, val: 2022–2023). MC Dropout for confidence-bounded 2024–2026 forecasts.' },
  { name: 'GLM (SGD)', desc: 'Stochastic Gradient Descent with Huber loss and partial_fit streaming. Diverged on national data — needs hyperparameter tuning.' },
  { name: 'Quantile XGBoost (OOP)', desc: 'Stage 2 model: 3 separate XGBoost boosters with reg:quantileerror objective (P10/P50/P90). Predicts patient out-of-pocket from allowed amount + demographics.' },
];

export default function AboutPage() {
  const [tab, setTab] = useState(0);

  // Data state
  const [states, setStates] = useState<StateSummary[]>([]);
  const [metrics, setMetrics] = useState<ModelMetric[]>([]);
  const [importances, setImportances] = useState<FeatureImportance[]>([]);
  const [sortKey, setSortKey] = useState<SortKey>('mean_allowed');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [filter, setFilter] = useState('');

  useEffect(() => {
    Promise.all([getStateSummary(), getModelMetrics(), getFeatureImportances()]).then(
      ([s, m, fi]) => { setStates(s); setMetrics(m); setImportances(fi); }
    );
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

  const stage1Models = ['Random Forest', 'XGBoost', 'LSTM', 'GLM (SGD)'].map((name) => {
    const m = metrics.filter((x) => x.model_name === name && x.stage === 1);
    return {
      name,
      mae: m.find((x) => x.metric_name === 'test_mae')?.metric_value ?? null,
      rmse: m.find((x) => x.metric_name === 'test_rmse')?.metric_value ?? null,
      r2: m.find((x) => x.metric_name === 'test_r2')?.metric_value ?? null,
    };
  });

  const fiChart = importances
    .filter((x) => x.model_name === 'Random Forest')
    .sort((a, b) => b.importance - a.importance)
    .map((x) => ({ name: FEATURE_DISPLAY_NAMES[x.feature_name] || x.feature_name, importance: x.importance }));

  return (
    <Box>
      <Box sx={{ pb: 3, mb: 0, borderBottom: '1px solid', borderColor: 'divider' }}>
        <Typography variant="h4" sx={{ fontWeight: 800, letterSpacing: '-0.02em' }} gutterBottom>
          About & Methodology
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

      {/* ── TAB 0: Overview ── */}
      {tab === 0 && (
        <Grid container spacing={4}>
          <Grid size={{ xs: 12, md: 8 }}>
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>Overview</Typography>
                <Typography variant="body1" paragraph>
                  An end-to-end data science pipeline predicting Medicare provider costs and patient out-of-pocket expenses. Processes over 103 million provider-service records from the CMS Medicare Physician &amp; Other Practitioners dataset spanning 2013–2023.
                </Typography>
                <Typography variant="body1">
                  The project implements a two-stage prediction architecture: Stage 1 predicts what Medicare allows for a service; Stage 2 estimates what the patient pays out of pocket based on demographic and insurance profile.
                </Typography>
              </CardContent>
            </Card>

            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>Data Sources</Typography>
                <Typography variant="body1" paragraph>
                  <strong>CMS Medicare Physician &amp; Other Practitioners (2013–2023)</strong> — Provider-level utilization and payment data for Part B services. Over 10 million records per year across all 50 states and territories.
                </Typography>
                <Typography variant="body1" paragraph>
                  <strong>CMS Medicare Current Beneficiary Survey (MCBS)</strong> — Public Use Files for beneficiary demographics, insurance coverage, and cost data. Used for Stage 2 patient cost modeling.
                </Typography>
                <Typography variant="body1">
                  <strong>CMS Provider Summary (by Provider)</strong> — NPI-level HCC risk scores (Bene_Avg_Risk_Scre) for beneficiary health burden adjustment.
                </Typography>
              </CardContent>
            </Card>

            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom color="primary">Stage 1 — Medicare Allowed Amount</Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Random Forest (R²=0.884), XGBoost (R²=0.833), LSTM (R²=0.886, best), GLM/SGD
                </Typography>
                <Divider sx={{ my: 1.5 }} />
                <Typography variant="h6" gutterBottom color="secondary" sx={{ mt: 1.5 }}>Stage 2 — Patient Out-of-Pocket</Typography>
                <Typography variant="body2" color="text.secondary">
                  Quantile XGBoost (P10/P50/P90) trained on synthetic MCBS-derived beneficiary data segmented by region, age, income, and insurance status.
                </Typography>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom color="warning.main">Disclaimer</Typography>
                <Box sx={{ bgcolor: '#FDF4EA', borderLeft: '3px solid #B8763A', borderRadius: 1, p: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    This is an academic research project. Estimates are based on aggregate historical data and should not be used for medical billing decisions. Stage 2 out-of-pocket estimates use synthetic beneficiary data modeled after MCBS distributions. Actual patient costs depend on specific plan details, deductibles, and coverage terms not captured in this model.
                  </Typography>
                </Box>
              </CardContent>
            </Card>
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
                <Typography variant="body2" paragraph>
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
                <Typography variant="body2">Python · pandas · PyArrow · scikit-learn</Typography>
                <Typography variant="body2">XGBoost (CUDA) · PyTorch (LSTM)</Typography>
                <Typography variant="body2">MLflow · Databricks</Typography>
                <Divider sx={{ my: 1.5 }} />
                <Typography variant="body2">Next.js · Material UI · Recharts</Typography>
                <Typography variant="body2">Supabase (PostgreSQL) · Vercel</Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* ── TAB 1: Data ── */}
      {tab === 1 && (
        <Box>
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
      )}

      {/* ── TAB 2: Models ── */}
      {tab === 2 && (
        <Box>
          <Card sx={{ mb: 4 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>Stage 1 — Medicare Allowed Amount</Typography>
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
                    {stage1Models.map((m) => {
                      const bestR2 = Math.max(...stage1Models.filter(x => x.r2 !== null && x.r2 > 0).map(x => x.r2!));
                      const isBest = m.r2 === bestR2;
                      return (
                        <TableRow key={m.name} sx={isBest ? { bgcolor: 'primary.50' } : {}}>
                          <TableCell sx={isBest ? { fontWeight: 700 } : {}}>{m.name}</TableCell>
                          <TableCell align="right">{m.mae != null ? formatDollars(m.mae) : 'N/A'}</TableCell>
                          <TableCell align="right">{m.rmse != null ? formatDollars(m.rmse) : 'N/A'}</TableCell>
                          <TableCell align="right" sx={isBest ? { fontWeight: 700, color: 'primary.main' } : {}}>
                            {m.r2 != null ? m.r2.toFixed(4) : 'N/A'}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>

          {fiChart.length > 0 && (
            <Card sx={{ mb: 4 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>Feature Importance — Random Forest</Typography>
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart data={fiChart} layout="vertical" margin={{ left: 140, right: 30 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 12 }} />
                    <Tooltip formatter={(v) => (Number(v) * 100).toFixed(1) + '%'} />
                    <Bar dataKey="importance" fill="#0F6E8C" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>Methodology</Typography>
          {METHODOLOGIES.map((m) => (
            <Accordion key={m.name} sx={{ mb: 1 }}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography fontWeight={600}>{m.name}</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography variant="body2" color="text.secondary">{m.desc}</Typography>
              </AccordionDetails>
            </Accordion>
          ))}
        </Box>
      )}

      {/* ── TAB 3: Pipeline ── */}
      {tab === 3 && (
        <Grid container spacing={3}>
          <Grid size={{ xs: 12, md: 8 }}>
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>Medallion Architecture</Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Bronze → Silver → Gold pipeline with two execution modes: PySpark + Delta Lake on Databricks for production, pandas + pyarrow locally for development. All models log to Databricks MLflow.
                </Typography>
                <Box component="pre" sx={{
                  fontFamily: '"IBM Plex Mono", monospace', fontSize: 12,
                  bgcolor: 'grey.50', borderRadius: 2, p: 2, overflowX: 'auto',
                  color: 'text.secondary', lineHeight: 1.8, whiteSpace: 'pre-wrap',
                }}>
{`CMS API (by Provider & Service)
  → pull_medicare_data.py
  → partition_medicare_data.py  (injects year)
  → csv_to_parquet.py

Bronze  →  Silver  →  Gold  →  EDA + Modeling
         (typed,    (features +      ↓
          cleaned)   encoding)   LSTM sequences

CMS MCBS PUF
  → 06_mcbs_bronze → 07_mcbs_silver
  → 08_mcbs_crosswalk
         ↓
  generate_synthetic_mcbs.py
  → Stage 2 OOP training data`}
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
                    ['hcpcs_bucket', 'Clinical category (Anesthesia → HCPCS II)'],
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
                        <Typography variant="body2" color="text.secondary" fontSize={12}>{desc}</Typography>
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
                  ].map(([region, states]) => (
                    <Grid key={region} size={{ xs: 12, sm: 6 }}>
                      <Box sx={{ p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
                        <Typography variant="body2" fontWeight={600} color="primary.main">{region}</Typography>
                        <Typography variant="body2" color="text.secondary" fontSize={11} sx={{ fontFamily: '"IBM Plex Mono", monospace' }}>{states}</Typography>
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
                <Typography variant="body2" color="text.secondary" paragraph>
                  2-layer PyTorch LSTM with static embeddings for specialty, state, and service bucket.
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  <strong>Train:</strong> 2013–2021 · <strong>Val:</strong> 2022–2023
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  MC Dropout produces confidence-bounded (P10/P90) forecasts for 2024–2026 by specialty.
                </Typography>
              </CardContent>
            </Card>

            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>Training Modes</Typography>
                <Typography variant="body2" fontWeight={600} color="primary.main">batch (default)</Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Incremental by Census region. Memory-efficient. XGBoost: 125 rounds/region. RF: warm_start adds 125 trees/region.
                </Typography>
                <Typography variant="body2" fontWeight={600} color="primary.main">full</Typography>
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
                <Typography variant="body2" color="text.secondary" paragraph>
                  Medicare allowed amount per service — what Medicare pays the provider (Stage 1).
                </Typography>
                <Typography variant="body2" color="text.secondary" fontSize={12}>
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
