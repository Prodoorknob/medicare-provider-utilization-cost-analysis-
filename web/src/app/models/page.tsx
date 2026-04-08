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
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { getModelMetrics, getFeatureImportances } from '@/lib/queries';
import { formatDollars } from '@/lib/formatters';
import { FEATURE_DISPLAY_NAMES } from '@/lib/constants';
import type { ModelMetric, FeatureImportance } from '@/lib/types';

interface ModelRow {
  name: string;
  mae: number | null;
  rmse: number | null;
  r2: number | null;
}

const METHODOLOGIES = [
  { name: 'Random Forest', desc: '625 trees trained with warm_start batch mode across Census regions. RandomizedSearchCV for hyperparameter optimization. Best overall Stage 1 performer.' },
  { name: 'XGBoost', desc: 'Gradient-boosted trees with CUDA acceleration. Incremental training by Census region (125 rounds/region). Early stopping with validation set.' },
  { name: 'LSTM', desc: 'PyTorch 2-layer LSTM with static embeddings for specialty/state/bucket. Temporal train/val split (train: 2013-2021, val: 2022-2023). MC Dropout for confidence-bounded 2024-2026 forecasts.' },
  { name: 'GLM (SGD)', desc: 'Stochastic Gradient Descent with Huber loss and partial_fit streaming. Diverged on national data - needs hyperparameter tuning.' },
  { name: 'Quantile XGBoost (OOP)', desc: 'Stage 2 model: 3 separate XGBoost boosters with reg:quantileerror objective (P10/P50/P90). Predicts patient out-of-pocket from allowed amount + demographics.' },
];

export default function ModelsPage() {
  const [metrics, setMetrics] = useState<ModelMetric[]>([]);
  const [importances, setImportances] = useState<FeatureImportance[]>([]);

  useEffect(() => {
    Promise.all([getModelMetrics(), getFeatureImportances()]).then(([m, fi]) => {
      setMetrics(m);
      setImportances(fi);
    });
  }, []);

  const stage1Models: ModelRow[] = ['Random Forest', 'XGBoost', 'LSTM', 'GLM (SGD)'].map((name) => {
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
    .map((x) => ({
      name: FEATURE_DISPLAY_NAMES[x.feature_name] || x.feature_name,
      importance: x.importance,
    }));

  return (
    <Box>
      <Typography variant="h4" gutterBottom>Model Comparison</Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Performance metrics across 4 model families trained on 103M+ Medicare records.
      </Typography>

      {/* Stage 1 Table */}
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
                  <TableCell align="right"><strong>Test R&sup2;</strong></TableCell>
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

      {/* Feature Importance */}
      {fiChart.length > 0 && (
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>Feature Importance (Random Forest)</Typography>
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={fiChart} layout="vertical" margin={{ left: 140, right: 30 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(v) => (Number(v) * 100).toFixed(1) + '%'} />
                <Bar dataKey="importance" fill="#1565c0" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Methodology */}
      <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>Methodology</Typography>
      {METHODOLOGIES.map((m) => (
        <Accordion key={m.name}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography sx={{ fontWeight: 600 }}>{m.name}</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Typography variant="body2" color="text.secondary">{m.desc}</Typography>
          </AccordionDetails>
        </Accordion>
      ))}
    </Box>
  );
}
