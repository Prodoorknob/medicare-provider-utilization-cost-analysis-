'use client';

import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Button from '@mui/material/Button';
import Link from 'next/link';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import LocalHospitalIcon from '@mui/icons-material/LocalHospital';
import BarChartIcon from '@mui/icons-material/BarChart';
import PsychologyIcon from '@mui/icons-material/Psychology';

const STATS = [
  { label: 'Records Analyzed', value: '103M+', icon: <BarChartIcon fontSize="large" /> },
  { label: 'States & Territories', value: '63', icon: <LocalHospitalIcon fontSize="large" /> },
  { label: 'Model Families', value: '4', icon: <PsychologyIcon fontSize="large" /> },
  { label: 'Forecast Through', value: '2026', icon: <TrendingUpIcon fontSize="large" /> },
];

const FINDINGS = [
  { title: 'R\u00B2 = 0.886', subtitle: 'LSTM achieves the best validation accuracy for temporal cost prediction (2022-2023 holdout).' },
  { title: '62% Explained by Charge', subtitle: 'Submitted charge amount is the single strongest predictor of Medicare allowed amounts.' },
  { title: 'P10-P90 OOP Range', subtitle: 'Quantile regression provides low/typical/high out-of-pocket estimates, not just a single number.' },
];

export default function HomePage() {
  return (
    <Box>
      <Box sx={{ textAlign: 'center', py: { xs: 4, md: 6 } }}>
        <Typography variant="h3" gutterBottom>
          Medicare Provider Cost Analysis
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ maxWidth: 700, mx: 'auto', mb: 4 }}>
          Explore Medicare allowed amounts, predict patient out-of-pocket costs, and view LSTM-powered cost forecasts across 103M+ provider service records (2013-2023).
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
          <Button variant="contained" size="large" component={Link} href="/estimator" endIcon={<ArrowForwardIcon />}>
            Cost Estimator
          </Button>
          <Button variant="outlined" size="large" component={Link} href="/forecast">
            View Forecasts
          </Button>
        </Box>
      </Box>

      <Grid container spacing={3} sx={{ mb: 6 }}>
        {STATS.map((s) => (
          <Grid key={s.label} size={{ xs: 6, md: 3 }}>
            <Card sx={{ textAlign: 'center', py: 2 }}>
              <CardContent>
                <Box sx={{ color: 'primary.main', mb: 1 }}>{s.icon}</Box>
                <Typography variant="h4">{s.value}</Typography>
                <Typography variant="body2" color="text.secondary">{s.label}</Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Typography variant="h5" gutterBottom>Two-Stage Pipeline</Typography>
      <Grid container spacing={3} sx={{ mb: 6 }}>
        <Grid size={{ xs: 12, md: 6 }}>
          <Card sx={{ height: '100%', borderLeft: '4px solid', borderColor: 'primary.main' }}>
            <CardContent>
              <Typography variant="h6" color="primary">Stage 1: Medicare Allowed Amount</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Given provider specialty, state, procedure category, and service details, predict what Medicare allows for a service. Trained on 103M+ real CMS records using Random Forest, XGBoost, and LSTM models.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid size={{ xs: 12, md: 6 }}>
          <Card sx={{ height: '100%', borderLeft: '4px solid', borderColor: 'secondary.main' }}>
            <CardContent>
              <Typography variant="h6" color="secondary">Stage 2: Patient Out-of-Pocket</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Using the Stage 1 allowed amount plus beneficiary demographics (age, income, insurance status), estimate patient OOP costs at the 10th, 50th, and 90th percentiles via quantile XGBoost.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Typography variant="h5" gutterBottom>Key Findings</Typography>
      <Grid container spacing={3}>
        {FINDINGS.map((f) => (
          <Grid key={f.title} size={{ xs: 12, md: 4 }}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Typography variant="h5" color="primary" gutterBottom>{f.title}</Typography>
                <Typography variant="body2" color="text.secondary">{f.subtitle}</Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}
