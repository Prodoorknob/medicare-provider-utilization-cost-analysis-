'use client';

import { useState, useEffect } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Autocomplete from '@mui/material/Autocomplete';
import TextField from '@mui/material/TextField';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import ToggleButton from '@mui/material/ToggleButton';
import Switch from '@mui/material/Switch';
import FormControlLabel from '@mui/material/FormControlLabel';
import Button from '@mui/material/Button';
import Alert from '@mui/material/Alert';
import Divider from '@mui/material/Divider';
import CircularProgress from '@mui/material/CircularProgress';
import Chip from '@mui/material/Chip';
import { getLabels, getStage1Estimate, getStage2Estimate } from '@/lib/queries';
import { STATE_TO_REGION, CENSUS_REGION_NAMES, HCPCS_BUCKET_NAMES, AGE_GROUP_LABELS, INCOME_LABELS } from '@/lib/constants';
import { formatDollars, formatNumber } from '@/lib/formatters';
import type { LookupLabel, Stage1Estimate, Stage2Estimate } from '@/lib/types';

export default function EstimatorPage() {
  const [specialties, setSpecialties] = useState<LookupLabel[]>([]);
  const [states, setStates] = useState<LookupLabel[]>([]);
  const [selectedSpecialty, setSelectedSpecialty] = useState<LookupLabel | null>(null);
  const [selectedState, setSelectedState] = useState<LookupLabel | null>(null);
  const [bucket, setBucket] = useState<number>(4);
  const [pos, setPos] = useState<number>(0);

  const [ageGroup, setAgeGroup] = useState<number>(2);
  const [income, setIncome] = useState<number>(1);
  const [dual, setDual] = useState(false);
  const [supplemental, setSupplemental] = useState(false);

  const [s1Result, setS1Result] = useState<Stage1Estimate | null>(null);
  const [s2Result, setS2Result] = useState<Stage2Estimate | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getLabels('specialty'), getLabels('state')]).then(([sp, st]) => {
      setSpecialties(sp);
      setStates(st);
    });
  }, []);

  const censusRegion = selectedState
    ? STATE_TO_REGION[selectedState.label] ?? null
    : null;

  const handleEstimate = async () => {
    if (!selectedSpecialty || !selectedState) return;
    setLoading(true);
    setError(null);
    setS1Result(null);
    setS2Result(null);

    try {
      const s1 = await getStage1Estimate(selectedSpecialty.idx, bucket, selectedState.idx, pos);
      setS1Result(s1);

      if (s1 && censusRegion) {
        const s2 = await getStage2Estimate(
          selectedSpecialty.idx, bucket, censusRegion,
          dual ? 1 : 0, supplemental ? 1 : 0, ageGroup, income
        );
        setS2Result(s2);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch estimates');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>Cost Estimator</Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Select provider and patient details to estimate Medicare allowed amounts and out-of-pocket costs.
      </Typography>

      <Grid container spacing={4}>
        {/* Left: Forms */}
        <Grid size={{ xs: 12, md: 5 }}>
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">Provider Details</Typography>

              <Autocomplete
                options={specialties}
                getOptionLabel={(o) => o.label}
                value={selectedSpecialty}
                onChange={(_, v) => setSelectedSpecialty(v)}
                renderInput={(params) => <TextField {...params} label="Provider Specialty" size="small" />}
                sx={{ mb: 2 }}
              />

              <Autocomplete
                options={states}
                getOptionLabel={(o) => o.label}
                value={selectedState}
                onChange={(_, v) => setSelectedState(v)}
                renderInput={(params) => <TextField {...params} label="State" size="small" />}
                sx={{ mb: 2 }}
              />

              <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                <InputLabel>Service Category</InputLabel>
                <Select value={bucket} label="Service Category" onChange={(e) => setBucket(Number(e.target.value))}>
                  {Object.entries(HCPCS_BUCKET_NAMES).map(([k, v]) => (
                    <MenuItem key={k} value={Number(k)}>{v}</MenuItem>
                  ))}
                </Select>
              </FormControl>

              <Typography variant="body2" sx={{ mb: 1 }}>Place of Service</Typography>
              <ToggleButtonGroup value={pos} exclusive onChange={(_, v) => v !== null && setPos(v)} size="small" fullWidth>
                <ToggleButton value={0}>Office</ToggleButton>
                <ToggleButton value={1}>Facility</ToggleButton>
              </ToggleButtonGroup>
            </CardContent>
          </Card>

          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="secondary">Patient Details</Typography>

              {censusRegion && (
                <Chip label={`Region: ${CENSUS_REGION_NAMES[censusRegion]}`} color="secondary" variant="outlined" sx={{ mb: 2 }} />
              )}

              <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                <InputLabel>Age Group</InputLabel>
                <Select value={ageGroup} label="Age Group" onChange={(e) => setAgeGroup(Number(e.target.value))}>
                  {Object.entries(AGE_GROUP_LABELS).map(([k, v]) => (
                    <MenuItem key={k} value={Number(k)}>{v}</MenuItem>
                  ))}
                </Select>
              </FormControl>

              <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                <InputLabel>Income Bracket</InputLabel>
                <Select value={income} label="Income Bracket" onChange={(e) => setIncome(Number(e.target.value))}>
                  {Object.entries(INCOME_LABELS).map(([k, v]) => (
                    <MenuItem key={k} value={Number(k)}>{v}</MenuItem>
                  ))}
                </Select>
              </FormControl>

              <FormControlLabel control={<Switch checked={dual} onChange={(e) => setDual(e.target.checked)} />} label="Dual Eligible (Medicare + Medicaid)" />
              <FormControlLabel control={<Switch checked={supplemental} onChange={(e) => setSupplemental(e.target.checked)} />} label="Has Supplemental Insurance" />
            </CardContent>
          </Card>

          <Button variant="contained" fullWidth size="large" onClick={handleEstimate}
            disabled={!selectedSpecialty || !selectedState || loading}>
            {loading ? <CircularProgress size={24} color="inherit" /> : 'Estimate Costs'}
          </Button>
        </Grid>

        {/* Right: Results */}
        <Grid size={{ xs: 12, md: 7 }}>
          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

          {s1Result && (
            <Card sx={{ mb: 3, borderLeft: '4px solid', borderColor: 'primary.main' }}>
              <CardContent>
                <Typography variant="h6" color="primary" gutterBottom>Medicare Allowed Amount</Typography>
                <Typography variant="h3" sx={{ mb: 1 }}>{formatDollars(s1Result.mean_allowed)}</Typography>
                <Typography variant="body2" color="text.secondary">
                  Average amount Medicare allows for this service profile
                </Typography>
                <Divider sx={{ my: 2 }} />
                <Grid container spacing={2}>
                  <Grid size={3}>
                    <Typography variant="body2" color="text.secondary">Low (P10)</Typography>
                    <Typography variant="h6">{formatDollars(s1Result.p10_allowed)}</Typography>
                  </Grid>
                  <Grid size={3}>
                    <Typography variant="body2" color="text.secondary">Median</Typography>
                    <Typography variant="h6">{formatDollars(s1Result.median_allowed)}</Typography>
                  </Grid>
                  <Grid size={3}>
                    <Typography variant="body2" color="text.secondary">High (P90)</Typography>
                    <Typography variant="h6">{formatDollars(s1Result.p90_allowed)}</Typography>
                  </Grid>
                  <Grid size={3}>
                    <Typography variant="body2" color="text.secondary">Records</Typography>
                    <Typography variant="h6">{formatNumber(s1Result.n_records)}</Typography>
                  </Grid>
                </Grid>
                {s1Result.mean_charge != null && (
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    Avg. submitted charge: {formatDollars(s1Result.mean_charge)} &middot;
                    Avg. risk score: {s1Result.mean_risk_score?.toFixed(2) ?? 'N/A'}
                  </Typography>
                )}
              </CardContent>
            </Card>
          )}

          {s2Result && (
            <Card sx={{ borderLeft: '4px solid', borderColor: 'secondary.main' }}>
              <CardContent>
                <Typography variant="h6" color="secondary" gutterBottom>Estimated Out-of-Pocket Cost</Typography>
                <Grid container spacing={3} sx={{ textAlign: 'center', mt: 1 }}>
                  <Grid size={4}>
                    <Typography variant="body2" color="text.secondary">Low (P10)</Typography>
                    <Typography variant="h4" color="secondary.light">{formatDollars(s2Result.oop_p10)}</Typography>
                  </Grid>
                  <Grid size={4}>
                    <Typography variant="body2" color="text.secondary">Typical (P50)</Typography>
                    <Typography variant="h4" color="secondary.main" sx={{ fontWeight: 700 }}>{formatDollars(s2Result.oop_p50)}</Typography>
                  </Grid>
                  <Grid size={4}>
                    <Typography variant="body2" color="text.secondary">High (P90)</Typography>
                    <Typography variant="h4" color="secondary.dark">{formatDollars(s2Result.oop_p90)}</Typography>
                  </Grid>
                </Grid>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
                  Based on {formatNumber(s2Result.n_records)} records for this patient profile
                </Typography>
                <Alert severity="info" sx={{ mt: 2 }}>
                  OOP estimates use synthetic beneficiary data modeled after MCBS distributions. Actual costs depend on specific plan details, deductibles, and coverage terms.
                </Alert>
              </CardContent>
            </Card>
          )}

          {!s1Result && !loading && !error && (
            <Card sx={{ py: 8, textAlign: 'center' }}>
              <CardContent>
                <Typography variant="h6" color="text.secondary">
                  Select provider and patient details, then click Estimate Costs
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
}
