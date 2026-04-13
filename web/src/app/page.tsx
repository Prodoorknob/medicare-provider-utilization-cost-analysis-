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
import { getLabels, getFullPrediction } from '@/lib/queries';
import { STATE_TO_REGION, CENSUS_REGION_NAMES, HCPCS_BUCKET_NAMES, AGE_GROUP_LABELS, INCOME_LABELS } from '@/lib/constants';
import { formatDollars } from '@/lib/formatters';
import { PALETTE } from '@/app/theme';
import type { LookupLabel, FullPredictionResponse } from '@/lib/types';

export default function HomePage() {
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

  const [result, setResult] = useState<FullPredictionResponse | null>(null);
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
    setResult(null);

    try {
      const res = await getFullPrediction({
        provider_type: selectedSpecialty.label,
        state: selectedState.label,
        hcpcs_bucket: bucket,
        place_of_service: pos,
        age: ageGroup === 1 ? 60 : ageGroup === 2 ? 70 : 80,
        sex: 0,
        income: income,
        chronic_count: 2,
        dual_eligible: dual ? 1 : 0,
        has_supplemental: supplemental ? 1 : 0,
      });
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to get prediction');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      {/* Slim masthead */}
      <Box sx={{ pb: 3, mb: 4, borderBottom: '1px solid', borderColor: 'divider' }}>
        <Typography variant="h4" sx={{ fontWeight: 800, letterSpacing: '-0.02em' }} gutterBottom>
          Medicare Cost Estimator
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Predict what Medicare allows for a service and estimate patient out-of-pocket costs, powered by real-time ML inference on 103M+ CMS records.
        </Typography>
      </Box>

      <Grid container spacing={4}>
        {/* Left: Forms */}
        <Grid size={{ xs: 12, md: 5 }}>
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography
                variant="overline"
                color="primary"
                sx={{ display: 'block', mb: 2, letterSpacing: '0.08em', fontWeight: 600 }}
              >
                Stage 1 &middot; Provider Details
              </Typography>

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

              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>Place of Service</Typography>
              <ToggleButtonGroup value={pos} exclusive onChange={(_, v) => v !== null && setPos(v)} size="small" fullWidth>
                <ToggleButton value={0}>Office</ToggleButton>
                <ToggleButton value={1}>Facility</ToggleButton>
              </ToggleButtonGroup>
            </CardContent>
          </Card>

          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography
                variant="overline"
                color="secondary"
                sx={{ display: 'block', mb: 2, letterSpacing: '0.08em', fontWeight: 600 }}
              >
                Stage 2 &middot; Patient Details
              </Typography>

              {censusRegion && (
                <Chip label={`Region: ${CENSUS_REGION_NAMES[censusRegion]}`} color="secondary" variant="outlined" size="small" sx={{ mb: 2 }} />
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

              <FormControlLabel
                control={<Switch checked={dual} onChange={(e) => setDual(e.target.checked)} color="secondary" />}
                label={<Typography variant="body2">Dual Eligible (Medicare + Medicaid)</Typography>}
              />
              <FormControlLabel
                control={<Switch checked={supplemental} onChange={(e) => setSupplemental(e.target.checked)} color="secondary" />}
                label={<Typography variant="body2">Has Supplemental Insurance</Typography>}
              />
            </CardContent>
          </Card>

          <Button
            variant="contained"
            fullWidth
            size="large"
            onClick={handleEstimate}
            disabled={!selectedSpecialty || !selectedState || loading}
          >
            {loading ? <CircularProgress size={22} color="inherit" /> : 'Estimate Costs'}
          </Button>
        </Grid>

        {/* Right: Results */}
        <Grid size={{ xs: 12, md: 7 }}>
          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

          {result && (
            <>
              {/* Stage 1 result */}
              <Card sx={{ mb: 3, borderLeft: '4px solid', borderColor: 'primary.main' }}>
                <CardContent>
                  <Typography
                    variant="overline"
                    color="primary"
                    sx={{ display: 'block', mb: 1, letterSpacing: '0.08em', fontWeight: 600 }}
                  >
                    Stage 1 &middot; Medicare Allowed Amount
                  </Typography>
                  <Typography
                    variant="h3"
                    color="primary.main"
                    sx={{ fontFamily: '"IBM Plex Mono", monospace', fontWeight: 700, mb: 0.5 }}
                  >
                    {formatDollars(result.stage1.predicted_allowed_amount)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Predicted amount Medicare allows for this service (LightGBM, R&sup2; = 0.943)
                  </Typography>
                  <Divider sx={{ my: 2 }} />
                  <Grid container spacing={2}>
                    {[
                      { label: 'Specialty', value: result.stage1.provider_type },
                      { label: 'Category', value: result.stage1.hcpcs_bucket_name },
                      { label: 'Setting', value: result.stage1.place_of_service === 1 ? 'Facility' : 'Office' },
                    ].map((pill) => (
                      <Grid key={pill.label} size={4}>
                        <Box sx={{ bgcolor: PALETTE.primarySubtle, borderRadius: 1, p: 1.5, textAlign: 'center' }}>
                          <Typography variant="body2" color="text.secondary" sx={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.03em' }}>
                            {pill.label}
                          </Typography>
                          <Typography variant="body1" color="primary.main" sx={{ fontWeight: 600, mt: 0.5 }}>
                            {pill.value}
                          </Typography>
                        </Box>
                      </Grid>
                    ))}
                  </Grid>
                </CardContent>
              </Card>

              {/* Stage 2 result */}
              <Card sx={{ borderLeft: '4px solid', borderColor: 'secondary.main' }}>
                <CardContent>
                  <Typography
                    variant="overline"
                    color="secondary"
                    sx={{ display: 'block', mb: 1, letterSpacing: '0.08em', fontWeight: 600 }}
                  >
                    Stage 2 &middot; Patient Out-of-Pocket
                  </Typography>

                  {/* Percentile bar */}
                  <Grid
                    container
                    spacing={0}
                    sx={{
                      textAlign: 'center', mt: 1, mb: 2,
                      borderRadius: 2, overflow: 'hidden',
                      border: '1px solid', borderColor: 'divider',
                    }}
                  >
                    <Grid size={4} sx={{ p: 2, borderRight: '1px solid', borderColor: 'divider' }}>
                      <Typography variant="body2" color="text.disabled" sx={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                        Best Case &middot; P10
                      </Typography>
                      <Typography
                        variant="h5"
                        sx={{ fontFamily: '"IBM Plex Mono", monospace', mt: 0.5, color: SECONDARY_LIGHT }}
                      >
                        {formatDollars(result.stage2.oop_p10)}
                      </Typography>
                    </Grid>
                    <Grid size={4} sx={{ p: 2, borderRight: '1px solid', borderColor: 'divider', bgcolor: PALETTE.secondarySubtle }}>
                      <Typography variant="body2" color="text.disabled" sx={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                        Typical &middot; P50
                      </Typography>
                      <Typography
                        variant="h4"
                        color="secondary.main"
                        sx={{ fontFamily: '"IBM Plex Mono", monospace', fontWeight: 700, mt: 0.5 }}
                      >
                        {formatDollars(result.stage2.oop_p50)}
                      </Typography>
                    </Grid>
                    <Grid size={4} sx={{ p: 2 }}>
                      <Typography variant="body2" color="text.disabled" sx={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                        High End &middot; P90
                      </Typography>
                      <Typography
                        variant="h5"
                        color="secondary.dark"
                        sx={{ fontFamily: '"IBM Plex Mono", monospace', mt: 0.5 }}
                      >
                        {formatDollars(result.stage2.oop_p90)}
                      </Typography>
                    </Grid>
                  </Grid>

                  <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', mb: 2 }}>
                    Region: {result.stage2.census_region_name}
                  </Typography>

                  {/* Disclaimer callout */}
                  <Box sx={{ bgcolor: PALETTE.accentBg, borderLeft: `3px solid ${PALETTE.accent}`, borderRadius: 1, p: 1.5 }}>
                    <Typography variant="body2" color="text.secondary" sx={{ fontSize: 12, lineHeight: 1.6 }}>
                      OOP estimates use synthetic beneficiary data modeled after MCBS distributions. Actual costs depend on specific plan details, deductibles, and coverage terms.
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </>
          )}

          {!result && !loading && !error && (
            <Card sx={{ py: 10, textAlign: 'center', border: '1px dashed', borderColor: 'divider', boxShadow: 'none', bgcolor: 'transparent' }}>
              <CardContent>
                <Typography variant="body1" color="text.disabled">
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

// Need secondary light accessible outside theme for inline sx
const SECONDARY_LIGHT = '#1CA082';
