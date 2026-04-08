import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Grid from '@mui/material/Grid';
import Link from '@mui/material/Link';
import Divider from '@mui/material/Divider';

export default function AboutPage() {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>About This Project</Typography>

      <Grid container spacing={4}>
        <Grid size={{ xs: 12, md: 8 }}>
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>Overview</Typography>
              <Typography variant="body1" component="p">
                This is an end-to-end data science pipeline that predicts Medicare provider costs and patient out-of-pocket expenses. It processes over 103 million provider-service records from the CMS Medicare Physician & Other Practitioners dataset spanning 2013-2023.
              </Typography>
              <Typography variant="body1" component="p">
                The project implements a two-stage prediction architecture: Stage 1 predicts what Medicare allows for a service, and Stage 2 estimates what the patient pays out of pocket based on their demographic and insurance profile.
              </Typography>
            </CardContent>
          </Card>

          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>Data Sources</Typography>
              <Typography variant="body1" component="p">
                <strong>CMS Medicare Physician & Other Practitioners (2013-2023)</strong> — Provider-level utilization and payment data for Part B services. Over 10 million records per year across all 50 states and territories.
              </Typography>
              <Typography variant="body1" component="p">
                <strong>CMS Medicare Current Beneficiary Survey (MCBS)</strong> — Public Use Files for beneficiary demographics, insurance coverage, and cost data. Used for Stage 2 patient cost modeling.
              </Typography>
              <Typography variant="body1" component="p">
                <strong>CMS Provider Summary (by Provider)</strong> — NPI-level HCC risk scores (Bene_Avg_Risk_Scre) for beneficiary health burden adjustment.
              </Typography>
            </CardContent>
          </Card>

          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>Pipeline Architecture</Typography>
              <Typography variant="body1" component="p">
                Medallion architecture (Bronze → Silver → Gold) with dual execution modes: PySpark + Delta Lake on Databricks for production, pandas + pyarrow locally for development. All models log to Databricks MLflow for experiment tracking.
              </Typography>
              <Typography variant="body2" component="div">
                <strong>Stage 1 Models:</strong> Random Forest (R&sup2;=0.884), XGBoost (R&sup2;=0.833), GLM/SGD, LSTM (R&sup2;=0.886)
              </Typography>
              <Typography variant="body2" component="div" sx={{ mt: 1 }}>
                <strong>Stage 2 Model:</strong> Quantile XGBoost (P10/P50/P90) for out-of-pocket cost intervals
              </Typography>
              <Typography variant="body2" component="div" sx={{ mt: 1 }}>
                <strong>LSTM Forecasting:</strong> MC Dropout for 2024-2026 confidence-bounded cost projections by specialty
              </Typography>
            </CardContent>
          </Card>

          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom color="error">Disclaimer</Typography>
              <Typography variant="body2" color="text.secondary">
                This is an academic research project. Estimates are based on aggregate historical data and should not be used for medical billing decisions. Stage 2 out-of-pocket estimates use synthetic beneficiary data modeled after MCBS distributions. Actual patient costs depend on specific plan details, deductibles, and coverage terms not captured in this model.
              </Typography>
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
              <Typography variant="body2" component="div">
                <Link href="https://github.com/Prodoorknob/medicare-provider-utilization-cost-analysis-" target="_blank" rel="noopener">
                  GitHub Repository
                </Link>
              </Typography>
              <Typography variant="body2" component="div" sx={{ mt: 1 }}>
                <Link href="https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners" target="_blank" rel="noopener">
                  CMS Data Source
                </Link>
              </Typography>
            </CardContent>
          </Card>

          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Tech Stack</Typography>
              <Typography variant="body2" component="div">Python, pandas, PyArrow, scikit-learn</Typography>
              <Typography variant="body2" component="div">XGBoost (CUDA), PyTorch (LSTM)</Typography>
              <Typography variant="body2" component="div">MLflow, Databricks</Typography>
              <Divider sx={{ my: 1 }} />
              <Typography variant="body2" component="div">Next.js, Material UI, Recharts</Typography>
              <Typography variant="body2" component="div">Supabase (PostgreSQL)</Typography>
              <Typography variant="body2" component="div">Vercel</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
