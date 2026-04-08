import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Link from '@mui/material/Link';

export default function Footer() {
  return (
    <Box component="footer" sx={{ py: 3, px: 2, mt: 'auto', bgcolor: 'grey.100', textAlign: 'center' }}>
      <Typography variant="body2" color="text.secondary">
        Data source: CMS Medicare Physician & Other Practitioners (2013-2023)
      </Typography>
      <Typography variant="body2" color="text.secondary">
        Built by Raj Vedire &middot; Indiana University &middot;{' '}
        <Link href="https://github.com/Prodoorknob/medicare-provider-utilization-cost-analysis-" target="_blank" rel="noopener">
          GitHub
        </Link>
      </Typography>
    </Box>
  );
}
