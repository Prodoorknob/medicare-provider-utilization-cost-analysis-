import type { Metadata } from 'next';
import { AppRouterCacheProvider } from '@mui/material-nextjs/v15-appRouter';
import Box from '@mui/material/Box';
import Providers from './providers';
import Navbar from '@/components/layout/Navbar';
import Footer from '@/components/layout/Footer';

export const metadata: Metadata = {
  title: 'Medicare Provider Cost Analysis',
  description: 'Explore Medicare allowed amounts, patient out-of-pocket costs, and LSTM-powered cost forecasts across 103M+ provider records.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <AppRouterCacheProvider>
          <Providers>
            <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
              <Navbar />
              <Box component="main" sx={{ flex: 1, py: 3, px: { xs: 2, md: 4 }, maxWidth: 1400, mx: 'auto', width: '100%' }}>
                {children}
              </Box>
              <Footer />
            </Box>
          </Providers>
        </AppRouterCacheProvider>
      </body>
    </html>
  );
}
