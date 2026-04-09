'use client';

import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#0F6E8C',
      light: '#1389AC',
      dark: '#0A4F66',
      contrastText: '#FFFFFF',
    },
    secondary: {
      main: '#15755D',
      light: '#1CA082',
      dark: '#0E5241',
      contrastText: '#FFFFFF',
    },
    background: {
      default: '#FAFAF8',
      paper: '#FFFFFF',
    },
    text: {
      primary: '#1C1917',
      secondary: '#57534E',
      disabled: '#A8A29E',
    },
    divider: 'rgba(0, 0, 0, 0.08)',
  },
  typography: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    h3: { fontWeight: 800, letterSpacing: '-0.02em' },
    h4: { fontWeight: 700, letterSpacing: '-0.01em' },
    h5: { fontWeight: 700 },
    h6: { fontWeight: 600 },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)',
          border: '1px solid rgba(0,0,0,0.08)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          boxShadow: 'none',
          '&:hover': { boxShadow: 'none' },
        },
        sizeLarge: {
          padding: '13px 24px',
          fontSize: '0.95rem',
        },
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          backgroundColor: '#FFFFFF',
        },
        notchedOutline: {
          borderColor: 'rgba(0,0,0,0.15)',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#FFFFFF',
          color: '#1C1917',
          boxShadow: '0 1px 0 rgba(0,0,0,0.08)',
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
          fontSize: '0.9rem',
          '&.Mui-selected': { fontWeight: 600 },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: { borderRadius: 999 },
      },
    },
  },
});

export default theme;
