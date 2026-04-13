'use client';

import { createTheme } from '@mui/material/styles';

// Design system tokens
const PRIMARY = '#0F6E8C';
const PRIMARY_LIGHT = '#1389AC';
const PRIMARY_DARK = '#0A4F66';
const PRIMARY_TINT = '#D0EEF5';
const PRIMARY_SUBTLE = '#EBF7FB';

const SECONDARY = '#15755D';
const SECONDARY_LIGHT = '#1CA082';
const SECONDARY_DARK = '#0E5241';
const SECONDARY_TINT = '#E8F7F3';
const SECONDARY_SUBTLE = '#F0FAF7';

const ACCENT = '#B8763A';

const theme = createTheme({
  palette: {
    primary: {
      main: PRIMARY,
      light: PRIMARY_LIGHT,
      dark: PRIMARY_DARK,
      contrastText: '#FFFFFF',
    },
    secondary: {
      main: SECONDARY,
      light: SECONDARY_LIGHT,
      dark: SECONDARY_DARK,
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
    success: { main: '#10b981' },
    warning: { main: '#D97706' },
    error: { main: '#DC2626' },
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
          boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
          border: '1px solid rgba(0,0,0,0.08)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none' as const,
          fontWeight: 600,
          boxShadow: 'none',
          borderRadius: 8,
          '&:hover': { boxShadow: 'none' },
        },
        sizeLarge: {
          padding: '13px 24px',
          fontSize: '0.95rem',
          height: 48,
        },
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          backgroundColor: '#FFFFFF',
          '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
            borderColor: PRIMARY_LIGHT,
            borderWidth: 1,
            boxShadow: `0 0 0 3px rgba(15,110,140,0.12)`,
          },
        },
        notchedOutline: {
          borderColor: 'rgba(0,0,0,0.15)',
        },
      },
    },
    MuiToggleButtonGroup: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          border: '1px solid rgba(0,0,0,0.15)',
          overflow: 'hidden',
          '& .MuiToggleButtonGroup-grouped': {
            border: 0,
            borderRadius: 0,
            '&:not(:first-of-type)': {
              borderLeft: '1px solid rgba(0,0,0,0.15)',
            },
          },
        },
      },
    },
    MuiToggleButton: {
      styleOverrides: {
        root: {
          textTransform: 'none' as const,
          fontWeight: 500,
          fontSize: '0.8125rem',
          color: '#57534E',
          padding: '8px 16px',
          '&.Mui-selected': {
            backgroundColor: PRIMARY,
            color: '#FFFFFF',
            fontWeight: 600,
            '&:hover': {
              backgroundColor: PRIMARY_DARK,
            },
          },
        },
      },
    },
    MuiSwitch: {
      styleOverrides: {
        switchBase: {
          '&.Mui-checked': {
            color: '#FFFFFF',
          },
          '&.Mui-checked + .MuiSwitch-track': {
            backgroundColor: SECONDARY,
            opacity: 1,
          },
        },
        track: {
          backgroundColor: '#D6D3D1',
          opacity: 1,
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
          textTransform: 'none' as const,
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
    MuiAccordion: {
      styleOverrides: {
        root: {
          borderRadius: '8px !important',
          border: '1px solid rgba(0,0,0,0.08)',
          boxShadow: 'none',
          '&:before': { display: 'none' },
        },
      },
    },
  },
});

// Custom palette extensions (accessible via theme vars or direct import)
export const PALETTE = {
  primaryTint: PRIMARY_TINT,
  primarySubtle: PRIMARY_SUBTLE,
  secondaryTint: SECONDARY_TINT,
  secondarySubtle: SECONDARY_SUBTLE,
  accent: ACCENT,
  accentBg: '#FDF4EA',
};

export default theme;
