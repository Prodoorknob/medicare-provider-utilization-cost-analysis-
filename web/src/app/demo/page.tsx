'use client';

import { useRef, useState } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Button from '@mui/material/Button';
import IconButton from '@mui/material/IconButton';
import Stack from '@mui/material/Stack';
import Link from 'next/link';
import PlayArrowRoundedIcon from '@mui/icons-material/PlayArrowRounded';
import PauseRoundedIcon from '@mui/icons-material/PauseRounded';
import VolumeUpRoundedIcon from '@mui/icons-material/VolumeUpRounded';
import VolumeOffRoundedIcon from '@mui/icons-material/VolumeOffRounded';
import ReplayRoundedIcon from '@mui/icons-material/ReplayRounded';
import OpenInNewRoundedIcon from '@mui/icons-material/OpenInNewRounded';
import TrendingUpRoundedIcon from '@mui/icons-material/TrendingUpRounded';
import SearchRoundedIcon from '@mui/icons-material/SearchRounded';
import PaidRoundedIcon from '@mui/icons-material/PaidRounded';
import { PALETTE } from '@/app/theme';

const MONO = '"IBM Plex Mono", ui-monospace, monospace';
const INK = '#1C1917';
const INK2 = '#57534E';
const INK3 = '#A8A29E';
const TEAL = '#0F6E8C';
const TEAL_LIGHT = '#1389AC';
const GREEN = '#15755D';
const RED = '#DC2626';
const LINE = 'rgba(0,0,0,0.08)';

const RECAP_CARDS = [
  {
    tag: '01 · ESTIMATE',
    title: 'Instant allowed amount',
    body: 'Any HCPCS · any state · any specialty.',
    stat: '$87.24',
    statLabel: 'in 2 seconds',
    color: TEAL,
    icon: PaidRoundedIcon,
    href: '/',
  },
  {
    tag: '02 · PROJECT',
    title: 'Forecast through 2026',
    body: 'P10/P90 bounds on every rate.',
    stat: '+2.7%',
    statLabel: '3-yr CAGR',
    color: GREEN,
    icon: TrendingUpRoundedIcon,
    href: '/forecast',
  },
  {
    tag: '03 · INVESTIGATE',
    title: 'AI fraud detection',
    body: 'Claude-written briefs, analyst-ready.',
    stat: '1.26M',
    statLabel: 'providers scanned',
    color: RED,
    icon: SearchRoundedIcon,
    href: '/investigations',
  },
] as const;

export default function DemoPage() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [playing, setPlaying] = useState(true);
  const [muted, setMuted] = useState(true);

  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) {
      void v.play();
      setPlaying(true);
    } else {
      v.pause();
      setPlaying(false);
    }
  };

  const toggleMute = () => {
    const v = videoRef.current;
    if (!v) return;
    v.muted = !v.muted;
    setMuted(v.muted);
  };

  const restart = () => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = 0;
    void v.play();
    setPlaying(true);
  };

  return (
    <Box>
      {/* Hero */}
      <Box sx={{ textAlign: 'center', pb: 4, mb: 4 }}>
        <Typography
          sx={{
            fontFamily: MONO,
            fontSize: 13,
            fontWeight: 700,
            letterSpacing: '0.28em',
            textTransform: 'uppercase',
            color: TEAL,
            mb: 2,
          }}
        >
          One Tool · Three Superpowers
        </Typography>
        <Typography
          component="h1"
          sx={{
            fontSize: { xs: 40, sm: 56, md: 72 },
            fontWeight: 800,
            letterSpacing: '-0.035em',
            lineHeight: 1.04,
            color: INK,
            mb: 2,
          }}
        >
          Medicare costs, <Box component="span" sx={{ color: TEAL }}>decoded</Box>.
        </Typography>
        <Typography
          sx={{
            fontSize: { xs: 16, md: 18 },
            color: INK2,
            maxWidth: 640,
            mx: 'auto',
            lineHeight: 1.55,
          }}
        >
          A 52-second tour of AllowanceMap — what Medicare actually is, instant
          cost estimates, three-year forecasts, and an AI fraud-investigation
          agent built on 103M+ CMS records.
        </Typography>
      </Box>

      {/* Video player */}
      <Box
        sx={{
          position: 'relative',
          width: '100%',
          aspectRatio: '16 / 9',
          borderRadius: 3,
          overflow: 'hidden',
          background: '#0a0a0a',
          boxShadow: '0 30px 80px rgba(0,0,0,0.18), 0 0 0 1px rgba(0,0,0,0.05)',
          mb: 6,
        }}
      >
        <Box
          component="video"
          ref={videoRef}
          src="/AllowanceMap_ProductDemo.mp4"
          autoPlay
          loop
          muted
          playsInline
          preload="auto"
          onClick={togglePlay}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          sx={{
            width: '100%',
            height: '100%',
            display: 'block',
            cursor: 'pointer',
            objectFit: 'cover',
          }}
        />

        {/* Controls overlay */}
        <Stack
          direction="row"
          spacing={1}
          sx={{
            position: 'absolute',
            bottom: 16,
            right: 16,
            background: 'rgba(0,0,0,0.55)',
            backdropFilter: 'blur(8px)',
            borderRadius: 999,
            px: 1,
            py: 0.5,
            alignItems: 'center',
          }}
        >
          <IconButton
            size="small"
            onClick={togglePlay}
            sx={{ color: '#fff' }}
            aria-label={playing ? 'Pause' : 'Play'}
          >
            {playing ? <PauseRoundedIcon /> : <PlayArrowRoundedIcon />}
          </IconButton>
          <IconButton
            size="small"
            onClick={restart}
            sx={{ color: '#fff' }}
            aria-label="Restart"
          >
            <ReplayRoundedIcon fontSize="small" />
          </IconButton>
          <IconButton
            size="small"
            onClick={toggleMute}
            sx={{ color: '#fff' }}
            aria-label={muted ? 'Unmute' : 'Mute'}
          >
            {muted ? <VolumeOffRoundedIcon fontSize="small" /> : <VolumeUpRoundedIcon fontSize="small" />}
          </IconButton>
        </Stack>

        {/* Mono caption chip */}
        <Box
          sx={{
            position: 'absolute',
            top: 16,
            left: 16,
            background: 'rgba(255,255,255,0.92)',
            backdropFilter: 'blur(6px)',
            borderRadius: 999,
            px: 1.5,
            py: 0.5,
            fontFamily: MONO,
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: '0.18em',
            textTransform: 'uppercase',
            color: TEAL,
            display: 'flex',
            alignItems: 'center',
            gap: 0.75,
          }}
        >
          <Box
            sx={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: GREEN,
              boxShadow: `0 0 0 3px ${PALETTE.secondarySubtle}`,
            }}
          />
          Product Demo · 0:52
        </Box>
      </Box>

      {/* Recap cards */}
      <Grid container spacing={3} sx={{ mb: 8 }}>
        {RECAP_CARDS.map((c) => {
          const Icon = c.icon;
          return (
            <Grid key={c.tag} size={{ xs: 12, md: 4 }}>
              <Card
                component={Link}
                href={c.href}
                sx={{
                  display: 'block',
                  textDecoration: 'none',
                  height: '100%',
                  borderLeft: `4px solid ${c.color}`,
                  transition: 'transform 200ms ease, box-shadow 200ms ease',
                  '&:hover': {
                    transform: 'translateY(-3px)',
                    boxShadow: '0 14px 32px rgba(0,0,0,0.08)',
                  },
                }}
              >
                <CardContent sx={{ p: 4, display: 'flex', flexDirection: 'column', height: '100%', minHeight: 320 }}>
                  <Stack direction="row" sx={{ justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                    <Typography
                      sx={{
                        fontFamily: MONO,
                        fontSize: 11,
                        fontWeight: 700,
                        letterSpacing: '0.22em',
                        color: c.color,
                      }}
                    >
                      {c.tag}
                    </Typography>
                    <Icon sx={{ color: c.color, fontSize: 22, opacity: 0.85 }} />
                  </Stack>

                  <Typography
                    sx={{
                      fontSize: 24,
                      fontWeight: 800,
                      letterSpacing: '-0.02em',
                      lineHeight: 1.15,
                      color: INK,
                      mb: 1,
                    }}
                  >
                    {c.title}
                  </Typography>
                  <Typography sx={{ fontSize: 14, color: INK2, lineHeight: 1.55 }}>{c.body}</Typography>

                  <Box sx={{ flex: 1 }} />

                  <Box sx={{ borderTop: `1px solid ${LINE}`, pt: 2, mt: 3 }}>
                    <Typography
                      sx={{
                        fontFamily: MONO,
                        fontSize: 36,
                        fontWeight: 700,
                        color: c.color,
                        lineHeight: 1,
                        letterSpacing: '-0.02em',
                        fontVariantNumeric: 'tabular-nums',
                      }}
                    >
                      {c.stat}
                    </Typography>
                    <Typography
                      sx={{
                        fontFamily: MONO,
                        fontSize: 11,
                        fontWeight: 600,
                        letterSpacing: '0.2em',
                        textTransform: 'uppercase',
                        color: INK3,
                        mt: 0.5,
                      }}
                    >
                      {c.statLabel}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>

      {/* Dark CTA section — translated from SceneCTA */}
      <Box
        sx={{
          position: 'relative',
          background: INK,
          borderRadius: 3,
          overflow: 'hidden',
          py: { xs: 8, md: 12 },
          px: { xs: 4, md: 8 },
          textAlign: 'center',
          mb: 4,
        }}
      >
        {/* Ambient teal glow */}
        <Box
          sx={{
            position: 'absolute',
            inset: 0,
            background: `radial-gradient(ellipse at center, rgba(15,110,140,0.32), transparent 60%)`,
            pointerEvents: 'none',
          }}
        />

        <Box sx={{ position: 'relative' }}>
          <Typography
            sx={{
              fontSize: { xs: 48, md: 88 },
              fontWeight: 800,
              letterSpacing: '-0.04em',
              color: '#fff',
              lineHeight: 1,
              mb: 3,
            }}
          >
            AllowanceMap
          </Typography>
          <Typography
            sx={{
              fontSize: { xs: 18, md: 22 },
              color: '#C7C2BD',
              maxWidth: 720,
              mx: 'auto',
              lineHeight: 1.45,
              mb: 5,
            }}
          >
            Know what Medicare pays.{' '}
            <Box component="span" sx={{ color: TEAL_LIGHT, fontWeight: 600 }}>
              See what you&apos;ll owe.
            </Box>{' '}
            Catch what others miss.
          </Typography>

          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ justifyContent: 'center', alignItems: 'center' }}>
            <Button
              component={Link}
              href="/"
              variant="contained"
              size="large"
              endIcon={<OpenInNewRoundedIcon />}
              sx={{
                background: TEAL_LIGHT,
                color: INK,
                fontWeight: 700,
                fontSize: 16,
                px: 4,
                py: 1.75,
                borderRadius: 2,
                boxShadow: '0 16px 40px rgba(19,137,172,0.45)',
                '&:hover': { background: TEAL, color: '#fff', boxShadow: '0 18px 44px rgba(19,137,172,0.55)' },
              }}
            >
              Try the Estimator
            </Button>
            <Button
              component={Link}
              href="/investigations"
              variant="outlined"
              size="large"
              sx={{
                color: '#fff',
                borderColor: 'rgba(255,255,255,0.3)',
                fontWeight: 600,
                fontSize: 16,
                px: 4,
                py: 1.75,
                borderRadius: 2,
                '&:hover': { borderColor: TEAL_LIGHT, background: 'rgba(19,137,172,0.12)' },
              }}
            >
              See fraud briefs
            </Button>
          </Stack>

          <Stack
            direction={{ xs: 'column', sm: 'row' }}
            spacing={{ xs: 1, sm: 4.5 }}
            sx={{
              justifyContent: 'center',
              alignItems: 'center',
              mt: 6,
              fontFamily: MONO,
              fontSize: 11,
              letterSpacing: '0.22em',
              textTransform: 'uppercase',
              color: INK3,
            }}
          >
            <span>Built on CMS 2013–2023</span>
            <Box component="span" sx={{ color: 'rgba(255,255,255,0.25)', display: { xs: 'none', sm: 'inline' } }}>·</Box>
            <span>103M+ records</span>
            <Box component="span" sx={{ color: 'rgba(255,255,255,0.25)', display: { xs: 'none', sm: 'inline' } }}>·</Box>
            <Box component="span" sx={{ color: TEAL_LIGHT }}>Powered by Claude</Box>
          </Stack>
        </Box>
      </Box>
    </Box>
  );
}
