'use client';

import { useState } from 'react';
import { usePathname } from 'next/navigation';
import Link from 'next/link';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import IconButton from '@mui/material/IconButton';
import Drawer from '@mui/material/Drawer';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemText from '@mui/material/ListItemText';
import Box from '@mui/material/Box';
import useMediaQuery from '@mui/material/useMediaQuery';
import { useTheme } from '@mui/material/styles';
import MenuIcon from '@mui/icons-material/Menu';
import LocalHospitalIcon from '@mui/icons-material/LocalHospital';
import { NAV_LINKS } from '@/lib/constants';

export default function Navbar() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const pathname = usePathname();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  return (
    <>
      <AppBar position="sticky" elevation={0} sx={{ bgcolor: 'background.paper', borderBottom: '1px solid', borderColor: 'divider' }}>
        <Toolbar>
          {isMobile && (
            <IconButton edge="start" onClick={() => setDrawerOpen(true)} sx={{ mr: 1, color: 'text.primary' }}>
              <MenuIcon />
            </IconButton>
          )}
          <LocalHospitalIcon sx={{ mr: 1, color: 'primary.main' }} />
          <Typography variant="h6" component={Link} href="/" sx={{ flexGrow: 1, textDecoration: 'none', color: 'primary.main', fontWeight: 700 }}>
            AllowanceMap
          </Typography>
          {!isMobile && NAV_LINKS.map((link) => (
            <Button
              key={link.href}
              component={Link}
              href={link.href}
              sx={{
                mx: 0.5,
                color: pathname === link.href ? 'primary.main' : 'text.secondary',
                fontWeight: pathname === link.href ? 600 : 500,
                borderBottom: '2px solid',
                borderColor: pathname === link.href ? 'primary.main' : 'transparent',
                borderRadius: 0,
                pb: 0.5,
              }}
            >
              {link.label}
            </Button>
          ))}
        </Toolbar>
      </AppBar>

      <Drawer anchor="left" open={drawerOpen} onClose={() => setDrawerOpen(false)}>
        <Box sx={{ width: 260, pt: 2 }}>
          <Typography variant="h6" sx={{ px: 2, pb: 1, fontWeight: 700, color: 'primary.main' }}>
            AllowanceMap
          </Typography>
          <List>
            {NAV_LINKS.map((link) => (
              <ListItem key={link.href} disablePadding>
                <ListItemButton
                  component={Link}
                  href={link.href}
                  selected={pathname === link.href}
                  onClick={() => setDrawerOpen(false)}
                >
                  <ListItemText primary={link.label} />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
        </Box>
      </Drawer>
    </>
  );
}
