/* AllowanceMap deck components */

const TYPE = {
  title: 64, subtitle: 44, body: 34, small: 28, tiny: 22, micro: 18,
  metric: 96, metricSm: 72, eyebrow: 20,
};
const SP = {
  pt: 80, pb: 72, px: 96, titleGap: 48, itemGap: 24, sectionGap: 40,
};
const C = {
  bg: '#FAFAF8', surface: '#FFFFFF', surface2: '#F2F0ED',
  border: 'rgba(0,0,0,0.08)', border2: 'rgba(0,0,0,0.15)',
  text: '#1C1917', text2: '#57534E', text3: '#A8A29E', muted: '#D6D3D1',
  primary: '#0F6E8C', primaryLight: '#1389AC', primaryDark: '#0A4F66',
  primarySubtle: 'rgba(15,110,140,0.08)', primaryTint: '#D0EEF5',
  secondary: '#15755D', secondaryLight: '#1CA082', secondaryDark: '#0E5241',
  secondaryTint: '#E8F7F3',
  accent: '#B8763A', accentLight: '#D4944D', accentTint: '#FDF4EA',
  success: '#10b981', warning: '#D97706', error: '#DC2626',
};
const FONT = { body: 'Inter, system-ui, sans-serif', mono: '"IBM Plex Mono", monospace' };

window.TYPE = TYPE; window.SP = SP; window.C = C; window.FONT = FONT;

// ── Generic slide frame ──────────────────────────────────
function Slide({ bg = C.bg, children, style = {} }) {
  return (
    <div style={{
      width: '100%', height: '100%', background: bg, color: C.text,
      fontFamily: FONT.body, overflow: 'hidden', position: 'relative',
      display: 'flex', flexDirection: 'column',
      padding: `${SP.pt}px ${SP.px}px ${SP.pb}px`,
      ...style,
    }}>
      {children}
    </div>
  );
}

function Eyebrow({ children, color = C.primary }) {
  return (
    <div style={{
      fontFamily: FONT.mono, fontSize: TYPE.eyebrow, fontWeight: 700,
      letterSpacing: '0.18em', textTransform: 'uppercase',
      color: color, marginBottom: SP.itemGap,
    }}>{children}</div>
  );
}

function Title({ children, color = C.text, size = TYPE.title, style = {} }) {
  return (
    <div style={{
      fontSize: size, fontWeight: 800, letterSpacing: '-0.02em',
      lineHeight: 1.05, color, marginBottom: SP.titleGap, ...style,
    }}>{children}</div>
  );
}

function Subtitle({ children, color = C.text2, style = {} }) {
  return (
    <div style={{
      fontSize: TYPE.subtitle, fontWeight: 400, lineHeight: 1.3,
      color, marginBottom: SP.itemGap, ...style,
    }}>{children}</div>
  );
}

function Body({ children, color = C.text2, size = TYPE.body, style = {} }) {
  return (
    <div style={{ fontSize: size, lineHeight: 1.5, color, textWrap: 'pretty', ...style }}>{children}</div>
  );
}

function Mono({ children, color = C.primary, size = TYPE.small, weight = 500, style = {} }) {
  return (
    <span style={{ fontFamily: FONT.mono, fontSize: size, color, fontWeight: weight, ...style }}>{children}</span>
  );
}

function Rule({ color = C.border, style = {} }) {
  return <div style={{ height: 1, background: color, width: '100%', ...style }} />;
}

function Footer({ n, total, label }) {
  return (
    <div style={{
      position: 'absolute', bottom: 36, left: SP.px, right: SP.px,
      display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
      fontFamily: FONT.mono, fontSize: TYPE.micro, color: C.text3,
      letterSpacing: '0.12em', textTransform: 'uppercase',
    }}>
      <span>AllowanceMap · Medicare Provider Cost Analysis</span>
      <span>{label}</span>
      <span>{String(n).padStart(2,'0')} / {String(total).padStart(2,'0')}</span>
    </div>
  );
}

Object.assign(window, { Slide, Eyebrow, Title, Subtitle, Body, Mono, Rule, Footer });
