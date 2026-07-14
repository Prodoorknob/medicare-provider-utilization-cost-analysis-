/* Product Demo — Scenes 6-7: Feature recap, CTA outro */

// ── SCENE 6: Feature recap — three capabilities ──────────────────
function SceneRecap() {
  const { localTime: t, duration } = useSprite();
  const enter = Easing.easeOutCubic(Math.min(t / 0.4, 1));
  const exit = Math.max(0, Math.min(1, (t - (duration - 0.5)) / 0.5));

  const c1 = Math.min(Math.max((t - 0.3) / 0.45, 0), 1);
  const c2 = Math.min(Math.max((t - 0.7) / 0.45, 0), 1);
  const c3 = Math.min(Math.max((t - 1.1) / 0.45, 0), 1);

  const cards = [
    {
      p: c1,
      tag: '01 · ESTIMATE',
      title: 'Instant allowed amount',
      body: 'Any HCPCS · any state · any specialty.',
      stat: '$87.24',
      statLabel: 'in 2 seconds',
      color: AC.teal,
    },
    {
      p: c2,
      tag: '02 · PROJECT',
      title: 'Forecast through 2026',
      body: 'P10/P90 bounds on every rate.',
      stat: '+2.7%',
      statLabel: '3-yr CAGR',
      color: AC.green,
    },
    {
      p: c3,
      tag: '03 · INVESTIGATE',
      title: 'AI fraud detection',
      body: 'Claude-written briefs, analyst-ready.',
      stat: '1.26M',
      statLabel: 'providers scanned',
      color: AC.red,
    },
  ];

  return (
    <div style={{
      position: 'absolute', inset: 0, opacity: enter * (1 - exit),
      background: AC.bg,
    }}>
      <div style={{
        position: 'absolute', top: 120, left: 0, right: 0,
        textAlign: 'center',
        fontFamily: AF.mono, fontSize: 13, fontWeight: 700,
        letterSpacing: '0.28em', textTransform: 'uppercase',
        color: AC.teal,
      }}>ONE TOOL · THREE SUPERPOWERS</div>

      <div style={{
        position: 'absolute', top: 160, left: 120, right: 120,
        textAlign: 'center',
        fontFamily: AF.body, fontSize: 76, fontWeight: 800,
        letterSpacing: '-0.035em', lineHeight: 1.04, color: AC.ink,
      }}>
        Medicare costs, <span style={{ color: AC.teal }}>decoded</span>.
      </div>

      {/* Three cards */}
      <div style={{
        position: 'absolute', top: 360, left: 120, right: 120,
        display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 28,
      }}>
        {cards.map((c, i) => (
          <div key={i} style={{
            background: AC.surface, border: `1px solid ${AC.line}`,
            borderLeft: `4px solid ${c.color}`,
            borderRadius: 12, padding: 32, minHeight: 420,
            display: 'flex', flexDirection: 'column',
            opacity: c.p,
            transform: `translateY(${(1 - Easing.easeOutCubic(c.p)) * 30}px)`,
            boxShadow: `0 ${8 + c.p * 12}px ${24 + c.p * 16}px rgba(0,0,0,0.06)`,
          }}>
            <div style={{
              fontFamily: AF.mono, fontSize: 11, fontWeight: 700,
              letterSpacing: '0.22em', color: c.color,
            }}>{c.tag}</div>
            <div style={{
              fontFamily: AF.body, fontSize: 32, fontWeight: 800,
              letterSpacing: '-0.02em', color: AC.ink,
              marginTop: 10, lineHeight: 1.1,
            }}>{c.title}</div>
            <div style={{
              fontFamily: AF.body, fontSize: 16, color: AC.ink2,
              marginTop: 10, lineHeight: 1.5,
            }}>{c.body}</div>
            <div style={{ flex: 1 }}/>
            <div style={{ borderTop: `1px solid ${AC.line}`, paddingTop: 16 }}>
              <div style={{
                fontFamily: AF.mono, fontSize: 48, fontWeight: 700,
                color: c.color, lineHeight: 1,
                letterSpacing: '-0.02em',
                fontVariantNumeric: 'tabular-nums',
              }}>{c.stat}</div>
              <div style={{
                fontFamily: AF.mono, fontSize: 11, fontWeight: 600,
                letterSpacing: '0.2em', textTransform: 'uppercase',
                color: AC.ink3, marginTop: 6,
              }}>{c.statLabel}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── SCENE 7: CTA outro ────────────────────────────────────────────
function SceneCTA() {
  const { localTime: t, duration } = useSprite();
  const enter = Easing.easeOutCubic(Math.min(t / 0.6, 1));

  const logoP = Math.min(Math.max((t - 0.1) / 0.6, 0), 1);
  const wordP = Math.min(Math.max((t - 0.5) / 0.6, 0), 1);
  const urlP = Math.min(Math.max((t - 1.2) / 0.5, 0), 1);
  const tagP = Math.min(Math.max((t - 1.8) / 0.5, 0), 1);

  // Subtle pulse on CTA
  const pulse = Math.sin(t * 2.5) * 0.015;

  return (
    <div style={{
      position: 'absolute', inset: 0, opacity: enter,
      background: AC.ink, overflow: 'hidden',
    }}>
      {/* Ambient radial glow */}
      <div style={{
        position: 'absolute', inset: 0,
        background: `radial-gradient(ellipse at center, rgba(15,110,140,0.28), transparent 55%)`,
      }}/>

      {/* Floating dot grid (decoration) */}
      <svg viewBox="0 0 1920 1080" style={{ position: 'absolute', inset: 0 }}>
        {[...Array(60)].map((_, i) => {
          const seed = Math.sin(i * 9.31) * 10000;
          const fx = seed - Math.floor(seed);
          const seed2 = Math.sin(i * 17.7) * 10000;
          const fy = seed2 - Math.floor(seed2);
          const x = fx * 1920;
          const y = fy * 1080;
          const drift = Math.sin(t * 0.3 + i) * 8;
          return (
            <circle key={i} cx={x} cy={y + drift} r="1.5"
              fill={AC.tealLight} opacity={0.3 + Math.sin(t + i) * 0.2}/>
          );
        })}
      </svg>

      {/* Stack, centered */}
      <div style={{
        position: 'absolute', inset: 0,
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
      }}>
        {/* Logo */}
        <div style={{
          opacity: logoP,
          transform: `scale(${Easing.easeOutBack(logoP)})`,
          marginBottom: 40,
        }}>
          <AMLogo size={96} color={AC.tealLight}/>
        </div>

        {/* Wordmark */}
        <div style={{
          fontFamily: AF.body, fontSize: 120, fontWeight: 800,
          letterSpacing: '-0.04em', color: '#fff',
          opacity: wordP,
          transform: `translateY(${(1 - Easing.easeOutCubic(wordP)) * 30}px)`,
          lineHeight: 1,
        }}>
          AllowanceMap
        </div>

        {/* Tagline */}
        <div style={{
          fontFamily: AF.body, fontSize: 28, color: '#C7C2BD',
          marginTop: 28,
          opacity: tagP,
          transform: `translateY(${(1 - Easing.easeOutCubic(tagP)) * 20}px)`,
          textAlign: 'center', maxWidth: 800,
          lineHeight: 1.4,
        }}>
          Know what Medicare pays. <span style={{ color: AC.tealLight, fontWeight: 600 }}>
          See what you'll owe.</span> Catch what others miss.
        </div>

        {/* CTA button */}
        <a href="https://allowancemap.vercel.app/" target="_blank" rel="noopener noreferrer" style={{
          marginTop: 56,
          opacity: urlP,
          transform: `translateY(${(1 - Easing.easeOutCubic(urlP)) * 20}px) scale(${1 + pulse})`,
          display: 'flex', alignItems: 'center', gap: 12,
          background: AC.tealLight, color: AC.ink,
          padding: '22px 44px', borderRadius: 12,
          fontFamily: AF.body, fontSize: 24, fontWeight: 700,
          letterSpacing: '-0.01em', textDecoration: 'none',
          boxShadow: `0 20px 50px rgba(19,137,172,0.45)`,
        }}>
          <span>allowancemap.vercel.app</span>
          <span style={{ opacity: 0.8 }}>↗</span>
        </a>

        {/* Meta row */}
        <div style={{
          marginTop: 60,
          opacity: tagP * 0.8,
          display: 'flex', gap: 36, alignItems: 'center',
          fontFamily: AF.mono, fontSize: 12, letterSpacing: '0.22em',
          textTransform: 'uppercase', color: AC.ink3,
        }}>
          <span>Built on CMS 2013–2023</span>
          <span style={{ color: 'rgba(255,255,255,0.25)' }}>·</span>
          <span>103M+ records</span>
          <span style={{ color: 'rgba(255,255,255,0.25)' }}>·</span>
          <span style={{ color: AC.tealLight }}>Powered by Claude</span>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { SceneRecap, SceneCTA });
