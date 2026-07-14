/* Product Demo — Scene 1.5: What is Medicare? (primer for Part B context) */

// ── SCENE: Medicare primer ────────────────────────────────────────
function SceneMedicare() {
  const { localTime: t, duration } = useSprite();
  const enter = Easing.easeOutCubic(Math.min(t / 0.5, 1));
  const exit = Math.max(0, Math.min(1, (t - (duration - 0.5)) / 0.5));

  // Reveal timing
  const eyebrow = Math.min(Math.max((t - 0.05) / 0.35, 0), 1);
  const headline = Math.min(Math.max((t - 0.25) / 0.5, 0), 1);
  const sub = Math.min(Math.max((t - 0.7) / 0.4, 0), 1);

  // The four parts — stagger reveal
  const partA = Math.min(Math.max((t - 1.1) / 0.35, 0), 1);
  const partB = Math.min(Math.max((t - 1.4) / 0.35, 0), 1);
  const partC = Math.min(Math.max((t - 1.7) / 0.35, 0), 1);
  const partD = Math.min(Math.max((t - 2.0) / 0.35, 0), 1);
  // Highlight Part B as our focus
  const focusP = Math.min(Math.max((t - 2.6) / 0.5, 0), 1);
  const focusPulse = Math.sin(Math.max(0, t - 2.6) * 4) * 0.04;

  // Right column stats — stagger
  const stat1 = Math.min(Math.max((t - 1.3) / 0.4, 0), 1);
  const stat2 = Math.min(Math.max((t - 1.7) / 0.4, 0), 1);
  const stat3 = Math.min(Math.max((t - 2.1) / 0.4, 0), 1);
  const stat4 = Math.min(Math.max((t - 2.5) / 0.4, 0), 1);

  // Counters
  const benCount = Easing.easeOutCubic(stat1) * 63;
  const spendCount = Easing.easeOutCubic(stat3) * 900;
  const dailyCount = Easing.easeOutCubic(stat4) * 10000;

  const parts = [
    { id: 'A', label: 'Part A', desc: 'Hospital stays, skilled nursing', reveal: partA, focus: false },
    { id: 'B', label: 'Part B', desc: 'Doctor visits, outpatient care, labs', reveal: partB, focus: true },
    { id: 'C', label: 'Part C', desc: 'Medicare Advantage (private bundles)', reveal: partC, focus: false },
    { id: 'D', label: 'Part D', desc: 'Prescription drugs', reveal: partD, focus: false },
  ];

  return (
    <div style={{
      position: 'absolute', inset: 0, opacity: enter * (1 - exit),
      background: AC.bg,
    }}>
      {/* Eyebrow */}
      <div style={{
        position: 'absolute', top: 80, left: 120,
        opacity: eyebrow,
        transform: `translateY(${(1 - eyebrow) * 8}px)`,
        fontFamily: AF.mono, fontSize: 13, fontWeight: 700,
        letterSpacing: '0.22em', textTransform: 'uppercase',
        color: AC.teal, display: 'flex', alignItems: 'center', gap: 14,
      }}>
        <span>00 · PRIMER</span>
        <span style={{ color: AC.ink3 }}>·</span>
        <span style={{ color: AC.ink3 }}>WHAT IS MEDICARE?</span>
      </div>

      {/* Headline */}
      <div style={{
        position: 'absolute', top: 124, left: 120, right: 120,
        opacity: headline,
        transform: `translateY(${(1 - Easing.easeOutCubic(headline)) * 18}px)`,
        fontFamily: AF.body, fontSize: 72, fontWeight: 800,
        letterSpacing: '-0.035em', lineHeight: 1.02, color: AC.ink,
      }}>
        Federal health insurance for{' '}
        <span style={{ color: AC.teal }}>63 million Americans.</span>
      </div>

      {/* Divider */}
      <div style={{
        position: 'absolute', top: 268, left: 120, right: 120,
        height: 1, background: AC.line,
        opacity: headline,
      }}/>

      {/* Two columns */}
      <div style={{
        position: 'absolute', top: 304, left: 120, right: 120, bottom: 100,
        display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 64,
      }}>
        {/* LEFT COLUMN — Definition + Four Parts */}
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {/* Definition block */}
          <div style={{
            opacity: sub,
            transform: `translateY(${(1 - Easing.easeOutCubic(sub)) * 12}px)`,
          }}>
            <div style={{
              fontFamily: AF.mono, fontSize: 11, fontWeight: 700,
              letterSpacing: '0.22em', textTransform: 'uppercase',
              color: AC.teal, marginBottom: 12,
            }}>Medicare in one paragraph</div>
            <div style={{
              fontFamily: AF.body, fontSize: 22, lineHeight: 1.45,
              color: AC.ink, fontWeight: 400,
            }}>
              The U.S. federal health-insurance program, enacted in <b>1965</b>.{' '}
              <b>Covers Americans 65 and older</b>, plus some younger people with
              long-term disabilities or end-stage kidney failure.
            </div>
          </div>

          {/* Four parts list */}
          <div style={{ marginTop: 40 }}>
            <div style={{
              opacity: Math.min(partA * 2, 1),
              fontFamily: AF.mono, fontSize: 11, fontWeight: 700,
              letterSpacing: '0.22em', textTransform: 'uppercase',
              color: AC.teal, marginBottom: 14,
            }}>The four parts</div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {parts.map((p) => {
                const isFocus = p.focus;
                const fp = isFocus ? focusP : 0;
                return (
                  <div key={p.id} style={{
                    opacity: p.reveal,
                    transform: `translateX(${(1 - p.reveal) * -16}px)`,
                    display: 'flex', alignItems: 'center', gap: 16,
                  }}>
                    {/* Part badge */}
                    <div style={{
                      width: 88, height: 44,
                      background: isFocus
                        ? `rgba(234, 88, 12, ${0.85 + fp * 0.15})`
                        : AC.surface2,
                      color: isFocus ? '#fff' : AC.ink,
                      border: isFocus
                        ? `1px solid ${AC.orange}`
                        : `1px solid ${AC.line}`,
                      borderRadius: 6,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontFamily: AF.body, fontSize: 15, fontWeight: 700,
                      letterSpacing: '-0.005em',
                      boxShadow: isFocus
                        ? `0 0 0 ${4 + focusPulse * 40}px rgba(234, 88, 12, ${0.12 - focusPulse * 1.5})`
                        : 'none',
                      transform: isFocus ? `scale(${1 + focusPulse * 0.6})` : 'scale(1)',
                      transition: 'transform 80ms',
                      flexShrink: 0,
                    }}>{p.label}</div>
                    {/* Description */}
                    <div style={{
                      fontFamily: AF.body, fontSize: 18,
                      color: isFocus ? AC.ink : AC.ink2,
                      fontWeight: isFocus ? 600 : 500,
                      lineHeight: 1.3,
                    }}>{p.desc}</div>
                  </div>
                );
              })}
            </div>

            {/* Focus arrow / callout */}
            <div style={{
              marginTop: 22, marginLeft: 4,
              opacity: focusP,
              transform: `translateY(${(1 - focusP) * 8}px)`,
              display: 'flex', alignItems: 'center', gap: 10,
              fontFamily: AF.mono, fontSize: 13, fontWeight: 600,
              letterSpacing: '0.16em', textTransform: 'uppercase',
              color: AC.orange,
            }}>
              <span style={{ fontSize: 16, lineHeight: 1 }}>↑</span>
              <span>Part B is what AllowanceMap predicts</span>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN — Stat cards */}
        <div style={{
          display: 'flex', flexDirection: 'column', gap: 14,
          justifyContent: 'space-between',
        }}>
          <StatCard
            reveal={stat1} accent={AC.tealLight}
            value={`~${benCount.toFixed(0)}M`}
            label="Medicare beneficiaries (2023)"
            sub="Roughly 1 in 5 Americans"
          />
          <StatCard
            reveal={stat2} accent={AC.teal}
            value="80 / 20"
            label='Part B cost split'
            sub={'Medicare pays 80% of the "allowed amount", patient pays 20%'}
          />
          <StatCard
            reveal={stat3} accent={AC.tealDark}
            value={`$${spendCount.toFixed(0)}B+`}
            label="Annual Medicare spending"
            sub="Largest single U.S. health-insurance program"
          />
          <StatCard
            reveal={stat4} accent={AC.amber}
            value={`${dailyCount.toLocaleString('en-US', { maximumFractionDigits: 0 })}/day`}
            label="Americans aging into Medicare"
            sub="Caseload grows every day"
          />
        </div>
      </div>
    </div>
  );
}

function StatCard({ reveal, accent, value, label, sub }) {
  return (
    <div style={{
      opacity: reveal,
      transform: `translateX(${(1 - Easing.easeOutCubic(reveal)) * 24}px)`,
      background: AC.surface,
      border: `1px solid ${AC.line}`,
      borderLeft: `4px solid ${accent}`,
      borderRadius: 10,
      padding: '20px 24px',
      boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
      flex: 1,
      display: 'flex', flexDirection: 'column', justifyContent: 'center',
    }}>
      <div style={{
        fontFamily: AF.body, fontSize: 52, fontWeight: 800,
        color: AC.ink, letterSpacing: '-0.035em', lineHeight: 1,
        fontVariantNumeric: 'tabular-nums',
      }}>{value}</div>
      <div style={{
        fontFamily: AF.body, fontSize: 16, fontWeight: 600,
        color: AC.ink, marginTop: 8, letterSpacing: '-0.005em',
      }}>{label}</div>
      <div style={{
        fontFamily: AF.body, fontSize: 14, fontWeight: 400,
        color: AC.ink2, marginTop: 2, lineHeight: 1.35,
      }}>{sub}</div>
    </div>
  );
}

Object.assign(window, { SceneMedicare, StatCard });
