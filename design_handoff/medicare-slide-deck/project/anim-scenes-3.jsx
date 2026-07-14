/* Scenes 6-8: Bug+fix, Results, Product/CTA */

// ── SCENE 6: The Bug (teacher forcing) ──────────────────────────
function SceneBug() {
  const { localTime: t, duration } = useSprite();
  const enter = Easing.easeOutCubic(Math.min(t / 0.4, 1));
  const exit = Math.max(0, Math.min(1, (t - (duration - 0.6)) / 0.6));

  // Phases:
  // 0.0-1.5: reported R² = 0.8860 ← teacher-forced
  // 1.5-2.8: SHAKE + RED ALERT
  // 2.8-4.0: Fix annotation revealed
  // 4.0-end: Corrected R² locked
  const phase1 = t < 1.5;
  const phase2 = t >= 1.5 && t < 2.8;
  const phase3 = t >= 2.8;

  // Number morph: 0.8860 → 0.8689 (LSTM fair AR)
  const morphP = Math.min(Math.max((t - 2.8) / 1.2, 0), 1);
  const r2 = 0.8860 - (0.8860 - 0.8689) * Easing.easeInOutCubic(morphP);

  // Shake
  const shakeAmt = phase2 ? Math.sin(t * 80) * 6 * (1 - (t - 1.5) / 1.3) : 0;

  // Red flash
  const redFlash = phase2 ? Math.max(0, 1 - (t - 1.5) / 1.3) : 0;

  return (
    <div style={{
      position: 'absolute', inset: 0, opacity: enter * (1 - exit),
      transform: `translateX(${shakeAmt}px)`,
    }}>
      {/* Red overlay on flash */}
      <div style={{
        position: 'absolute', inset: 0, background: AC.red,
        opacity: redFlash * 0.12, pointerEvents: 'none',
      }}/>

      <div style={{
        position: 'absolute', top: 100, left: 80,
        fontFamily: AF.mono, fontSize: 15, fontWeight: 700,
        letterSpacing: '0.22em', textTransform: 'uppercase',
        color: phase2 ? AC.red : AC.amber,
      }}>
        Postmortem ▸ 05 {phase2 && <span>· ALERT</span>}
      </div>

      <div style={{
        position: 'absolute', top: 150, left: 80, right: 80,
        fontFamily: AF.body, fontSize: 72, fontWeight: 800,
        letterSpacing: '-0.03em', lineHeight: 1.05, color: AC.ink,
      }}>
        {phase1 && <>LSTM forecast looked <span style={{ color: AC.red }}>too good.</span></>}
        {phase2 && <span style={{ color: AC.red }}>Teacher forcing in evaluate().</span>}
        {phase3 && <>Fair autoregressive eval. <span style={{ color: AC.green }}>Honest numbers.</span></>}
      </div>

      {/* Central metric */}
      <div style={{
        position: 'absolute', top: 440, left: 0, right: 0,
        display: 'flex', justifyContent: 'center', alignItems: 'center',
        gap: 80,
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{
            fontFamily: AF.mono, fontSize: 15, letterSpacing: '0.2em',
            textTransform: 'uppercase', color: AC.ink3, marginBottom: 12,
          }}>LSTM Forecast R²</div>
          <div style={{
            fontFamily: AF.body, fontSize: 220, fontWeight: 900,
            letterSpacing: '-0.05em', lineHeight: 1,
            color: phase3 ? AC.green : (phase2 ? AC.red : AC.ink),
            fontVariantNumeric: 'tabular-nums',
            textShadow: redFlash > 0.3 ? `0 0 40px ${AC.red}` : 'none',
          }}>
            {r2.toFixed(4)}
          </div>
          <div style={{
            fontFamily: AF.mono, fontSize: 15, letterSpacing: '0.18em',
            textTransform: 'uppercase', marginTop: 12,
            color: phase3 ? AC.green : (phase1 ? AC.red : AC.ink3),
          }}>
            {phase1 && '▲ too good to be true'}
            {phase2 && '✕ LEAKAGE DETECTED'}
            {phase3 && '✓ honest · test'}
          </div>
        </div>
      </div>

      {/* Fix annotation callout */}
      {phase3 && (
        <div style={{
          position: 'absolute', top: 800, left: 80, right: 80,
          opacity: Easing.easeOutCubic(Math.min((t - 2.8) / 0.6, 1)),
        }}>
          <div style={{
            display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 20,
          }}>
            {[
              { tag: 'Root Cause', text: 'LSTM evaluate() fed the TRUE value at each step, not its own prediction. 1-step teacher forcing, not autoregressive rollout.' },
              { tag: 'Fix', text: 'Batched AR rollout in V2_12 Cell 6. LSTM feeds its own prediction back — matches inference-time behavior.' },
              { tag: 'Impact', text: 'R² 0.8860 → 0.8689. RMSE $36.42 → $18.91. “Complementary error profile” hypothesis retracted.' },
            ].map((c, i) => (
              <div key={i} style={{
                background: '#fff', border: `1px solid ${AC.line}`,
                padding: '16px 20px', borderLeft: `3px solid ${AC.green}`,
              }}>
                <div style={{
                  fontFamily: AF.mono, fontSize: 11, letterSpacing: '0.2em',
                  textTransform: 'uppercase', fontWeight: 700,
                  color: AC.green, marginBottom: 6,
                }}>{c.tag}</div>
                <div style={{
                  fontFamily: AF.body, fontSize: 15, lineHeight: 1.4,
                  color: AC.ink,
                }}>{c.text}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── SCENE 7: Results leaderboard + scatter ──────────────────────
function SceneResults() {
  const { localTime: t, duration } = useSprite();
  const enter = Easing.easeOutCubic(Math.min(t / 0.5, 1));
  const exit = Math.max(0, Math.min(1, (t - (duration - 0.6)) / 0.6));

  // Scatter plot reveal
  const scatterP = Math.min(Math.max((t - 0.5) / 2.5, 0), 1);
  // Generate deterministic "points" around y=x
  const N = 140;
  const pts = [];
  for (let i = 0; i < N; i++) {
    const seed = Math.sin(i * 12.9898) * 43758.5453;
    const rx = (seed - Math.floor(seed));
    const ry = (Math.sin(i * 78.233) * 43758.5453);
    const jitter = (ry - Math.floor(ry)) - 0.5;
    const x = rx; // 0..1
    const y = Math.max(0, Math.min(1, x + jitter * 0.08));
    const vis = Math.min(Math.max((scatterP - (i / N) * 0.7) / 0.3, 0), 1);
    pts.push({ x, y, vis });
  }

  // Metric counters — Stage 1 LightGBM V2 no-charge (PRODUCTION)
  const counterP = Math.min(Math.max((t - 1.2) / 1.8, 0), 1);
  const r2 = Easing.easeOutCubic(counterP) * 0.9428;
  const mae = 15 - Easing.easeOutCubic(counterP) * 7.30; // → $7.70
  const rmse = 28 - Easing.easeOutCubic(counterP) * 12.23; // → $15.77

  const SCAT_X = 1080, SCAT_Y = 260, SCAT_W = 760, SCAT_H = 660;

  return (
    <div style={{
      position: 'absolute', inset: 0, opacity: enter * (1 - exit),
    }}>
      <div style={{
        position: 'absolute', top: 100, left: 80,
        fontFamily: AF.mono, fontSize: 15, fontWeight: 700,
        letterSpacing: '0.22em', textTransform: 'uppercase',
        color: AC.green,
      }}>Results ▸ 06</div>

      <div style={{
        position: 'absolute', top: 150, left: 80, right: 800,
        fontFamily: AF.body, fontSize: 72, fontWeight: 800,
        letterSpacing: '-0.03em', lineHeight: 1.05, color: AC.ink,
      }}>
        Stage 1 production.<br/>
        <span style={{ color: AC.green }}>Predicted vs. actual.</span>
      </div>

      {/* Metrics stack */}
      <div style={{
        position: 'absolute', top: 400, left: 80, right: 900,
        display: 'flex', flexDirection: 'column', gap: 20,
      }}>
        <BigMetric label="R² · no-charge LGBM" value={r2.toFixed(4)} color={AC.green} />
        <BigMetric label="MAE" value={`$${mae.toFixed(2)}`} color={AC.teal} />
        <BigMetric label="RMSE" value={`$${rmse.toFixed(2)}`} color={AC.amber} />
      </div>

      {/* Scatter */}
      <svg viewBox={`${SCAT_X - 60} ${SCAT_Y - 40} ${SCAT_W + 100} ${SCAT_H + 100}`}
        style={{
          position: 'absolute',
          left: SCAT_X - 60, top: SCAT_Y - 40,
          width: SCAT_W + 100, height: SCAT_H + 100,
        }}>
        {/* Axes */}
        <line x1={SCAT_X} y1={SCAT_Y} x2={SCAT_X} y2={SCAT_Y + SCAT_H}
          stroke={AC.line2} strokeWidth="1.5"/>
        <line x1={SCAT_X} y1={SCAT_Y + SCAT_H} x2={SCAT_X + SCAT_W} y2={SCAT_Y + SCAT_H}
          stroke={AC.line2} strokeWidth="1.5"/>
        {/* y=x reference */}
        <line x1={SCAT_X} y1={SCAT_Y + SCAT_H} x2={SCAT_X + SCAT_W} y2={SCAT_Y}
          stroke={AC.green} strokeWidth="2" strokeDasharray="4 6" opacity="0.7"/>
        {/* Labels */}
        <text x={SCAT_X - 10} y={SCAT_Y - 18} textAnchor="start"
          fontFamily={AF.mono} fontSize="12" fill={AC.ink3}
          letterSpacing="2">PREDICTED ($)</text>
        <text x={SCAT_X + SCAT_W} y={SCAT_Y + SCAT_H + 28} textAnchor="end"
          fontFamily={AF.mono} fontSize="12" fill={AC.ink3}
          letterSpacing="2">ACTUAL ($)</text>
        {/* Points */}
        {pts.map((p, i) => {
          if (p.vis <= 0) return null;
          return (
            <circle key={i}
              cx={SCAT_X + p.x * SCAT_W}
              cy={SCAT_Y + SCAT_H - p.y * SCAT_H}
              r={3 * p.vis} fill={AC.teal} opacity={p.vis * 0.75}/>
          );
        })}
        {/* Caption for y=x — placed above the diagonal, outside point cluster */}
        <text x={SCAT_X + SCAT_W + 8} y={SCAT_Y - 8} textAnchor="end"
          fontFamily={AF.mono} fontSize="12" fill={AC.green}
          letterSpacing="2">── y = x  PERFECT FIT</text>
      </svg>

      {/* Bottom tags */}
      <div style={{
        position: 'absolute', bottom: 80, left: 80,
        display: 'flex', gap: 20, opacity: Math.min((t - 3.0) / 0.6, 1),
      }}>
        {['126.8M rows', '80/20 random', 'log1p target', 'railway API'].map(tag => (
          <span key={tag} style={{
            fontFamily: AF.mono, fontSize: 12, letterSpacing: '0.16em',
            textTransform: 'uppercase', color: AC.green, fontWeight: 700,
            padding: '6px 12px', background: AC.greenTint,
            border: `1px solid ${AC.green}`,
          }}>✓ {tag}</span>
        ))}
      </div>
    </div>
  );
}

function BigMetric({ label, value, color }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'baseline', gap: 20,
      borderBottom: `1px solid ${AC.line}`, paddingBottom: 12,
    }}>
      <div style={{
        fontFamily: AF.body, fontSize: 88, fontWeight: 800,
        color, letterSpacing: '-0.03em', lineHeight: 1,
        fontVariantNumeric: 'tabular-nums',
        minWidth: 280,
      }}>{value}</div>
      <div style={{
        fontFamily: AF.mono, fontSize: 14, letterSpacing: '0.18em',
        textTransform: 'uppercase', color: AC.ink3, fontWeight: 500,
      }}>{label}</div>
    </div>
  );
}

// ── SCENE 8: The Product — US map + CTA ─────────────────────────
function SceneProduct() {
  const { localTime: t, duration } = useSprite();
  const enter = Easing.easeOutCubic(Math.min(t / 0.6, 1));

  // Map points pulse
  const cities = [
    { x: 280, y: 320, label: 'Seattle' },
    { x: 240, y: 420, label: 'SF' },
    { x: 300, y: 540, label: 'LA' },
    { x: 480, y: 480, label: 'Denver' },
    { x: 640, y: 560, label: 'Austin' },
    { x: 720, y: 380, label: 'Chicago' },
    { x: 880, y: 340, label: 'NYC' },
    { x: 860, y: 420, label: 'DC' },
    { x: 820, y: 540, label: 'Atlanta' },
    { x: 780, y: 600, label: 'Miami' },
  ];

  return (
    <div style={{
      position: 'absolute', inset: 0, opacity: enter,
      background: AC.ink,
    }}>
      {/* Starfield */}
      <div style={{
        position: 'absolute', inset: 0,
        background: `radial-gradient(ellipse at 60% 50%, rgba(15,110,140,0.25), transparent 60%)`,
      }}/>

      <div style={{
        position: 'absolute', top: 100, left: 80,
        fontFamily: AF.mono, fontSize: 15, fontWeight: 700,
        letterSpacing: '0.22em', textTransform: 'uppercase',
        color: AC.tealLight,
      }}>The Product ▸ 07</div>

      <div style={{
        position: 'absolute', top: 150, left: 80, right: 960,
        fontFamily: AF.body, fontSize: 88, fontWeight: 800,
        letterSpacing: '-0.035em', lineHeight: 1, color: '#fff',
      }}>
        Every provider.<br/>
        Every code.<br/>
        <span style={{ color: AC.tealLight }}>One map.</span>
      </div>

      {/* Subtitle */}
      <div style={{
        position: 'absolute', top: 540, left: 80, right: 960,
        fontFamily: AF.body, fontSize: 24, lineHeight: 1.5, color: '#C7C2BD',
      }}>
        Predict allowed amounts for any HCPCS code, state, and
        provider specialty — in real time. Built on 11 years of
        CMS filings, 1.26M providers, 10.4M procedure rows.
      </div>

      {/* CTA */}
      <div style={{
        position: 'absolute', top: 740, left: 80,
        display: 'flex', gap: 16, alignItems: 'center',
      }}>
        <a href="https://allowancemap.vercel.app" target="_blank" rel="noopener"
          style={{
            background: AC.tealLight, color: AC.ink,
            padding: '18px 32px', fontFamily: AF.body,
            fontSize: 20, fontWeight: 700, letterSpacing: '-0.01em',
            textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: 10,
            cursor: 'pointer', boxShadow: `0 8px 24px rgba(28,156,196,0.35)`,
          }}>
          <span>allowancemap.vercel.app ↗</span>
        </a>
        <a href="https://github.com/Prodoorknob/medicare-provider-utilization-cost-analysis-"
          target="_blank" rel="noopener" style={{
          fontFamily: AF.mono, fontSize: 14, letterSpacing: '0.18em',
          textTransform: 'uppercase', color: AC.ink3,
          textDecoration: 'none', borderBottom: `1px dashed ${AC.ink3}`,
        }}>github.com/Prodoorknob ↗</a>
      </div>

      {/* Map */}
      <svg viewBox="0 0 1100 720" style={{
        position: 'absolute', right: 0, top: 140, width: 960, height: 720,
      }}>
        {/* USA rough outline — stylized dots */}
        <USADots t={t} AC={AC} />

        {/* Highlighted provider pulses */}
        {cities.map((c, i) => {
          const delay = 0.6 + i * 0.15;
          if (t < delay) return null;
          const pulse = ((t - delay) % 1.5) / 1.5;
          return (
            <g key={c.label}>
              <circle cx={c.x} cy={c.y} r={6 + pulse * 18}
                fill="none" stroke={AC.tealLight}
                strokeWidth="2" opacity={(1 - pulse) * 0.8}/>
              <circle cx={c.x} cy={c.y} r="5" fill={AC.tealLight}/>
              <text x={c.x + 12} y={c.y + 5}
                fontFamily={AF.mono} fontSize="11" fill="#fff"
                letterSpacing="2" opacity="0.8">{c.label.toUpperCase()}</text>
            </g>
          );
        })}

        {/* Connecting lines (subtle) */}
        {cities.map((c, i) => {
          if (i === 0) return null;
          const delay = 1.5 + i * 0.1;
          if (t < delay) return null;
          const pc = cities[i - 1];
          const p = Math.min((t - delay) / 0.8, 1);
          return (
            <line key={i} x1={pc.x} y1={pc.y} x2={c.x} y2={c.y}
              stroke={AC.tealLight} strokeWidth="1"
              strokeDasharray={`${p * 400} 400`} opacity="0.3"/>
          );
        })}
      </svg>

      {/* Final wordmark */}
      <div style={{
        position: 'absolute', bottom: 80, left: 80, right: 80,
        display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
        borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: 18,
      }}>
        <span style={{
          fontFamily: AF.body, fontSize: 28, fontWeight: 800,
          color: '#fff', letterSpacing: '-0.02em',
        }}>AllowanceMap</span>
        <span style={{
          fontFamily: AF.mono, fontSize: 13, letterSpacing: '0.22em',
          textTransform: 'uppercase', color: AC.ink3,
        }}>Medicare · Provider · Cost · Intelligence</span>
        <span style={{
          fontFamily: AF.mono, fontSize: 13, letterSpacing: '0.2em',
          textTransform: 'uppercase', color: AC.tealLight,
        }}>2026 ▸ v2.4</span>
      </div>
    </div>
  );
}

function USADots({ t, AC }) {
  // A rough diagonal stripe pattern suggesting USA footprint
  const dots = [];
  const rows = 26, cols = 42;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      // Rough USA boundary mask (hand-tuned ellipse region)
      const nx = (c - 21) / 21;
      const ny = (r - 14) / 14;
      const inBounds = (nx * nx * 1.0 + ny * ny * 2.2) < 1;
      if (!inBounds) continue;
      // Carve out a crude shape variation
      if (ny < -0.6 && Math.abs(nx) > 0.3) continue;
      if (ny > 0.6 && nx < -0.3) continue;
      dots.push({
        x: 80 + c * 22 + (r % 2) * 11,
        y: 100 + r * 22,
        d: (r * cols + c) * 0.02,
      });
    }
  }
  return (
    <g opacity="0.4">
      {dots.map((d, i) => {
        const vis = Math.min(Math.max((t - d.d) / 0.3, 0), 1);
        return (
          <circle key={i} cx={d.x} cy={d.y} r={1.6 * vis}
            fill="#fff" opacity={vis * 0.35}/>
        );
      })}
    </g>
  );
}

Object.assign(window, { SceneBug, SceneResults, BigMetric, SceneProduct, USADots });
