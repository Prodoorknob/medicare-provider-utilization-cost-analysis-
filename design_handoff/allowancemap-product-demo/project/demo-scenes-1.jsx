/* Product Demo — Scenes 1-3: Hook, Cost Estimator, OOP Distribution */

// Brand palette — AllowanceMap
const AC = {
  bg: '#FAFAF8', surface: '#FFFFFF', surface2: '#F2F0ED',
  ink: '#1C1917', ink2: '#57534E', ink3: '#A8A29E',
  teal: '#0F6E8C', tealLight: '#1389AC', tealDark: '#0A4F66', tealTint: '#EBF7FB',
  green: '#15755D', greenLight: '#1CA082', greenTint: '#E8F7F3',
  amber: '#B8763A', amberTint: '#FDF4EA',
  red: '#DC2626', orange: '#EA580C',
  line: 'rgba(0,0,0,0.08)', line2: 'rgba(0,0,0,0.15)',
};
const AF = { body: 'Inter, system-ui, sans-serif', mono: '"IBM Plex Mono", ui-monospace, monospace' };

// Tiny helper: render the AllowanceMap mark (hospital cross in rounded square)
function AMLogo({ size = 36, color = AC.teal }) {
  return (
    <div style={{
      width: size, height: size, borderRadius: size * 0.22,
      background: color, display: 'flex', alignItems: 'center', justifyContent: 'center',
      color: '#fff', fontFamily: AF.body, fontWeight: 800,
      fontSize: size * 0.58, letterSpacing: '-0.04em',
    }}>A</div>
  );
}

// Browser chrome shell — framing for UI mockups
function BrowserShell({ children, url = 'allowancemap.vercel.app', tab = 'Cost Estimator' }) {
  return (
    <div style={{
      position: 'absolute', inset: '80px 80px 80px 80px',
      background: AC.bg, borderRadius: 14, overflow: 'hidden',
      boxShadow: '0 40px 100px rgba(0,0,0,0.12), 0 0 0 1px rgba(0,0,0,0.05)',
      display: 'flex', flexDirection: 'column',
    }}>
      {/* Title bar */}
      <div style={{
        height: 42, background: '#EEEBE6', display: 'flex', alignItems: 'center',
        padding: '0 16px', gap: 10, borderBottom: `1px solid ${AC.line}`,
      }}>
        <div style={{ display: 'flex', gap: 7 }}>
          {['#FF5F57', '#FEBC2E', '#28C840'].map(c => (
            <div key={c} style={{ width: 13, height: 13, borderRadius: '50%', background: c }}/>
          ))}
        </div>
        <div style={{
          marginLeft: 20, background: '#FFFFFF', borderRadius: 8,
          padding: '5px 14px', display: 'flex', alignItems: 'center', gap: 8,
          fontFamily: AF.mono, fontSize: 13, color: AC.ink2, minWidth: 320,
          border: `1px solid ${AC.line}`,
        }}>
          <span style={{ color: AC.green }}>●</span>
          <span>{url}</span>
        </div>
      </div>
      {/* Nav */}
      <div style={{
        height: 64, background: AC.surface, borderBottom: `1px solid ${AC.line}`,
        display: 'flex', alignItems: 'center', padding: '0 32px', gap: 32,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <AMLogo size={28}/>
          <span style={{ fontFamily: AF.body, fontWeight: 700, fontSize: 17, color: AC.ink }}>
            AllowanceMap
          </span>
        </div>
        <div style={{ flex: 1 }}/>
        <nav style={{ display: 'flex', gap: 28, fontFamily: AF.body, fontSize: 14 }}>
          {['Cost Estimator', 'Forecast', 'Investigations', 'About'].map(n => (
            <span key={n} style={{
              color: n === tab ? AC.teal : AC.ink2,
              fontWeight: n === tab ? 600 : 500,
              borderBottom: n === tab ? `2px solid ${AC.teal}` : '2px solid transparent',
              paddingBottom: 3,
            }}>{n}</span>
          ))}
        </nav>
      </div>
      {/* Content */}
      <div style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
        {children}
      </div>
    </div>
  );
}

// ── SCENE 1: Hook ─────────────────────────────────────────────────
function SceneHook() {
  const { localTime: t, duration } = useSprite();

  const exit = Math.max(0, Math.min(1, (t - (duration - 0.6)) / 0.6));
  const q1 = Math.min(Math.max((t - 0.1) / 0.5, 0), 1);
  const q2 = Math.min(Math.max((t - 0.9) / 0.5, 0), 1);
  const ans = Math.min(Math.max((t - 1.7) / 0.7, 0), 1);
  const pulse = Math.sin(t * 3) * 0.02;

  return (
    <div style={{
      position: 'absolute', inset: 0,
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      opacity: 1 - exit, background: AC.bg,
    }}>
      {/* Top eyebrow */}
      <div style={{
        opacity: q1,
        transform: `translateY(${(1 - q1) * 12}px)`,
        fontFamily: AF.mono, fontSize: 14, fontWeight: 600,
        letterSpacing: '0.28em', textTransform: 'uppercase',
        color: AC.teal, marginBottom: 40,
        display: 'flex', alignItems: 'center', gap: 14,
      }}>
        <AMLogo size={32}/>
        <span>AllowanceMap</span>
      </div>

      {/* Huge question */}
      <div style={{
        fontFamily: AF.body, fontSize: 128, fontWeight: 800,
        letterSpacing: '-0.04em', lineHeight: 1,
        color: AC.ink, textAlign: 'center',
      }}>
        <div style={{
          opacity: q1,
          transform: `translateY(${(1 - Easing.easeOutCubic(q1)) * 30}px)`,
        }}>
          How much will
        </div>
        <div style={{
          opacity: q2,
          transform: `translateY(${(1 - Easing.easeOutCubic(q2)) * 30}px)`,
          marginTop: 8,
        }}>
          Medicare <span style={{ color: AC.teal }}>really</span> cost?
        </div>
      </div>

      {/* Answer teaser */}
      <div style={{
        marginTop: 64,
        opacity: ans,
        transform: `scale(${0.92 + Easing.easeOutBack(ans) * 0.08 + pulse})`,
        display: 'flex', alignItems: 'baseline', gap: 16,
        background: AC.surface, padding: '20px 36px',
        border: `1px solid ${AC.line}`,
        borderLeft: `5px solid ${AC.teal}`,
        boxShadow: '0 12px 40px rgba(0,0,0,0.08)',
        borderRadius: 12,
      }}>
        <span style={{
          fontFamily: AF.mono, fontSize: 13, letterSpacing: '0.18em',
          textTransform: 'uppercase', color: AC.ink3,
        }}>answer</span>
        <span style={{
          fontFamily: AF.mono, fontSize: 56, fontWeight: 700,
          color: AC.teal, letterSpacing: '-0.02em', lineHeight: 1,
          fontVariantNumeric: 'tabular-nums',
        }}>in 2 seconds ↓</span>
      </div>
    </div>
  );
}

// ── SCENE 2: Cost Estimator — live form fill ─────────────────────
function SceneEstimator() {
  const { localTime: t, duration } = useSprite();
  const enter = Easing.easeOutCubic(Math.min(t / 0.5, 1));
  const exit = Math.max(0, Math.min(1, (t - (duration - 0.5)) / 0.5));

  // Form field reveal timing
  const f1 = Math.min(Math.max((t - 0.5) / 0.35, 0), 1);
  const f2 = Math.min(Math.max((t - 0.95) / 0.35, 0), 1);
  const f3 = Math.min(Math.max((t - 1.4) / 0.35, 0), 1);
  const f4 = Math.min(Math.max((t - 1.85) / 0.35, 0), 1);
  const btn = Math.min(Math.max((t - 2.4) / 0.4, 0), 1);
  const submit = t > 3.2;
  const resultP = Math.min(Math.max((t - 3.3) / 1.5, 0), 1);

  // Ticker to final allowed amount ($87.24)
  const allowed = 0 + Easing.easeOutCubic(resultP) * 87.24;
  const actualAmt = 0 + Easing.easeOutCubic(Math.max(0, (t - 3.7) / 1.2)) * 76.84;

  return (
    <div style={{
      position: 'absolute', inset: 0, opacity: enter * (1 - exit),
      background: AC.bg,
    }}>
      <BrowserShell tab="Cost Estimator">
        <div style={{
          padding: '36px 56px',
          display: 'grid',
          gridTemplateColumns: '5fr 7fr',
          gap: 32, height: '100%',
        }}>
          {/* Left: Form */}
          <div>
            <div style={{
              fontFamily: AF.mono, fontSize: 12, fontWeight: 700,
              letterSpacing: '0.22em', textTransform: 'uppercase',
              color: AC.teal,
            }}>STAGE 1 · PROVIDER DETAILS</div>
            <div style={{
              fontFamily: AF.body, fontSize: 32, fontWeight: 800,
              letterSpacing: '-0.02em', color: AC.ink,
              marginTop: 6, marginBottom: 32,
            }}>Medicare Cost Estimator</div>

            <FormField label="HCPCS Code" value="99213 — Office visit, established" reveal={f1} highlight={AC.teal}/>
            <FormField label="State" value="New York" reveal={f2}/>
            <FormField label="Specialty" value="Internal Medicine" reveal={f3}/>
            <FormField label="Place of Service" value="Office (11)" reveal={f4}/>

            {/* CTA button */}
            <div style={{
              marginTop: 24,
              opacity: btn,
              transform: `translateY(${(1 - btn) * 12}px)`,
            }}>
              <div style={{
                display: 'inline-flex', alignItems: 'center', gap: 10,
                background: submit ? AC.tealDark : AC.teal, color: '#fff',
                padding: '14px 28px', borderRadius: 8,
                fontFamily: AF.body, fontSize: 16, fontWeight: 600,
                transform: submit ? 'scale(0.98)' : 'scale(1)',
                transition: 'transform 100ms, background 150ms',
                boxShadow: submit ? 'none' : '0 4px 16px rgba(15,110,140,0.25)',
              }}>
                Estimate Costs
                <span style={{ opacity: 0.9 }}>→</span>
              </div>
            </div>
          </div>

          {/* Right: Result card */}
          <div style={{ position: 'relative' }}>
            {/* Placeholder when no result */}
            {!submit && (
              <div style={{
                position: 'absolute', inset: 0,
                border: `2px dashed ${AC.line2}`, borderRadius: 12,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontFamily: AF.mono, fontSize: 13, letterSpacing: '0.18em',
                textTransform: 'uppercase', color: AC.ink3,
              }}>
                waiting for input …
              </div>
            )}

            {submit && (
              <div style={{
                position: 'absolute', inset: 0,
                opacity: resultP,
                transform: `translateY(${(1 - Easing.easeOutCubic(resultP)) * 20}px)`,
              }}>
                {/* Big allowed amount card */}
                <div style={{
                  background: AC.surface, border: `1px solid ${AC.line}`,
                  borderLeft: `4px solid ${AC.teal}`,
                  padding: 32, borderRadius: 12,
                  boxShadow: '0 4px 16px rgba(0,0,0,0.08)',
                }}>
                  <div style={{
                    fontFamily: AF.mono, fontSize: 11, fontWeight: 700,
                    letterSpacing: '0.22em', textTransform: 'uppercase',
                    color: AC.teal,
                  }}>MEDICARE ALLOWED AMOUNT</div>
                  <div style={{
                    fontFamily: AF.mono, fontSize: 88, fontWeight: 700,
                    color: AC.teal, letterSpacing: '-0.03em',
                    lineHeight: 1, marginTop: 12,
                    fontVariantNumeric: 'tabular-nums',
                  }}>
                    ${allowed.toFixed(2)}
                  </div>
                  <div style={{
                    fontFamily: AF.body, fontSize: 14, color: AC.ink2,
                    marginTop: 10,
                  }}>
                    Predicted amount Medicare allows for this service
                  </div>

                  {/* Sub row: actual vs predicted */}
                  <div style={{
                    marginTop: 24, paddingTop: 20,
                    borderTop: `1px solid ${AC.line}`,
                    display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16,
                  }}>
                    {[
                      { label: 'CMS Actual', value: `$${actualAmt.toFixed(2)}`, color: AC.ink },
                      { label: 'Model MAE', value: '$7.70', color: AC.ink2 },
                      { label: 'Confidence', value: 'R² 0.94', color: AC.green },
                    ].map(s => (
                      <div key={s.label}>
                        <div style={{
                          fontFamily: AF.mono, fontSize: 10,
                          letterSpacing: '0.18em', textTransform: 'uppercase',
                          color: AC.ink3, marginBottom: 4,
                        }}>{s.label}</div>
                        <div style={{
                          fontFamily: AF.mono, fontSize: 20, fontWeight: 600,
                          color: s.color, fontVariantNumeric: 'tabular-nums',
                        }}>{s.value}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Scanning line (model inference) */}
                {resultP < 1 && (
                  <div style={{
                    position: 'absolute', left: 0, right: 0,
                    top: `${resultP * 100}%`, height: 2,
                    background: `linear-gradient(90deg, transparent, ${AC.teal}, transparent)`,
                    opacity: 0.8,
                  }}/>
                )}

                {/* Tag */}
                <div style={{
                  marginTop: 16,
                  display: 'flex', gap: 8,
                  opacity: Math.min((t - 4.2) / 0.4, 1),
                }}>
                  <span style={{
                    padding: '6px 12px', borderRadius: 999,
                    background: AC.greenTint, color: AC.green,
                    fontFamily: AF.mono, fontSize: 11, fontWeight: 600,
                    letterSpacing: '0.14em', textTransform: 'uppercase',
                  }}>● Real-time ML inference</span>
                  <span style={{
                    padding: '6px 12px', borderRadius: 999,
                    background: AC.tealTint, color: AC.teal,
                    fontFamily: AF.mono, fontSize: 11, fontWeight: 600,
                    letterSpacing: '0.14em', textTransform: 'uppercase',
                  }}>103M+ CMS records</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </BrowserShell>

      {/* Caption strip */}
      <div style={{
        position: 'absolute', bottom: 30, left: 0, right: 0,
        display: 'flex', justifyContent: 'center',
        fontFamily: AF.body, fontSize: 20, fontWeight: 600,
        color: AC.ink2, letterSpacing: '-0.01em',
      }}>
        <span style={{ color: AC.teal, fontWeight: 700 }}>Type a service.</span>
        <span style={{ margin: '0 10px', color: AC.ink3 }}>·</span>
        <span>Know the price.</span>
      </div>
    </div>
  );
}

function FormField({ label, value, reveal, highlight }) {
  const typedLen = Math.floor(reveal * value.length * 1.1);
  const typed = value.slice(0, Math.min(typedLen, value.length));
  const showCaret = reveal > 0 && reveal < 1;
  return (
    <div style={{
      marginBottom: 16,
      opacity: Math.min(reveal * 3, 1),
      transform: `translateY(${(1 - Math.min(reveal * 2, 1)) * 8}px)`,
    }}>
      <div style={{
        fontFamily: AF.mono, fontSize: 11, fontWeight: 700,
        letterSpacing: '0.18em', textTransform: 'uppercase',
        color: AC.ink3, marginBottom: 6,
      }}>{label}</div>
      <div style={{
        background: AC.surface, border: `1.5px solid ${highlight || AC.line2}`,
        borderRadius: 8, padding: '12px 14px',
        fontFamily: AF.body, fontSize: 15, color: AC.ink,
        minHeight: 44, display: 'flex', alignItems: 'center',
        boxShadow: highlight ? `0 0 0 3px ${AC.tealTint}` : 'none',
      }}>
        {typed}
        {showCaret && (
          <span style={{
            display: 'inline-block', width: 2, height: 18,
            background: AC.teal, marginLeft: 2,
            animation: 'blink 0.8s step-end infinite',
          }}/>
        )}
      </div>
    </div>
  );
}

// ── SCENE 3: OOP distribution ─────────────────────────────────────
function SceneOOP() {
  const { localTime: t, duration } = useSprite();
  const enter = Easing.easeOutCubic(Math.min(t / 0.5, 1));
  const exit = Math.max(0, Math.min(1, (t - (duration - 0.5)) / 0.5));

  // Curve draw-in
  const curveP = Math.min(Math.max((t - 0.4) / 1.5, 0), 1);
  const p10P = Math.min(Math.max((t - 1.6) / 0.5, 0), 1);
  const p50P = Math.min(Math.max((t - 2.1) / 0.5, 0), 1);
  const p90P = Math.min(Math.max((t - 2.6) / 0.5, 0), 1);
  const labelsP = Math.min(Math.max((t - 3.1) / 0.4, 0), 1);

  // Curve geometry — bell curve, right skew
  const W = 1160, H = 340;
  const curveFn = (x) => {
    // skewed normal, mode near x=0.28
    const k = 8;
    const y = Math.exp(-k * Math.pow(x - 0.28, 2)) * 0.88
            + Math.exp(-18 * Math.pow(x - 0.55, 2)) * 0.15;
    return y;
  };
  const N = 120;
  const pts = [];
  for (let i = 0; i <= N; i++) {
    const x = i / N;
    pts.push([x * W, H - curveFn(x) * H]);
  }
  const pathLen = N;
  const visPts = Math.floor(pathLen * curveP);
  const dPath = pts.slice(0, visPts + 1).map((p, i) => (i === 0 ? `M${p[0]},${H}L${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(' ')
             + (visPts < pathLen ? `L${pts[visPts][0]},${H}Z` : `L${W},${H}Z`);

  const x10 = 0.14 * W, x50 = 0.32 * W, x90 = 0.62 * W;
  const v10 = curveFn(0.14), v50 = curveFn(0.32), v90 = curveFn(0.62);

  return (
    <div style={{
      position: 'absolute', inset: 0, opacity: enter * (1 - exit),
      background: AC.bg,
    }}>
      {/* Eyebrow */}
      <div style={{
        position: 'absolute', top: 80, left: 120,
        fontFamily: AF.mono, fontSize: 13, fontWeight: 700,
        letterSpacing: '0.22em', textTransform: 'uppercase',
        color: AC.green,
      }}>STAGE 2 · YOUR OUT-OF-POCKET</div>

      {/* Headline */}
      <div style={{
        position: 'absolute', top: 120, left: 120, right: 120,
        fontFamily: AF.body, fontSize: 68, fontWeight: 800,
        letterSpacing: '-0.03em', lineHeight: 1.04, color: AC.ink,
      }}>
        What <span style={{ color: AC.green }}>you'll pay</span> — tuned to you.
      </div>
      <div style={{
        position: 'absolute', top: 244, left: 120, right: 700,
        fontFamily: AF.body, fontSize: 22, lineHeight: 1.45, color: AC.ink2,
      }}>
        Age, income, dual eligibility, supplemental plan, region —
        all feed a quantile model for your <b>P10/P50/P90</b> out-of-pocket range.
      </div>

      {/* Distribution card */}
      <div style={{
        position: 'absolute', top: 400, left: 120, right: 120, bottom: 100,
        background: AC.surface, border: `1px solid ${AC.line}`,
        borderLeft: `4px solid ${AC.green}`,
        borderRadius: 12, padding: '32px 40px',
        boxShadow: '0 8px 32px rgba(0,0,0,0.06)',
        display: 'flex', flexDirection: 'column',
      }}>
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
        }}>
          <div>
            <div style={{
              fontFamily: AF.mono, fontSize: 11, fontWeight: 700,
              letterSpacing: '0.2em', textTransform: 'uppercase',
              color: AC.green,
            }}>OOP DISTRIBUTION · 72-YO · NY · MEDIGAP G</div>
            <div style={{
              fontFamily: AF.body, fontSize: 24, fontWeight: 700,
              color: AC.ink, marginTop: 4,
            }}>Your expected out-of-pocket range</div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{
              fontFamily: AF.mono, fontSize: 11, fontWeight: 700,
              letterSpacing: '0.2em', textTransform: 'uppercase',
              color: AC.ink3,
            }}>MEDIAN (P50)</div>
            <div style={{
              fontFamily: AF.mono, fontSize: 48, fontWeight: 700,
              color: AC.green, lineHeight: 1,
              fontVariantNumeric: 'tabular-nums',
            }}>$18.42</div>
          </div>
        </div>

        {/* Chart */}
        <div style={{ flex: 1, marginTop: 20, position: 'relative' }}>
          <svg viewBox={`-20 -20 ${W + 40} ${H + 40}`}
               preserveAspectRatio="none"
               style={{ width: '100%', height: '100%' }}>
            {/* Grid lines */}
            {[0.25, 0.5, 0.75].map((g, i) => (
              <line key={i} x1="0" y1={H * g} x2={W} y2={H * g}
                stroke={AC.line} strokeWidth="1" strokeDasharray="3 6"/>
            ))}
            {/* Filled curve */}
            <path d={dPath} fill={AC.green} fillOpacity="0.12" stroke="none"/>
            <path d={pts.slice(0, visPts + 1).map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(' ')}
                  fill="none" stroke={AC.green} strokeWidth="3"/>

            {/* P markers */}
            {[
              { x: x10, y: v10 * H, label: 'P10', val: '$4.20', p: p10P, c: AC.tealLight },
              { x: x50, y: v50 * H, label: 'P50', val: '$18.42', p: p50P, c: AC.green },
              { x: x90, y: v90 * H, label: 'P90', val: '$52.77', p: p90P, c: AC.amber },
            ].map((m, i) => (
              <g key={i} opacity={m.p}>
                <line x1={m.x} y1={H - m.y} x2={m.x} y2={H}
                  stroke={m.c} strokeWidth="2" strokeDasharray="4 4"/>
                <circle cx={m.x} cy={H - m.y} r="7" fill={m.c}
                        stroke="#fff" strokeWidth="3"/>
                <rect x={m.x - 42} y={H - m.y - 48} width="84" height="36"
                      fill={m.c} rx="4"/>
                <text x={m.x} y={H - m.y - 30}
                  fontFamily={AF.mono} fontSize="10" fontWeight="700"
                  letterSpacing="2" fill="#fff" textAnchor="middle">{m.label}</text>
                <text x={m.x} y={H - m.y - 16}
                  fontFamily={AF.mono} fontSize="14" fontWeight="700"
                  fill="#fff" textAnchor="middle">{m.val}</text>
              </g>
            ))}

            {/* Axis tick labels */}
            {[0, 0.25, 0.5, 0.75, 1.0].map((g, i) => (
              <text key={i} x={g * W} y={H + 20}
                fontFamily={AF.mono} fontSize="11" fill={AC.ink3}
                textAnchor="middle">
                {`$${Math.round(g * 120)}`}
              </text>
            ))}
          </svg>
        </div>

        {/* Bottom callout row */}
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 24,
          marginTop: 12, opacity: labelsP,
        }}>
          {[
            ['Best case', '$4.20', 'P10', AC.tealLight],
            ['Typical', '$18.42', 'P50', AC.green],
            ['Worst case', '$52.77', 'P90', AC.amber],
          ].map(([label, val, tag, color]) => (
            <div key={tag} style={{
              borderTop: `2px solid ${color}`, paddingTop: 10,
              display: 'flex', alignItems: 'baseline', gap: 10,
            }}>
              <div style={{
                fontFamily: AF.mono, fontSize: 10, fontWeight: 700,
                letterSpacing: '0.2em', textTransform: 'uppercase',
                color: AC.ink3, minWidth: 80,
              }}>{label}</div>
              <div style={{
                fontFamily: AF.mono, fontSize: 22, fontWeight: 700,
                color, fontVariantNumeric: 'tabular-nums',
              }}>{val}</div>
              <div style={{
                fontFamily: AF.mono, fontSize: 10,
                letterSpacing: '0.2em', color: AC.ink3,
              }}>{tag}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { AC, AF, AMLogo, BrowserShell, SceneHook, SceneEstimator, SceneOOP, FormField });
