/* Product Demo — Scenes 4-5: Forecast, Fraud Agent */

// ── SCENE 4: Forecast chart ───────────────────────────────────────
function SceneForecast() {
  const { localTime: t, duration } = useSprite();
  const enter = Easing.easeOutCubic(Math.min(t / 0.5, 1));
  const exit = Math.max(0, Math.min(1, (t - (duration - 0.5)) / 0.5));

  // Chart draw progress
  const histP = Math.min(Math.max((t - 0.3) / 1.4, 0), 1);
  const fcastP = Math.min(Math.max((t - 1.7) / 1.4, 0), 1);
  const bandP = Math.min(Math.max((t - 2.0) / 1.2, 0), 1);
  const cardsP = Math.min(Math.max((t - 3.2) / 0.5, 0), 1);

  // Historical data points (2013–2023), stylized dollars
  const hist = [92.10, 94.20, 96.80, 99.50, 101.20, 103.40, 105.80, 107.60, 109.20, 111.70, 114.30];
  // Forecast 2024–2026 with uncertainty
  const fcast = [117.20, 120.40, 123.80];
  const p10F = [114.50, 116.20, 118.40];
  const p90F = [119.80, 124.60, 129.20];

  const years = [...Array(14).keys()].map(i => 2013 + i);
  const allYears = hist.map((v, i) => ({ year: years[i], val: v, hist: true }))
    .concat(fcast.map((v, i) => ({ year: years[11 + i], val: v, p10: p10F[i], p90: p90F[i] })));

  const CW = 1400, CH = 420;
  const padL = 80, padR = 40, padT = 20, padB = 50;
  const iw = CW - padL - padR, ih = CH - padT - padB;
  const minY = 85, maxY = 135;
  const xAt = (i) => padL + (i / 13) * iw;
  const yAt = (v) => padT + (1 - (v - minY) / (maxY - minY)) * ih;

  // Historical line visible up to histP
  const histVis = Math.floor(histP * 11);
  const histPath = hist.slice(0, histVis + 1).map((v, i) =>
    (i === 0 ? 'M' : 'L') + xAt(i) + ',' + yAt(v)).join(' ');

  // Forecast line
  const fcastFull = [hist[10], ...fcast];
  const fcVis = fcastP * 3;
  const fcPath = [];
  for (let i = 0; i < fcastFull.length; i++) {
    const fracVisible = Math.min(Math.max(fcVis - (i - 1), 0), 1);
    if (fracVisible <= 0) break;
    const prevX = i === 0 ? xAt(10) : xAt(10 + i - 1);
    const prevY = i === 0 ? yAt(fcastFull[0]) : yAt(fcastFull[i - 1]);
    const curX = xAt(10 + i);
    const curY = yAt(fcastFull[i]);
    const x = prevX + (curX - prevX) * fracVisible;
    const y = prevY + (curY - prevY) * fracVisible;
    if (i === 0) fcPath.push(`M${x},${y}`); else fcPath.push(`L${x},${y}`);
  }

  // Confidence band path
  const bandPoints = [];
  for (let i = 0; i < 3; i++) {
    bandPoints.push([xAt(11 + i), yAt(p10F[i])]);
  }
  const bandPointsRev = [];
  for (let i = 2; i >= 0; i--) {
    bandPointsRev.push([xAt(11 + i), yAt(p90F[i])]);
  }
  const bandStartX = xAt(10), bandStartY = yAt(hist[10]);
  const bandPath = `M${bandStartX},${bandStartY} `
    + bandPoints.map(p => `L${p[0]},${p[1]}`).join(' ')
    + ' ' + bandPointsRev.map(p => `L${p[0]},${p[1]}`).join(' ')
    + ` L${bandStartX},${bandStartY} Z`;

  return (
    <div style={{
      position: 'absolute', inset: 0, opacity: enter * (1 - exit),
      background: AC.bg,
    }}>
      <BrowserShell tab="Forecast">
        <div style={{ padding: '28px 48px', height: '100%', display: 'flex', flexDirection: 'column' }}>
          <div style={{
            fontFamily: AF.mono, fontSize: 12, fontWeight: 700,
            letterSpacing: '0.22em', textTransform: 'uppercase',
            color: AC.teal,
          }}>FORECAST EXPLORER · INTERNAL MEDICINE · NY</div>
          <div style={{
            fontFamily: AF.body, fontSize: 34, fontWeight: 800,
            letterSpacing: '-0.02em', color: AC.ink, marginTop: 4,
          }}>Where Medicare rates are <span style={{ color: AC.teal }}>headed</span></div>

          {/* Chart */}
          <div style={{ flex: 1, marginTop: 16, position: 'relative' }}>
            <svg viewBox={`0 0 ${CW} ${CH}`} preserveAspectRatio="none"
                 style={{ width: '100%', height: '100%' }}>
              {/* Grid */}
              {[95, 105, 115, 125, 135].map(v => (
                <g key={v}>
                  <line x1={padL} y1={yAt(v)} x2={CW - padR} y2={yAt(v)}
                    stroke={AC.line} strokeDasharray="3 6"/>
                  <text x={padL - 10} y={yAt(v) + 4}
                    fontFamily={AF.mono} fontSize="12" fill={AC.ink3}
                    textAnchor="end">${v}</text>
                </g>
              ))}
              {/* Historical vs forecast divider */}
              <line x1={xAt(10)} y1={padT} x2={xAt(10)} y2={CH - padB}
                stroke={AC.line2} strokeWidth="1.5" strokeDasharray="4 4"
                opacity={fcastP}/>
              <text x={xAt(10) - 6} y={padT + 14}
                fontFamily={AF.mono} fontSize="10" fontWeight="700"
                fill={AC.ink3} textAnchor="end" letterSpacing="2"
                opacity={fcastP}>HISTORY</text>
              <text x={xAt(10) + 6} y={padT + 14}
                fontFamily={AF.mono} fontSize="10" fontWeight="700"
                fill={AC.teal} textAnchor="start" letterSpacing="2"
                opacity={fcastP}>FORECAST</text>

              {/* Confidence band */}
              {bandP > 0 && (
                <path d={bandPath} fill={AC.teal} fillOpacity={bandP * 0.12}
                      stroke="none"/>
              )}

              {/* Historical line */}
              <path d={histPath} fill="none" stroke={AC.teal}
                    strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
              {hist.slice(0, histVis + 1).map((v, i) => (
                <circle key={i} cx={xAt(i)} cy={yAt(v)} r="4" fill={AC.teal}/>
              ))}

              {/* Forecast line */}
              <path d={fcPath.join(' ')} fill="none"
                    stroke={AC.green} strokeWidth="3"
                    strokeDasharray="8 5"
                    strokeLinecap="round" strokeLinejoin="round"/>
              {fcast.map((v, i) => {
                const vis = Math.min(Math.max(fcVis - i, 0), 1);
                return (
                  <circle key={i} cx={xAt(11 + i)} cy={yAt(v)} r={5 * vis}
                    fill={AC.green} stroke="#fff" strokeWidth="2"/>
                );
              })}

              {/* Year axis */}
              {[2013, 2016, 2019, 2023, 2026].map(y => {
                const i = y - 2013;
                return (
                  <text key={y} x={xAt(i)} y={CH - padB + 24}
                    fontFamily={AF.mono} fontSize="11" fill={AC.ink3}
                    textAnchor="middle" letterSpacing="1">{y}</text>
                );
              })}

              {/* P90 / P10 annotation */}
              {bandP > 0.5 && (
                <>
                  <text x={xAt(13) - 10} y={yAt(p90F[2]) - 6}
                    fontFamily={AF.mono} fontSize="11" fontWeight="700"
                    fill={AC.teal} textAnchor="end">P90</text>
                  <text x={xAt(13) - 10} y={yAt(p10F[2]) + 16}
                    fontFamily={AF.mono} fontSize="11" fontWeight="700"
                    fill={AC.teal} textAnchor="end">P10</text>
                </>
              )}
            </svg>
          </div>

          {/* Stat cards */}
          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16,
            marginTop: 12, opacity: cardsP,
            transform: `translateY(${(1 - cardsP) * 10}px)`,
          }}>
            {[
              { label: '2026 Forecast', val: '$123.80', sub: '+8.3% vs 2023', color: AC.teal },
              { label: '3-yr CAGR', val: '+2.7%', sub: 'steady growth', color: AC.green },
              { label: 'P10 — P90 band', val: '$118 – $129', sub: 'model uncertainty', color: AC.amber },
            ].map(c => (
              <div key={c.label} style={{
                background: AC.surface, border: `1px solid ${AC.line}`,
                borderLeft: `4px solid ${c.color}`,
                padding: '14px 18px', borderRadius: 8,
              }}>
                <div style={{
                  fontFamily: AF.mono, fontSize: 10, fontWeight: 700,
                  letterSpacing: '0.2em', textTransform: 'uppercase',
                  color: AC.ink3,
                }}>{c.label}</div>
                <div style={{
                  fontFamily: AF.mono, fontSize: 28, fontWeight: 700,
                  color: c.color, marginTop: 2, lineHeight: 1,
                  fontVariantNumeric: 'tabular-nums',
                }}>{c.val}</div>
                <div style={{
                  fontFamily: AF.body, fontSize: 12, color: AC.ink2,
                  marginTop: 4,
                }}>{c.sub}</div>
              </div>
            ))}
          </div>
        </div>
      </BrowserShell>
    </div>
  );
}

// ── SCENE 5: Fraud Agent (Claude) — the hero feature ──────────────
function SceneFraud() {
  const { localTime: t, duration } = useSprite();
  const enter = Easing.easeOutCubic(Math.min(t / 0.5, 1));
  const exit = Math.max(0, Math.min(1, (t - (duration - 0.5)) / 0.5));

  // Phases — pre-brief setup extended by 2s so viewers can read the hook
  const scanP = Math.min(t / 3.2, 1); // scanning providers (slower build)
  const flag = t > 3.2;
  const briefP = Math.min(Math.max((t - 3.8) / 0.6, 0), 1); // brief slides in
  const rulesP = Math.min(Math.max((t - 4.8) / 1.2, 0), 1); // rule chips reveal
  const actionsP = Math.min(Math.max((t - 6.6) / 0.5, 0), 1);

  // Number of rules revealed
  const rules = [
    { id: 'VOLUME_SPIKE', status: 'TRIGGERED', desc: '+9,655% YoY on 24,874 services' },
    { id: 'CHARGE_INFLATION', status: 'TRIGGERED', desc: 'charge/allowed ratio 25.51 · P100' },
    { id: 'HIGH_INTENSITY', status: 'TRIGGERED', desc: 'srvcs/bene 2.71 at P99' },
    { id: 'OUT_OF_SPECIALTY', status: 'NOT_TRIGGERED', desc: '10.2% — below 20% threshold' },
    { id: 'PROCEDURE_CONCENTRATION', status: 'NOT_TRIGGERED', desc: 'Herfindahl 0.255 · P12' },
    { id: 'IMPOSSIBLE_DAY', status: 'NOT_EVALUABLE', desc: 'no date-of-service field' },
  ];

  // Provider dot grid (scanning)
  const rows = 8, cols = 22;
  const dots = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c;
      const delay = (idx / (rows * cols)) * 2.6;
      const vis = Math.min(Math.max((t - delay) / 0.4, 0), 1);
      const flagged = idx === 97; // one specific cell lights up
      dots.push({ r, c, vis, flagged, idx });
    }
  }

  return (
    <div style={{
      position: 'absolute', inset: 0, opacity: enter * (1 - exit),
      background: '#0a0a0a',
    }}>
      {/* Dark gradient bg */}
      <div style={{
        position: 'absolute', inset: 0,
        background: `radial-gradient(ellipse at 30% 30%, rgba(220,38,38,0.12), transparent 55%), radial-gradient(ellipse at 75% 75%, rgba(15,110,140,0.15), transparent 60%)`,
      }}/>

      {/* Eyebrow */}
      <div style={{
        position: 'absolute', top: 80, left: 120,
        display: 'flex', alignItems: 'center', gap: 14,
        fontFamily: AF.mono, fontSize: 13, fontWeight: 700,
        letterSpacing: '0.22em', textTransform: 'uppercase',
        color: '#fff',
      }}>
        <div style={{
          width: 10, height: 10, borderRadius: '50%',
          background: flag ? AC.red : AC.orange,
          boxShadow: `0 0 12px ${flag ? AC.red : AC.orange}`,
          animation: 'pulse 1.2s ease-in-out infinite',
        }}/>
        <span style={{ color: flag ? AC.red : AC.orange }}>
          {flag ? 'FRAUD ALERT' : 'INVESTIGATIONS AGENT · SCANNING'}
        </span>
        <span style={{ color: AC.ink3 }}>· powered by Claude</span>
      </div>

      {/* Headline */}
      <div style={{
        position: 'absolute', top: 118, left: 120, right: 120,
        fontFamily: AF.body, fontSize: 68, fontWeight: 800,
        letterSpacing: '-0.03em', lineHeight: 1.02, color: '#fff',
      }}>
        {flag ? (
          <>One provider, <span style={{ color: AC.red }}>flagged</span> in seconds.</>
        ) : (
          <>1.26M providers. <span style={{ color: AC.tealLight }}>One AI.</span></>
        )}
      </div>

      {/* LEFT: provider grid */}
      <div style={{
        position: 'absolute', top: 280, left: 120,
        width: 620,
        opacity: briefP > 0.5 ? 0.35 : 1,
        transition: 'opacity 300ms',
      }}>
        <div style={{
          fontFamily: AF.mono, fontSize: 11, fontWeight: 700,
          letterSpacing: '0.2em', textTransform: 'uppercase',
          color: AC.ink3, marginBottom: 14,
        }}>
          PROVIDER UNIVERSE · CMS 2013–2023
        </div>
        <svg viewBox="0 0 620 220" style={{ width: '100%', height: 220 }}>
          {dots.map((d, i) => {
            const x = 10 + d.c * 28;
            const y = 10 + d.r * 26;
            let color = 'rgba(255,255,255,0.28)';
            if (d.flagged && flag) color = AC.red;
            else if (d.vis > 0.5 && ((d.idx * 13) % 37) === 0) color = AC.orange;
            const size = d.flagged && flag ? 5 + Math.sin(t * 6) * 2 : 3;
            return (
              <g key={i} opacity={d.vis}>
                <circle cx={x} cy={y} r={size} fill={color}/>
                {d.flagged && flag && (
                  <circle cx={x} cy={y} r={15 + Math.sin(t * 4) * 5}
                          fill="none" stroke={AC.red}
                          strokeWidth="1.5" opacity="0.5"/>
                )}
              </g>
            );
          })}
          {/* Scanning line */}
          {!flag && (
            <line x1="0" y1={scanP * 220} x2="620" y2={scanP * 220}
              stroke={AC.tealLight} strokeWidth="1.5"
              strokeDasharray="3 4" opacity="0.6"/>
          )}
        </svg>
        {/* Small meter */}
        <div style={{
          marginTop: 14, fontFamily: AF.mono, fontSize: 12,
          color: AC.ink3, display: 'flex', justifyContent: 'space-between',
        }}>
          <span>
            scanned: <span style={{ color: '#fff' }}>
              {Math.floor(scanP * 1260000).toLocaleString('en-US')}
            </span>
          </span>
          <span>
            flagged: <span style={{ color: flag ? AC.red : AC.orange }}>
              {flag ? '1' : '—'}
            </span>
          </span>
        </div>
      </div>

      {/* RIGHT: Investigation brief card */}
      {briefP > 0 && (
        <div style={{
          position: 'absolute', top: 260, right: 120,
          width: 820,
          opacity: briefP,
          transform: `translateX(${(1 - Easing.easeOutCubic(briefP)) * 40}px)`,
        }}>
          <div style={{
            background: AC.surface, borderRadius: 14,
            boxShadow: '0 30px 80px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.1)',
            overflow: 'hidden',
            borderLeft: `5px solid ${AC.red}`,
          }}>
            {/* Brief header */}
            <div style={{
              padding: '22px 28px',
              borderBottom: `1px solid ${AC.line}`,
              display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
            }}>
              <div>
                <div style={{
                  fontFamily: AF.mono, fontSize: 10, fontWeight: 700,
                  letterSpacing: '0.22em', textTransform: 'uppercase',
                  color: AC.red,
                }}>INVESTIGATION BRIEF · 2023</div>
                <div style={{
                  fontFamily: AF.body, fontSize: 22, fontWeight: 700,
                  color: AC.ink, marginTop: 4,
                }}>
                  NPI <span style={{ fontFamily: AF.mono, color: AC.teal }}>1295****51</span>
                </div>
                <div style={{
                  fontFamily: AF.body, fontSize: 14, color: AC.ink2, marginTop: 2,
                }}>Emergency Medicine · NY</div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{
                  display: 'inline-flex', alignItems: 'center', gap: 6,
                  padding: '6px 12px', borderRadius: 999,
                  background: AC.red, color: '#fff',
                  fontFamily: AF.mono, fontSize: 10, fontWeight: 700,
                  letterSpacing: '0.18em',
                }}>● CRITICAL · 88</div>
                <div style={{
                  fontFamily: AF.mono, fontSize: 10, color: AC.ink3,
                  marginTop: 6, letterSpacing: '0.18em',
                }}>RISK SCORE</div>
              </div>
            </div>

            {/* Claude-generated summary */}
            <div style={{
              padding: '18px 28px', background: AC.surface2,
              fontFamily: AF.body, fontSize: 14, lineHeight: 1.55,
              color: AC.ink, borderBottom: `1px solid ${AC.line}`,
            }}>
              <span style={{
                fontFamily: AF.mono, fontSize: 10, fontWeight: 700,
                letterSpacing: '0.2em', textTransform: 'uppercase',
                color: AC.ink3, marginRight: 8,
              }}>FINDING</span>
              Catastrophic volume spike on supply codes with a <b>25×
              charge-to-allowed ratio</b> — structurally incompatible with
              hospital-based emergency medicine practice. Prior-year dormancy
              rules out organic growth.
            </div>

            {/* Rule grid */}
            <div style={{ padding: '16px 28px' }}>
              <div style={{
                fontFamily: AF.mono, fontSize: 10, fontWeight: 700,
                letterSpacing: '0.22em', textTransform: 'uppercase',
                color: AC.ink3, marginBottom: 10,
              }}>RULE CHECKS</div>
              <div style={{
                display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8,
              }}>
                {rules.map((r, i) => {
                  const p = Math.min(Math.max((rulesP * rules.length) - i, 0), 1);
                  const colorFor = {
                    TRIGGERED: AC.red,
                    NOT_TRIGGERED: AC.green,
                    NOT_EVALUABLE: AC.ink3,
                  }[r.status];
                  return (
                    <div key={r.id} style={{
                      display: 'flex', alignItems: 'flex-start', gap: 8,
                      padding: '8px 10px', background: AC.bg,
                      borderLeft: `3px solid ${colorFor}`,
                      borderRadius: 4,
                      opacity: p,
                      transform: `translateY(${(1 - p) * 6}px)`,
                    }}>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{
                          fontFamily: AF.mono, fontSize: 10, fontWeight: 700,
                          letterSpacing: '0.16em', color: AC.ink,
                        }}>{r.id}</div>
                        <div style={{
                          fontFamily: AF.body, fontSize: 11, color: AC.ink2,
                          marginTop: 2, lineHeight: 1.3,
                        }}>{r.desc}</div>
                      </div>
                      <div style={{
                        fontFamily: AF.mono, fontSize: 9, fontWeight: 700,
                        letterSpacing: '0.12em',
                        color: colorFor, whiteSpace: 'nowrap',
                      }}>{r.status.replace('_', ' ')}</div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Actions */}
            <div style={{
              padding: '14px 28px', borderTop: `1px solid ${AC.line}`,
              display: 'flex', gap: 10, alignItems: 'center',
              opacity: actionsP,
            }}>
              <span style={{
                fontFamily: AF.mono, fontSize: 10, fontWeight: 700,
                letterSpacing: '0.2em', textTransform: 'uppercase',
                color: AC.ink3, marginRight: 'auto',
              }}>ANALYST ACTION</span>
              {[
                ['Approve', AC.green],
                ['Escalate', AC.red],
                ['Dismiss', AC.ink3],
              ].map(([label, c], i) => (
                <div key={label} style={{
                  padding: '8px 16px', borderRadius: 6,
                  border: `1.5px solid ${c}`,
                  color: label === 'Escalate' && actionsP > 0.5 ? '#fff' : c,
                  background: label === 'Escalate' && actionsP > 0.5 ? c : 'transparent',
                  fontFamily: AF.body, fontSize: 13, fontWeight: 600,
                  transition: 'all 200ms',
                }}>{label}</div>
              ))}
            </div>
          </div>

          {/* "Claude thinking" ambient label */}
          <div style={{
            marginTop: 14, display: 'flex', alignItems: 'center', gap: 10,
            fontFamily: AF.mono, fontSize: 11, letterSpacing: '0.18em',
            textTransform: 'uppercase', color: AC.ink3,
          }}>
            <svg width="16" height="16" viewBox="0 0 16 16">
              <circle cx="8" cy="8" r="3" fill={AC.orange}/>
              <circle cx="8" cy="8" r="6" fill="none" stroke={AC.orange}
                strokeWidth="1" opacity="0.5">
                <animate attributeName="r" from="3" to="7"
                         dur="1.2s" repeatCount="indefinite"/>
                <animate attributeName="opacity" from="0.6" to="0"
                         dur="1.2s" repeatCount="indefinite"/>
              </circle>
            </svg>
            Claude-generated · 6 rules · 1 analyst tap to escalate
          </div>
        </div>
      )}
    </div>
  );
}

Object.assign(window, { SceneForecast, SceneFraud });
