/* Scenes 4-5: Two-stage pipeline, Model zoo race */

// ── SCENE 4: Two-Stage Pipeline ─────────────────────────────────
function ScenePipeline() {
  const { localTime: t, duration } = useSprite();
  const enter = Easing.easeOutCubic(Math.min(t / 0.5, 1));
  const exit = Math.max(0, Math.min(1, (t - (duration - 0.6)) / 0.6));

  // Stages reveal
  const s1 = Math.min(Math.max((t - 0.4) / 0.6, 0), 1);
  const arr1 = Math.min(Math.max((t - 1.1) / 0.7, 0), 1);
  const s2 = Math.min(Math.max((t - 1.6) / 0.6, 0), 1);
  const arr2 = Math.min(Math.max((t - 2.3) / 0.7, 0), 1);
  const s3 = Math.min(Math.max((t - 2.8) / 0.6, 0), 1);

  // Packet flow along the pipeline
  const packetT = ((t - 1.5) % 2.2) / 2.2;
  const packetVis = t > 1.5 ? 1 : 0;

  // Positions (1920x1080)
  const Y = 540;
  const X1 = 260, X2 = 960, X3 = 1660;
  const BOX_W = 360, BOX_H = 220;

  return (
    <div style={{
      position: 'absolute', inset: 0, opacity: enter * (1 - exit),
    }}>
      <div style={{
        position: 'absolute', top: 100, left: 80,
        fontFamily: AF.mono, fontSize: 15, fontWeight: 700,
        letterSpacing: '0.22em', textTransform: 'uppercase',
        color: AC.green,
      }}>The Pipeline ▸ 03</div>

      <div style={{
        position: 'absolute', top: 150, left: 80, right: 80,
        fontFamily: AF.body, fontSize: 72, fontWeight: 800,
        letterSpacing: '-0.03em', lineHeight: 1.05, color: AC.ink,
      }}>
        Two stages, one truth:<br/>
        what Medicare <span style={{ color: AC.green }}>actually pays</span>.
      </div>

      {/* Pipeline SVG */}
      <svg viewBox="0 0 1920 1080" style={{
        position: 'absolute', inset: 0, width: '100%', height: '100%',
      }}>
        {/* Connecting rails */}
        <line x1={X1 + BOX_W/2} y1={Y} x2={X2 - BOX_W/2} y2={Y}
          stroke={AC.line2} strokeWidth="2"
          strokeDasharray={`${arr1 * 400} 400`}/>
        <line x1={X2 + BOX_W/2} y1={Y} x2={X3 - BOX_W/2} y2={Y}
          stroke={AC.line2} strokeWidth="2"
          strokeDasharray={`${arr2 * 400} 400`}/>

        {/* Flowing packets */}
        {packetVis > 0 && [0, 0.33, 0.66].map((off, i) => {
          const p = (packetT + off) % 1;
          const x = p < 0.5
            ? X1 + BOX_W/2 + (p * 2) * (X2 - X1 - BOX_W)
            : X2 + BOX_W/2 + ((p - 0.5) * 2) * (X3 - X2 - BOX_W);
          const color = p < 0.5 ? AC.teal : AC.green;
          return (
            <g key={i}>
              <circle cx={x} cy={Y} r="6" fill={color}/>
              <circle cx={x} cy={Y} r="12" fill={color} opacity="0.25"/>
            </g>
          );
        })}
      </svg>

      {/* Stage 1: Raw */}
      <PipelineBox
        x={X1 - BOX_W/2} y={Y - BOX_H/2} w={BOX_W} h={BOX_H}
        progress={s1}
        tag="INPUT"
        title="Submitted Charge"
        subtitle="Provider's list price"
        color={AC.teal}
        rows={[
          ['HCPCS', '99213'], ['State', 'NY'],
          ['Specialty', 'Internal Med'], ['$ billed', '$109.42'],
        ]}
      />

      {/* Stage 2: Model */}
      <PipelineBox
        x={X2 - BOX_W/2} y={Y - BOX_H/2} w={BOX_W} h={BOX_H}
        progress={s2}
        tag="MODEL"
        title="LightGBM V2 no-charge"
        subtitle="+ XGB Quantile V1 for OOP"
        color={AC.green}
        rows={[
          ['Stage 1', 'Allowed amt (prod)'],
          ['Stage 2', 'OOP quantile (P10/50/90)'],
          ['Features', '12 no-charge / 13 full'],
          ['Target', 'log1p(Avg_Mdcr_Alowd_Amt)'],
        ]}
        featured
      />

      {/* Stage 3: Output */}
      <PipelineBox
        x={X3 - BOX_W/2} y={Y - BOX_H/2} w={BOX_W} h={BOX_H}
        progress={s3}
        tag="OUTPUT"
        title="Allowed Amount"
        subtitle="What CMS actually pays"
        color={AC.amber}
        rows={[
          ['Predicted', '$76.84'],
          ['MAE', '$7.70'],
          ['R² (test)', '0.9428'],
          ['Actual', '$77.11'],
        ]}
      />

      {/* Bottom caption */}
      <div style={{
        position: 'absolute', bottom: 80, left: 80, right: 80,
        display: 'flex', justifyContent: 'space-between',
        fontFamily: AF.mono, fontSize: 13, letterSpacing: '0.14em',
        textTransform: 'uppercase', color: AC.ink3,
        opacity: s3,
      }}>
        <span>→ charges are noisy.  allowed amounts are policy.</span>
        <span>r² = 0.9428 · mae $7.70 · rmse $15.77</span>
      </div>
    </div>
  );
}

function PipelineBox({ x, y, w, h, progress, tag, title, subtitle, color, rows, featured }) {
  const scale = 0.85 + 0.15 * Easing.easeOutBack(progress);
  return (
    <div style={{
      position: 'absolute', left: x, top: y, width: w, height: h,
      opacity: progress, transform: `scale(${scale})`,
      transformOrigin: 'center', willChange: 'transform',
    }}>
      <div style={{
        width: '100%', height: '100%', background: '#fff',
        border: `2px solid ${color}`,
        boxShadow: featured
          ? `0 30px 80px rgba(21,117,93,0.25), 0 0 0 8px ${AC.greenTint}`
          : `0 8px 24px rgba(0,0,0,0.06)`,
        padding: 20, display: 'flex', flexDirection: 'column',
      }}>
        <div style={{
          fontFamily: AF.mono, fontSize: 11, fontWeight: 700,
          letterSpacing: '0.22em', color,
        }}>{tag}</div>
        <div style={{
          fontFamily: AF.body, fontSize: 26, fontWeight: 800,
          letterSpacing: '-0.02em', color: AC.ink, marginTop: 4,
        }}>{title}</div>
        <div style={{
          fontFamily: AF.body, fontSize: 14, color: AC.ink2, marginTop: 2,
        }}>{subtitle}</div>
        <div style={{
          marginTop: 14, borderTop: `1px solid ${AC.line}`, paddingTop: 10,
          display: 'grid', gridTemplateColumns: '1fr auto',
          gap: '6px 12px',
          fontFamily: AF.mono, fontSize: 13,
        }}>
          {rows.map((r, i) => (
            <React.Fragment key={i}>
              <span style={{ color: AC.ink3 }}>{r[0]}</span>
              <span style={{ color: AC.ink, fontWeight: 600 }}>{r[1]}</span>
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── SCENE 5: Model Zoo Race ─────────────────────────────────────
function SceneModelZoo() {
  const { localTime: t, duration } = useSprite();
  const enter = Easing.easeOutCubic(Math.min(t / 0.5, 1));
  const exit = Math.max(0, Math.min(1, (t - (duration - 0.6)) / 0.6));

  const models = [
    { name: 'RF V1 (30% sample)',  target: 0.8843, color: AC.ink3,      badge: 'baseline' },
    { name: 'CatBoost V2',         target: 0.9070, color: AC.amber,     badge: 'gbt' },
    { name: 'XGBoost V2',          target: 0.9452, color: AC.tealDark,  badge: 'gbt' },
    { name: 'LightGBM V2 (full)',  target: 0.9575, color: AC.teal,      badge: 'charge required' },
    { name: 'Ensemble V2 (5-fold)',target: 0.9580, color: AC.greenLight,badge: '+0.0005 · not shipped' },
    { name: 'LightGBM V2 no-charge', target: 0.9428, color: AC.green,   badge: '★ production' },
  ];

  // Stagger each bar's animation
  const animateBar = (i) => {
    const start = 0.3 + i * 0.25;
    const end = start + 1.2;
    const p = Math.min(Math.max((t - start) / (end - start), 0), 1);
    return Easing.easeOutCubic(p);
  };

  // Flash winner at ~3.5s
  const winnerFlash = Math.max(0, Math.sin(Math.max(0, (t - 3.5)) * 8) * 0.5 + 0.5);

  return (
    <div style={{
      position: 'absolute', inset: 0, opacity: enter * (1 - exit),
    }}>
      <div style={{
        position: 'absolute', top: 100, left: 80,
        fontFamily: AF.mono, fontSize: 15, fontWeight: 700,
        letterSpacing: '0.22em', textTransform: 'uppercase',
        color: AC.teal,
      }}>The Model Zoo ▸ 04</div>

      <div style={{
        position: 'absolute', top: 150, left: 80, right: 80,
        fontFamily: AF.body, fontSize: 72, fontWeight: 800,
        letterSpacing: '-0.03em', lineHeight: 1.05, color: AC.ink,
      }}>
        Stage 1 · <span style={{ color: AC.teal }}>allowed amount</span>.
        <br/>No-charge LightGBM ships.
      </div>

      {/* Race track */}
      <div style={{
        position: 'absolute', top: 360, left: 80, right: 80,
        bottom: 80,
      }}>
        {/* Scale ticks */}
        <div style={{
          position: 'absolute', top: 0, left: 220, right: 20, height: 20,
          display: 'flex', justifyContent: 'space-between',
          fontFamily: AF.mono, fontSize: 11, letterSpacing: '0.16em',
          color: AC.ink3,
        }}>
          {[0.85, 0.88, 0.91, 0.94, 0.97].map(v => (
            <span key={v}>R² = {v.toFixed(2)}</span>
          ))}
        </div>

        {models.map((m, i) => {
          const p = animateBar(i);
          // Map [0.85, 0.97] to [0, 100%]
          const widthPct = Math.max(0, (m.target * p - 0.85) / 0.12) * 100;
          const isWinner = m.badge.startsWith('★');
          return (
            <div key={m.name} style={{
              position: 'absolute', top: 40 + i * 88, left: 0, right: 0,
              height: 70, display: 'flex', alignItems: 'center', gap: 16,
            }}>
              {/* Label */}
              <div style={{ width: 210, textAlign: 'right' }}>
                <div style={{
                  fontFamily: AF.body, fontSize: 22, fontWeight: 700,
                  color: isWinner ? AC.green : AC.ink,
                  letterSpacing: '-0.01em',
                }}>{m.name}</div>
                <div style={{
                  fontFamily: AF.mono, fontSize: 11, letterSpacing: '0.16em',
                  textTransform: 'uppercase',
                  color: isWinner ? AC.green : AC.ink3,
                  fontWeight: isWinner ? 700 : 500,
                }}>{m.badge}</div>
              </div>
              {/* Bar */}
              <div style={{
                flex: 1, height: 36, position: 'relative',
                background: AC.line, borderRadius: 2,
              }}>
                <div style={{
                  position: 'absolute', top: 0, left: 0, bottom: 0,
                  width: `${widthPct}%`,
                  background: m.color,
                  transition: 'width 40ms linear',
                  boxShadow: isWinner ? `0 0 ${20 * winnerFlash}px ${m.color}` : 'none',
                }}/>
                {/* Value at end */}
                {p > 0.1 && (
                  <div style={{
                    position: 'absolute', top: '50%', left: `${widthPct}%`,
                    transform: 'translate(8px, -50%)',
                    fontFamily: AF.mono, fontSize: 18, fontWeight: 700,
                    color: isWinner ? AC.green : AC.ink,
                    fontVariantNumeric: 'tabular-nums',
                  }}>
                    {(m.target * p).toFixed(4)}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

Object.assign(window, { ScenePipeline, PipelineBox, SceneModelZoo });
