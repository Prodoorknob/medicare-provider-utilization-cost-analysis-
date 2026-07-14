/* Scenes 1-3: Cold open, Problem, Data */

// Brand palette
const AC = {
  bg: '#FAFAF8', ink: '#1C1917', ink2: '#57534E', ink3: '#A8A29E',
  teal: '#0F6E8C', tealLight: '#1389AC', tealDark: '#0A4F66', tealTint: '#D0EEF5',
  green: '#15755D', greenLight: '#1CA082', greenTint: '#E8F7F3',
  amber: '#B8763A', amberLight: '#D4944D', amberTint: '#FDF4EA',
  red: '#DC2626', warn: '#D97706',
  line: 'rgba(0,0,0,0.08)', line2: 'rgba(0,0,0,0.18)',
};
const AF = { body: 'Inter, system-ui, sans-serif', mono: '"IBM Plex Mono", ui-monospace, monospace' };

// ── Persistent chrome (always on) ────────────────────────────────
function Chrome() {
  const t = useTime();
  return (
    <>
      {/* Top bar */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: 56,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 40px',
        fontFamily: AF.mono, fontSize: 13, letterSpacing: '0.14em',
        textTransform: 'uppercase', color: AC.ink3, zIndex: 50,
      }}>
        <span>AllowanceMap</span>
        <span style={{ display: 'flex', gap: 24 }}>
          <span>Medicare · CMS 2013–2023</span>
          <span style={{ color: AC.teal }}>● REC {(t).toFixed(2)}s</span>
        </span>
      </div>
      {/* Bottom bar */}
      <div style={{
        position: 'absolute', bottom: 0, left: 0, right: 0, height: 40,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 40px',
        fontFamily: AF.mono, fontSize: 11, letterSpacing: '0.18em',
        textTransform: 'uppercase', color: AC.ink3, zIndex: 50,
        borderTop: `1px solid ${AC.line}`,
      }}>
        <span>provider.utilization.cost</span>
        <span>two-stage · charge → allowance</span>
        <span>v2.4.1</span>
      </div>
      {/* Corner ticks */}
      {[
        { top: 60, left: 20 }, { top: 60, right: 20 },
        { bottom: 46, left: 20 }, { bottom: 46, right: 20 },
      ].map((pos, i) => (
        <div key={i} style={{
          position: 'absolute', ...pos, width: 14, height: 14,
          borderTop: `1px solid ${AC.ink3}`, borderLeft: `1px solid ${AC.ink3}`,
          transform: pos.right != null ? 'scaleX(-1)' : 'none',
          ...(pos.bottom != null ? { transform: pos.right != null ? 'scale(-1,-1)' : 'scaleY(-1)' } : {}),
        }}/>
      ))}
    </>
  );
}

// ── SCENE 1: Cold open — logo mark assembly ──────────────────────
function SceneOpen() {
  const { localTime: t } = useSprite();
  // Build the "AM" monogram: diamond + ring + label
  const ringScale = interpolate([0, 0.6, 1.0], [0, 1.1, 1], Easing.easeOutBack)(Math.min(t / 1.0, 1));
  const ringRot = t * 180; // spin subtly
  const diamondT = Math.min(Math.max((t - 0.4) / 0.5, 0), 1);
  const dScale = Easing.easeOutBack(diamondT);
  const labelT = Math.min(Math.max((t - 0.9) / 0.6, 0), 1);
  const labelY = (1 - Easing.easeOutCubic(labelT)) * 20;
  const taglineT = Math.min(Math.max((t - 1.5) / 0.7, 0), 1);
  const tlY = (1 - Easing.easeOutCubic(taglineT)) * 12;
  const fadeOut = Math.max(0, Math.min(1, (t - 2.4) / 0.6));

  return (
    <div style={{
      position: 'absolute', inset: 0, display: 'flex',
      flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      opacity: 1 - fadeOut,
    }}>
      {/* Ring + diamond */}
      <div style={{ position: 'relative', width: 220, height: 220, marginBottom: 40 }}>
        <svg viewBox="0 0 220 220" style={{
          position: 'absolute', inset: 0,
          transform: `scale(${ringScale}) rotate(${ringRot}deg)`,
        }}>
          <circle cx="110" cy="110" r="98" fill="none" stroke={AC.teal} strokeWidth="2"
            strokeDasharray="4 6" opacity="0.5"/>
          <circle cx="110" cy="110" r="82" fill="none" stroke={AC.teal} strokeWidth="1.5"/>
        </svg>
        {/* Pulse ring */}
        {[0, 0.5].map((off, i) => {
          const p = ((t + off) % 1.2) / 1.2;
          return (
            <div key={i} style={{
              position: 'absolute', inset: 0,
              border: `2px solid ${AC.teal}`, borderRadius: '50%',
              transform: `scale(${0.8 + p * 0.5})`,
              opacity: (1 - p) * 0.4,
            }}/>
          );
        })}
        {/* Diamond mark */}
        <div style={{
          position: 'absolute', inset: 0, display: 'flex',
          alignItems: 'center', justifyContent: 'center',
        }}>
          <div style={{
            width: 90, height: 90, background: AC.teal,
            transform: `rotate(45deg) scale(${dScale})`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 20px 40px rgba(15,110,140,0.35)',
          }}>
            <div style={{
              transform: 'rotate(-45deg)', color: '#fff',
              fontFamily: AF.body, fontSize: 42, fontWeight: 900,
              letterSpacing: '-0.04em',
            }}>AM</div>
          </div>
        </div>
      </div>
      {/* Wordmark */}
      <div style={{
        opacity: labelT, transform: `translateY(${labelY}px)`,
        fontFamily: AF.body, fontSize: 84, fontWeight: 800,
        letterSpacing: '-0.035em', color: AC.ink,
      }}>
        AllowanceMap
      </div>
      {/* Tagline */}
      <div style={{
        opacity: taglineT, transform: `translateY(${tlY}px)`,
        fontFamily: AF.mono, fontSize: 18, fontWeight: 500,
        letterSpacing: '0.22em', textTransform: 'uppercase',
        color: AC.ink2, marginTop: 20,
      }}>
        Medicare · Provider · Cost · Intelligence
      </div>
    </div>
  );
}

// ── SCENE 2: The Problem — numbers + swarm ───────────────────────
function SceneProblem() {
  const { localTime: t, duration } = useSprite();

  // Counter tickers
  const spendProg = Math.min(t / 2.0, 1);
  const spend = Easing.easeOutCubic(spendProg) * 944; // $944B Medicare spend
  const providerProg = Math.min(Math.max((t - 0.6) / 2.0, 0), 1);
  const providers = Math.floor(Easing.easeOutCubic(providerProg) * 1.26e6);
  const procProg = Math.min(Math.max((t - 1.2) / 1.8, 0), 1);
  const procs = Math.floor(Easing.easeOutCubic(procProg) * 9500);

  // Exit fade
  const fadeOut = Math.max(0, Math.min(1, (t - (duration - 0.7)) / 0.7));
  const enter = Math.min(Easing.easeOutCubic(Math.min(t / 0.6, 1)), 1);

  // Dot swarm (providers visualised)
  const dots = [];
  const count = 180;
  for (let i = 0; i < count; i++) {
    const seed = i * 7.31;
    const vis = Math.max(0, Math.min(1, (t - 1.0 - (i / count) * 1.8) / 0.4));
    if (vis <= 0) continue;
    const angle = seed;
    const radius = 80 + ((i * 17) % 260);
    const cx = 1460 + Math.cos(angle) * radius + Math.sin(t * 0.4 + i) * 6;
    const cy = 480 + Math.sin(angle) * radius * 0.6 + Math.cos(t * 0.5 + i) * 4;
    const hue = (i * 41) % 3;
    const color = hue === 0 ? AC.teal : hue === 1 ? AC.green : AC.amber;
    dots.push({ cx, cy, color, vis, r: 3 + ((i * 13) % 5) });
  }

  return (
    <div style={{
      position: 'absolute', inset: 0, opacity: (1 - fadeOut) * enter,
    }}>
      {/* Eyebrow */}
      <div style={{
        position: 'absolute', top: 100, left: 80,
        fontFamily: AF.mono, fontSize: 15, fontWeight: 700,
        letterSpacing: '0.22em', textTransform: 'uppercase',
        color: AC.amber,
      }}>The Problem ▸ 01</div>

      {/* Headline */}
      <div style={{
        position: 'absolute', top: 150, left: 80, right: 700,
        fontFamily: AF.body, fontSize: 72, fontWeight: 800,
        letterSpacing: '-0.03em', lineHeight: 1.05, color: AC.ink,
      }}>
        Medicare pricing is<br/>
        <span style={{ color: AC.red }}>opaque</span>, fragmented,<br/>
        and <span style={{ color: AC.amber }}>massive</span>.
      </div>

      {/* Big stats */}
      <div style={{
        position: 'absolute', top: 490, left: 80, right: 800,
        display: 'flex', gap: 40, flexWrap: 'wrap',
      }}>
        <Stat label="Annual Spend" value={`$${spend.toFixed(0)}B`} color={AC.teal} />
        <Stat label="Providers" value={providers.toLocaleString('en-US')} color={AC.green} />
        <Stat label="HCPCS Codes" value={procs.toLocaleString('en-US')} color={AC.amber} />
      </div>

      {/* Swarm */}
      <svg viewBox="0 0 1920 1080" style={{
        position: 'absolute', inset: 0, width: '100%', height: '100%',
        pointerEvents: 'none',
      }}>
        {/* Guide circle */}
        <circle cx="1460" cy="480" r="340" fill="none"
          stroke={AC.line2} strokeDasharray="2 8" opacity="0.6"/>
        {dots.map((d, i) => (
          <circle key={i} cx={d.cx} cy={d.cy} r={d.r * d.vis}
            fill={d.color} opacity={d.vis * 0.8}/>
        ))}
        {/* Caption */}
        <text x="1460" y="860" textAnchor="middle"
          fontFamily={AF.mono} fontSize="14" fill={AC.ink3}
          letterSpacing="3">1.26M PROVIDERS · 50 STATES</text>
      </svg>
    </div>
  );
}

function Stat({ label, value, color }) {
  return (
    <div style={{ minWidth: 260 }}>
      <div style={{
        fontFamily: AF.mono, fontSize: 13, letterSpacing: '0.18em',
        textTransform: 'uppercase', color: AC.ink3, marginBottom: 8,
      }}>{label}</div>
      <div style={{
        fontFamily: AF.body, fontSize: 64, fontWeight: 800,
        letterSpacing: '-0.02em', color, lineHeight: 1,
        fontVariantNumeric: 'tabular-nums',
      }}>{value}</div>
    </div>
  );
}

// ── SCENE 3: The Data — streaming rows ───────────────────────────
function SceneData() {
  const { localTime: t, duration } = useSprite();
  const enter = Easing.easeOutCubic(Math.min(t / 0.5, 1));
  const exit = Math.max(0, Math.min(1, (t - (duration - 0.6)) / 0.6));

  // Counters
  const rowsProg = Math.min(t / 3.5, 1);
  const rows = Math.floor(Easing.easeOutCubic(rowsProg) * 126_800_000);
  const yearsProg = Math.min(t / 2, 1);
  const years = Math.floor(Easing.easeOutCubic(yearsProg) * 11) + 2013;

  // Streaming rows in table
  const tableRows = [];
  const ROW_H = 36;
  const TABLE_TOP = 480;
  const speed = 120; // px/s
  const startOffset = t * speed;
  for (let i = 0; i < 16; i++) {
    const y = TABLE_TOP + i * ROW_H - (startOffset % ROW_H);
    const idx = Math.floor(startOffset / ROW_H) + i;
    tableRows.push({ y, idx });
  }

  const SAMPLE = [
    ['99213', 'Office visit, est.', '$109.42', '$76.84', 'NY'],
    ['93000', 'Electrocardiogram', '$55.19', '$18.22', 'CA'],
    ['G0439', 'Annual wellness', '$171.03', '$119.82', 'TX'],
    ['70450', 'CT head, w/o', '$284.67', '$180.11', 'FL'],
    ['36415', 'Venipuncture', '$9.80', '$3.11', 'OH'],
    ['88305', 'Tissue exam, path', '$135.48', '$73.02', 'IL'],
    ['85025', 'CBC w/auto diff', '$22.10', '$10.84', 'PA'],
    ['J0129', 'Abatacept inj', '$1,206.40', '$903.17', 'GA'],
  ];

  return (
    <div style={{
      position: 'absolute', inset: 0, opacity: enter * (1 - exit),
    }}>
      <div style={{
        position: 'absolute', top: 100, left: 80,
        fontFamily: AF.mono, fontSize: 15, fontWeight: 700,
        letterSpacing: '0.22em', textTransform: 'uppercase',
        color: AC.teal,
      }}>The Data ▸ 02</div>

      <div style={{
        position: 'absolute', top: 150, left: 80, right: 80,
        fontFamily: AF.body, fontSize: 72, fontWeight: 800,
        letterSpacing: '-0.03em', lineHeight: 1.05, color: AC.ink,
      }}>
        11 years. <span style={{ color: AC.teal }}>{years}</span>. Every
        <br/>Part B claim, every provider.
      </div>

      {/* Big counters */}
      <div style={{
        position: 'absolute', top: 320, left: 80, right: 80,
        display: 'flex', gap: 48, alignItems: 'flex-start',
      }}>
        <div style={{ flex: '0 0 auto' }}>
          <div style={{
            fontFamily: AF.body, fontSize: 64, fontWeight: 800,
            color: AC.teal, letterSpacing: '-0.02em', lineHeight: 1,
            fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap',
          }}>{rows.toLocaleString('en-US')}</div>
          <div style={{
            fontFamily: AF.mono, fontSize: 13, letterSpacing: '0.18em',
            textTransform: 'uppercase', color: AC.ink3, marginTop: 6,
          }}>rows ingested</div>
        </div>
        <div style={{ width: 1, background: AC.line, alignSelf: 'stretch' }}/>
        <div style={{ flex: '0 0 auto' }}>
          <div style={{
            fontFamily: AF.body, fontSize: 64, fontWeight: 800,
            color: AC.green, letterSpacing: '-0.02em', lineHeight: 1,
            whiteSpace: 'nowrap',
          }}>42 GB</div>
          <div style={{
            fontFamily: AF.mono, fontSize: 13, letterSpacing: '0.18em',
            textTransform: 'uppercase', color: AC.ink3, marginTop: 6,
          }}>parquet · zstd</div>
        </div>
        <div style={{ width: 1, background: AC.line, alignSelf: 'stretch' }}/>
        <div style={{ flex: '0 0 auto' }}>
          <div style={{
            fontFamily: AF.body, fontSize: 64, fontWeight: 800,
            color: AC.amber, letterSpacing: '-0.02em', lineHeight: 1,
            whiteSpace: 'nowrap',
          }}>CMS</div>
          <div style={{
            fontFamily: AF.mono, fontSize: 13, letterSpacing: '0.18em',
            textTransform: 'uppercase', color: AC.ink3, marginTop: 6,
          }}>primary source</div>
        </div>
      </div>

      {/* Streaming table */}
      <div style={{
        position: 'absolute', top: TABLE_TOP - 40, left: 80, right: 80,
        bottom: 80, borderTop: `1px solid ${AC.line}`,
        borderBottom: `1px solid ${AC.line}`,
        overflow: 'hidden',
      }}>
        {/* Header */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '140px 1fr 160px 160px 80px',
          gap: 16, padding: '10px 20px',
          fontFamily: AF.mono, fontSize: 12, fontWeight: 700,
          letterSpacing: '0.16em', textTransform: 'uppercase',
          color: AC.ink3, borderBottom: `1px solid ${AC.line}`,
          background: AC.bg, position: 'sticky', top: 0,
        }}>
          <span>hcpcs</span><span>description</span>
          <span style={{ textAlign: 'right' }}>submitted</span>
          <span style={{ textAlign: 'right' }}>allowed</span>
          <span>state</span>
        </div>

        {tableRows.map(({ y, idx }, i) => {
          const row = SAMPLE[idx % SAMPLE.length];
          const offset = y - TABLE_TOP;
          const alpha = 1 - Math.abs(offset - 200) / 500;
          return (
            <div key={i} style={{
              position: 'absolute', top: offset, left: 0, right: 0,
              display: 'grid',
              gridTemplateColumns: '140px 1fr 160px 160px 80px',
              gap: 16, padding: '10px 20px',
              fontFamily: AF.mono, fontSize: 15,
              color: AC.ink2,
              opacity: Math.max(0, Math.min(1, alpha)),
              borderBottom: `1px solid ${AC.line}`,
              fontVariantNumeric: 'tabular-nums',
            }}>
              <span style={{ color: AC.teal, fontWeight: 600 }}>{row[0]}</span>
              <span>{row[1]}</span>
              <span style={{ textAlign: 'right', color: AC.ink }}>{row[2]}</span>
              <span style={{ textAlign: 'right', color: AC.green, fontWeight: 600 }}>{row[3]}</span>
              <span>{row[4]}</span>
            </div>
          );
        })}

        {/* Fade gradients */}
        <div style={{
          position: 'absolute', top: 44, left: 0, right: 0, height: 80,
          background: `linear-gradient(to bottom, ${AC.bg}, transparent)`,
          pointerEvents: 'none',
        }}/>
        <div style={{
          position: 'absolute', bottom: 0, left: 0, right: 0, height: 100,
          background: `linear-gradient(to top, ${AC.bg}, transparent)`,
          pointerEvents: 'none',
        }}/>
      </div>
    </div>
  );
}

Object.assign(window, { AC, AF, Chrome, SceneOpen, SceneProblem, SceneData, Stat });
