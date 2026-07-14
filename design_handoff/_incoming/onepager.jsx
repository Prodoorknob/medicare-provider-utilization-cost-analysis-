/* Fraud Detection Agent - Compact One-Pager
   Warm light bg, teal + red accents, matches the demo system. */

const AC = {
  bg: '#FAFAF8', surface: '#FFFFFF', surface2: '#F2F0ED',
  ink: '#1C1917', ink2: '#57534E', ink3: '#A8A29E',
  teal: '#0F6E8C', tealLight: '#1389AC', tealDark: '#0A4F66', tealTint: '#EBF7FB',
  green: '#15755D', greenLight: '#1CA082', greenTint: '#E8F7F3',
  amber: '#B8763A', amberTint: '#FDF4EA',
  red: '#DC2626', redTint: '#FEF2F2',
  orange: '#EA580C',
  line: 'rgba(0,0,0,0.08)', line2: 'rgba(0,0,0,0.15)',
};
const AF = {
  body: 'Inter, system-ui, sans-serif',
  mono: '"IBM Plex Mono", ui-monospace, monospace',
};

// ── Bits ─────────────────────────────────────────────────────────────
function Eyebrow({ children, color = AC.teal }) {
  return (
    <div style={{
      fontFamily: AF.mono, fontSize: 11, fontWeight: 700,
      letterSpacing: '0.22em', textTransform: 'uppercase', color,
    }}>{children}</div>
  );
}

function Dot({ color, glow }) {
  return (
    <span style={{
      display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
      background: color, boxShadow: glow ? `0 0 10px ${color}` : 'none',
    }}/>
  );
}

// ── Share card (inline w/ hero) ──────────────────────────────────────
function ShareCard() {
  const rows = 8, cols = 26;
  const dots = [];
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      dots.push({ r, c, flagged: r === 4 && c === 10 });

  return (
    <div style={{
      background: '#0D0C0B', borderRadius: 14, overflow: 'hidden',
      position: 'relative', width: '100%',
      padding: '24px 28px',
      display: 'flex', flexDirection: 'column', gap: 18,
      boxShadow: '0 20px 50px rgba(0,0,0,0.15), 0 0 0 1px rgba(0,0,0,0.08)',
    }}>
      <div style={{
        position: 'absolute', inset: 0, pointerEvents: 'none',
        background: 'radial-gradient(ellipse at 30% 30%, rgba(220,38,38,0.18), transparent 55%), radial-gradient(ellipse at 78% 78%, rgba(19,137,172,0.18), transparent 60%)',
      }}/>

      {/* Eyebrow row */}
      <div style={{
        position: 'relative',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 16,
      }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          fontFamily: AF.mono, fontSize: 10, fontWeight: 700,
          letterSpacing: '0.2em', textTransform: 'uppercase', color: AC.red,
        }}>
          <Dot color={AC.red} glow/>
          FRAUD DETECTION AGENT
        </div>
        <div style={{
          fontFamily: AF.mono, fontSize: 10, color: AC.ink3,
          letterSpacing: '0.18em', textTransform: 'uppercase',
        }}>CMS 2013–2023</div>
      </div>

      {/* Headline */}
      <div style={{
        position: 'relative',
        fontFamily: AF.body, fontSize: 30, fontWeight: 800,
        color: '#fff', letterSpacing: '-0.025em', lineHeight: 1.08,
        textWrap: 'balance',
      }}>
        An AI agent reading Medicare like a{' '}
        <span style={{ color: AC.tealLight }}>forensic accountant</span>.
      </div>

      {/* Provider grid */}
      <svg viewBox="0 0 520 96" preserveAspectRatio="none" style={{
        position: 'relative', width: '100%', height: 96, display: 'block',
      }}>
        {dots.map((d, i) => {
          const x = 10 + d.c * 20;
          const y = 8 + d.r * 11;
          let color = 'rgba(255,255,255,0.18)';
          if (d.flagged) color = AC.red;
          else if ((i * 13) % 41 === 0) color = AC.orange;
          return <circle key={i} cx={x} cy={y} r={d.flagged ? 4 : 2} fill={color}/>;
        })}
        <circle cx={10 + 10 * 20} cy={8 + 4 * 11} r="12"
                fill="none" stroke={AC.red} strokeWidth="1.5" opacity="0.55"/>
      </svg>

      {/* Stat row */}
      <div style={{
        position: 'relative',
        display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12,
        fontFamily: AF.mono, color: '#fff',
      }}>
        {[
          ['scanned', '1.26M', '#fff'],
          ['flagged', '1', AC.red],
          ['time', '< 8s', '#fff'],
          ['powered by', 'Claude', AC.tealLight],
        ].map(([label, val, c]) => (
          <div key={label}>
            <div style={{ fontSize: 9, color: AC.ink3, letterSpacing: '0.18em', textTransform: 'uppercase' }}>{label}</div>
            <div style={{ fontSize: 20, fontWeight: 700, color: c, marginTop: 2 }}>{val}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Hero ─────────────────────────────────────────────────────────────
function Hero() {
  return (
    <section style={{
      padding: '88px 72px 56px',
      borderBottom: `1px solid ${AC.line}`,
    }}>
      <div style={{
        display: 'grid', gridTemplateColumns: '1.3fr 1fr', gap: 56,
        alignItems: 'center',
      }}>
        <div>
          <Eyebrow color={AC.red}>
            <Dot color={AC.red} glow/> <span style={{ marginLeft: 8 }}>Product teardown · 4 min read</span>
          </Eyebrow>
          <h1 style={{
            fontFamily: AF.body, fontSize: 68, fontWeight: 800,
            letterSpacing: '-0.035em', lineHeight: 1.02,
            margin: '16px 0 20px', color: AC.ink,
          }}>
            How an AI agent reads Medicare like a <span style={{ color: AC.teal }}>forensic accountant</span>.
          </h1>
          <p style={{
            fontFamily: AF.body, fontSize: 20, lineHeight: 1.5,
            color: AC.ink2, margin: 0, maxWidth: 640,
            textWrap: 'pretty',
          }}>
            A look at the Fraud Detection Agent inside AllowanceMap - a Claude-powered
            reviewer that scans <b>1.26M providers</b> across a decade of Medicare claims,
            flags statistical outliers, and hands analysts an evidence-backed brief in seconds.
          </p>

          {/* Meta row */}
          <div style={{
            marginTop: 32, display: 'flex', gap: 28, alignItems: 'center',
            fontFamily: AF.mono, fontSize: 12, color: AC.ink2,
            letterSpacing: '0.08em',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div style={{
                width: 28, height: 28, borderRadius: '50%',
                background: AC.teal, color: '#fff',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontFamily: AF.body, fontWeight: 700, fontSize: 13,
              }}>R</div>
              <span>@rvedire.com</span>
            </div>
            <span style={{ color: AC.ink3 }}>·</span>
            <span>April 2026</span>
            <span style={{ color: AC.ink3 }}>·</span>
            <a href="#" style={{
              color: AC.teal, textDecoration: 'none', fontWeight: 600,
              borderBottom: `1px solid ${AC.teal}`, paddingBottom: 1,
            }}>See the live demo →</a>
          </div>
        </div>

        <ShareCard/>
      </div>
    </section>
  );
}

// ── Medicare primer (background context for Part B) ─────────────────
function MedicarePrimer() {
  const parts = [
    { id: 'A', desc: 'Hospital stays, skilled nursing', focus: false },
    { id: 'B', desc: 'Doctor visits, outpatient care, labs', focus: true },
    { id: 'C', desc: 'Medicare Advantage (private bundles)', focus: false },
    { id: 'D', desc: 'Prescription drugs', focus: false },
  ];
  const stats = [
    { big: '~63M', label: 'Medicare beneficiaries (2023)', sub: 'Roughly 1 in 5 Americans', accent: AC.tealLight },
    { big: '80 / 20', label: 'Part B cost split', sub: 'Medicare pays 80% of the "allowed amount", patient pays 20%', accent: AC.teal },
    { big: '$900B+', label: 'Annual Medicare spending', sub: 'Largest single U.S. health-insurance program', accent: AC.tealDark },
    { big: '10,000/day', label: 'Americans aging into Medicare', sub: 'Caseload grows every day', accent: AC.amber },
  ];
  return (
    <section style={{
      padding: '64px 72px',
      borderBottom: `1px solid ${AC.line}`,
    }}>
      <Eyebrow>§ 00 · Primer · what is Medicare?</Eyebrow>
      <h2 style={{
        fontFamily: AF.body, fontSize: 40, fontWeight: 700,
        letterSpacing: '-0.025em', lineHeight: 1.1, maxWidth: 1000,
        margin: '14px 0 12px', color: AC.ink, textWrap: 'balance',
      }}>
        Federal health insurance for <span style={{ color: AC.teal }}>63 million Americans</span>.
      </h2>
      <p style={{
        fontFamily: AF.body, fontSize: 17, lineHeight: 1.55,
        color: AC.ink2, maxWidth: 880, margin: '0 0 36px',
      }}>
        Quick context for the rest of this teardown - Medicare is the U.S. federal
        health-insurance program, enacted in <b>1965</b>. It covers Americans <b>65
        and older</b>, plus some younger people with long-term disabilities or
        end-stage kidney failure. We focus on <b>Part B</b>, the outpatient slice.
      </p>

      <div style={{
        display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 32,
      }}>
        {/* Left: four parts */}
        <div>
          <div style={{
            fontFamily: AF.mono, fontSize: 11, fontWeight: 700,
            letterSpacing: '0.22em', textTransform: 'uppercase',
            color: AC.teal, marginBottom: 14,
          }}>The four parts</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {parts.map((p) => {
              const isFocus = p.focus;
              return (
                <div key={p.id} style={{
                  display: 'flex', alignItems: 'center', gap: 14,
                  background: isFocus ? '#FFF7F0' : AC.surface,
                  border: `1px solid ${isFocus ? 'rgba(234,88,12,0.35)' : AC.line}`,
                  borderLeft: `4px solid ${isFocus ? AC.orange : AC.line2}`,
                  borderRadius: 8, padding: '14px 18px',
                }}>
                  <div style={{
                    width: 76, height: 38, flexShrink: 0,
                    background: isFocus ? AC.orange : AC.surface2,
                    color: isFocus ? '#fff' : AC.ink,
                    border: `1px solid ${isFocus ? AC.orange : AC.line}`,
                    borderRadius: 5,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontFamily: AF.body, fontSize: 14, fontWeight: 700,
                  }}>Part {p.id}</div>
                  <div style={{
                    fontFamily: AF.body, fontSize: 15,
                    color: isFocus ? AC.ink : AC.ink2,
                    fontWeight: isFocus ? 600 : 500,
                    lineHeight: 1.3,
                  }}>{p.desc}</div>
                </div>
              );
            })}
          </div>
          <div style={{
            marginTop: 14, marginLeft: 4,
            display: 'flex', alignItems: 'center', gap: 8,
            fontFamily: AF.mono, fontSize: 11, fontWeight: 600,
            letterSpacing: '0.18em', textTransform: 'uppercase',
            color: AC.orange,
          }}>
            <span>↑ Part B is what AllowanceMap audits</span>
          </div>
        </div>

        {/* Right: stat cards */}
        <div style={{
          display: 'flex', flexDirection: 'column', gap: 10,
        }}>
          {stats.map((s) => (
            <div key={s.label} style={{
              background: AC.surface,
              border: `1px solid ${AC.line}`,
              borderLeft: `4px solid ${s.accent}`,
              borderRadius: 8,
              padding: '14px 18px',
            }}>
              <div style={{
                fontFamily: AF.body, fontSize: 32, fontWeight: 800,
                color: AC.ink, letterSpacing: '-0.03em', lineHeight: 1,
                fontVariantNumeric: 'tabular-nums',
              }}>{s.big}</div>
              <div style={{
                fontFamily: AF.body, fontSize: 14, fontWeight: 600,
                color: AC.ink, marginTop: 6, letterSpacing: '-0.005em',
              }}>{s.label}</div>
              <div style={{
                fontFamily: AF.body, fontSize: 12, fontWeight: 400,
                color: AC.ink2, marginTop: 1, lineHeight: 1.35,
              }}>{s.sub}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ── "The problem" stat band ──────────────────────────────────────────
function ProblemBand() {
  const stats = [
    { big: '$60B+', label: 'Estimated annual Medicare\nimproper payments', sub: 'CMS, conservative estimates' },
    { big: '1.26M', label: 'Providers billing Part B\nin any given year', sub: 'across 10,000+ procedure codes' },
    { big: '~3%', label: 'Of total program spend,\nimproper on the low end', sub: '- multiples higher in some pockets' },
  ];
  return (
    <section style={{
      padding: '64px 72px', background: AC.surface2,
      borderBottom: `1px solid ${AC.line}`,
    }}>
      <Eyebrow>§ 01 · The problem</Eyebrow>
      <h2 style={{
        fontFamily: AF.body, fontSize: 40, fontWeight: 700,
        letterSpacing: '-0.025em', lineHeight: 1.1, maxWidth: 900,
        margin: '14px 0 44px', color: AC.ink, textWrap: 'balance',
      }}>
        The signal is in the data. The problem is scale -
        a human team can't read every row.
      </h2>
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 32,
      }}>
        {stats.map((s, i) => (
          <div key={i} style={{
            padding: '28px 0', borderTop: `2px solid ${AC.ink}`,
          }}>
            <div style={{
              fontFamily: AF.body, fontSize: 64, fontWeight: 800,
              letterSpacing: '-0.04em', lineHeight: 1, color: AC.ink,
            }}>{s.big}</div>
            <div style={{
              marginTop: 18, fontFamily: AF.body, fontSize: 17,
              color: AC.ink, lineHeight: 1.35, whiteSpace: 'pre-line',
              fontWeight: 500,
            }}>{s.label}</div>
            <div style={{
              marginTop: 10, fontFamily: AF.mono, fontSize: 11,
              color: AC.ink3, letterSpacing: '0.04em',
            }}>{s.sub}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

// ── How it works - 6 rules grid ──────────────────────────────────────
function HowItWorks() {
  const rules = [
    {
      id: 'VOLUME_SPIKE', human: 'A provider who went from zero to firehose.',
      math: 'YoY growth vs. prior-3yr baseline, z-scored within specialty.',
      example: '+9,655% YoY on 24,874 services',
      color: AC.red, status: 'high-signal',
    },
    {
      id: 'CHARGE_INFLATION', human: "Billing way more than Medicare will pay.",
      math: 'ratio of submitted charge ÷ Medicare allowed amount, percentile-ranked.',
      example: '25.51× charge/allowed - P100',
      color: AC.red, status: 'high-signal',
    },
    {
      id: 'HIGH_INTENSITY', human: 'Same handful of patients, suspiciously many services.',
      math: 'services per unique beneficiary, specialty-normalized.',
      example: '2.71 srvcs/bene - P99',
      color: AC.red, status: 'high-signal',
    },
    {
      id: 'OUT_OF_SPECIALTY', human: "Doing procedures outside what their specialty does.",
      math: '% of services with HCPCS codes outside specialty mode set.',
      example: '10.2% - below 20% threshold',
      color: AC.green, status: 'negative control',
    },
    {
      id: 'PROCEDURE_CONCENTRATION', human: 'All their revenue from one niche code.',
      math: 'Herfindahl–Hirschman index over HCPCS mix.',
      example: 'HHI 0.255 - P12',
      color: AC.green, status: 'negative control',
    },
    {
      id: 'IMPOSSIBLE_DAY', human: 'More services in one day than hours in a day.',
      math: 'services-per-day ceiling vs. typical operating hours.',
      example: 'no date-of-service field in public CMS',
      color: AC.ink3, status: 'not evaluable',
    },
  ];

  return (
    <section style={{ padding: '72px 72px' }}>
      <Eyebrow>§ 02 · How it works</Eyebrow>
      <h2 style={{
        fontFamily: AF.body, fontSize: 40, fontWeight: 700,
        letterSpacing: '-0.025em', lineHeight: 1.1, maxWidth: 900,
        margin: '14px 0 12px', color: AC.ink, textWrap: 'balance',
      }}>
        Six rules. Each a question a human investigator would ask.
      </h2>
      <p style={{
        fontFamily: AF.body, fontSize: 17, lineHeight: 1.55,
        color: AC.ink2, maxWidth: 720, margin: '0 0 40px',
      }}>
        The agent runs every provider against a transparent panel of checks,
        then asks Claude to read the evidence and write a prose finding.
        Rules that <i>don't</i> trigger matter as much as the ones that do -
        they rule out boring explanations.
      </p>

      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 18,
      }}>
        {rules.map((r) => (
          <div key={r.id} style={{
            background: AC.surface, borderRadius: 10, padding: '22px 22px 20px',
            border: `1px solid ${AC.line}`,
            borderLeft: `4px solid ${r.color}`,
          }}>
            <div style={{
              display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
              marginBottom: 14,
            }}>
              <span style={{
                fontFamily: AF.mono, fontSize: 11, fontWeight: 700,
                letterSpacing: '0.12em', color: AC.ink,
              }}>{r.id}</span>
              <span style={{
                fontFamily: AF.mono, fontSize: 9, fontWeight: 700,
                letterSpacing: '0.14em', color: r.color, textTransform: 'uppercase',
              }}>{r.status}</span>
            </div>
            <div style={{
              fontFamily: AF.body, fontSize: 17, lineHeight: 1.35, color: AC.ink,
              fontWeight: 600, marginBottom: 14,
            }}>
              "{r.human}"
            </div>
            <div style={{
              paddingTop: 14, borderTop: `1px dashed ${AC.line2}`,
              fontFamily: AF.mono, fontSize: 11, lineHeight: 1.5,
              color: AC.ink2,
            }}>
              <div style={{ color: AC.ink3, marginBottom: 6, letterSpacing: '0.08em' }}>MATH</div>
              {r.math}
              <div style={{
                marginTop: 12, padding: '8px 10px', background: AC.bg,
                borderRadius: 4, color: AC.ink,
              }}>{r.example}</div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

// ── Investigation brief screenshot ───────────────────────────────────
function BriefScreenshot() {
  const rules = [
    { id: 'VOLUME_SPIKE', status: 'TRIGGERED', desc: '+9,655% YoY on 24,874 services' },
    { id: 'CHARGE_INFLATION', status: 'TRIGGERED', desc: 'charge/allowed ratio 25.51 · P100' },
    { id: 'HIGH_INTENSITY', status: 'TRIGGERED', desc: 'srvcs/bene 2.71 at P99' },
    { id: 'OUT_OF_SPECIALTY', status: 'NOT_TRIGGERED', desc: '10.2% - below 20% threshold' },
    { id: 'PROCEDURE_CONCENTRATION', status: 'NOT_TRIGGERED', desc: 'Herfindahl 0.255 · P12' },
    { id: 'IMPOSSIBLE_DAY', status: 'NOT_EVALUABLE', desc: 'no date-of-service field' },
  ];

  return (
    <section style={{
      padding: '72px 72px', background: '#0D0C0B', color: '#fff',
      position: 'relative', overflow: 'hidden',
    }}>
      <div style={{
        position: 'absolute', inset: 0,
        background: 'radial-gradient(ellipse at 20% 20%, rgba(220,38,38,0.10), transparent 55%), radial-gradient(ellipse at 80% 80%, rgba(19,137,172,0.12), transparent 60%)',
        pointerEvents: 'none',
      }}/>
      <div style={{ position: 'relative' }}>
        <Eyebrow color={AC.red}>§ 03 · The deliverable</Eyebrow>
        <h2 style={{
          fontFamily: AF.body, fontSize: 40, fontWeight: 700,
          letterSpacing: '-0.025em', lineHeight: 1.1, maxWidth: 900,
          margin: '14px 0 12px', color: '#fff', textWrap: 'balance',
        }}>
          An investigation brief, not a ranked list.
        </h2>
        <p style={{
          fontFamily: AF.body, fontSize: 17, lineHeight: 1.55,
          color: '#A8A29E', maxWidth: 720, margin: '0 0 40px',
        }}>
          When the agent escalates, an analyst gets this - a single-pane
          document with the finding, the math behind each rule, and three
          possible actions. Identifying details are hashed client-side.
        </p>

        <div style={{
          background: AC.surface, borderRadius: 14,
          boxShadow: '0 30px 80px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.08)',
          overflow: 'hidden',
          borderLeft: `5px solid ${AC.red}`,
          color: AC.ink,
          maxWidth: 980,
        }}>
          <div style={{
            padding: '22px 28px', borderBottom: `1px solid ${AC.line}`,
            display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
          }}>
            <div>
              <div style={{
                fontFamily: AF.mono, fontSize: 10, fontWeight: 700,
                letterSpacing: '0.22em', textTransform: 'uppercase', color: AC.red,
              }}>INVESTIGATION BRIEF · 2023</div>
              <div style={{
                fontFamily: AF.body, fontSize: 22, fontWeight: 700,
                color: AC.ink, marginTop: 4,
              }}>
                Provider <span style={{ fontFamily: AF.mono, color: AC.teal }}>#a4f··2c1</span>
              </div>
              <div style={{
                fontFamily: AF.body, fontSize: 14, color: AC.ink2, marginTop: 2,
              }}>Emergency Medicine · Northeast region</div>
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
            Catastrophic volume spike on supply codes with a <b>25× charge-to-allowed
            ratio</b> - structurally incompatible with hospital-based emergency medicine
            practice. Prior-year dormancy rules out organic growth.
          </div>

          <div style={{ padding: '16px 28px' }}>
            <div style={{
              fontFamily: AF.mono, fontSize: 10, fontWeight: 700,
              letterSpacing: '0.22em', textTransform: 'uppercase',
              color: AC.ink3, marginBottom: 10,
            }}>RULE CHECKS</div>
            <div style={{
              display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8,
            }}>
              {rules.map((r) => {
                const c = {
                  TRIGGERED: AC.red,
                  NOT_TRIGGERED: AC.green,
                  NOT_EVALUABLE: AC.ink3,
                }[r.status];
                return (
                  <div key={r.id} style={{
                    display: 'flex', alignItems: 'flex-start', gap: 8,
                    padding: '10px 12px', background: AC.bg,
                    borderLeft: `3px solid ${c}`, borderRadius: 4,
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
                      letterSpacing: '0.12em', color: c, whiteSpace: 'nowrap',
                    }}>{r.status.replace('_', ' ')}</div>
                  </div>
                );
              })}
            </div>
          </div>

          <div style={{
            padding: '14px 28px', borderTop: `1px solid ${AC.line}`,
            display: 'flex', gap: 10, alignItems: 'center',
          }}>
            <span style={{
              fontFamily: AF.mono, fontSize: 10, fontWeight: 700,
              letterSpacing: '0.2em', textTransform: 'uppercase',
              color: AC.ink3, marginRight: 'auto',
            }}>ANALYST ACTION</span>
            {[
              ['Approve', AC.green, false],
              ['Escalate', AC.red, true],
              ['Dismiss', AC.ink3, false],
            ].map(([label, c, active]) => (
              <div key={label} style={{
                padding: '8px 16px', borderRadius: 6,
                border: `1.5px solid ${c}`,
                color: active ? '#fff' : c,
                background: active ? c : 'transparent',
                fontFamily: AF.body, fontSize: 13, fontWeight: 600,
              }}>{label}</div>
            ))}
          </div>
        </div>

        <div style={{
          marginTop: 16, fontFamily: AF.mono, fontSize: 11,
          letterSpacing: '0.18em', textTransform: 'uppercase', color: '#A8A29E',
        }}>
          Claude-generated · 6 rules · 1 analyst tap to escalate
        </div>
      </div>
    </section>
  );
}

// ── Why Claude + why an agent ────────────────────────────────────────
function WhyClaude() {
  const rows = [
    {
      legacy: 'Black-box classifier',
      legacyDesc: 'Gives you a score. Maybe feature importances. An analyst still has to reconstruct the "why" manually.',
      agent: 'A reviewer that writes its reasoning',
      agentDesc: 'Claude reads the rule outputs + the provider profile and produces a prose finding an analyst can defend in a meeting.',
    },
    {
      legacy: 'Retrain to change behavior',
      legacyDesc: 'Adjusting a threshold means a new training run, a new validation set, and a new model artifact to deploy.',
      agent: 'Edit the rules, ship in minutes',
      agentDesc: 'Rules are code. Thresholds are config. The LLM adapts its narrative to whatever the rule panel decides is important.',
    },
    {
      legacy: 'Ranked list of suspects',
      legacyDesc: 'Analysts spend the day deciding which 10 rows to look at first. The top of the list is never the most interesting.',
      agent: 'Triaged brief with context',
      agentDesc: 'Every escalation comes with the evidence, the statistical rank, and the negative controls - so analysts spend time acting, not excavating.',
    },
  ];

  return (
    <section style={{
      padding: '72px 72px', background: AC.surface2,
      borderTop: `1px solid ${AC.line}`, borderBottom: `1px solid ${AC.line}`,
    }}>
      <Eyebrow>§ 04 · Why Claude · why an agent</Eyebrow>
      <h2 style={{
        fontFamily: AF.body, fontSize: 40, fontWeight: 700,
        letterSpacing: '-0.025em', lineHeight: 1.1, maxWidth: 900,
        margin: '14px 0 44px', color: AC.ink, textWrap: 'balance',
      }}>
        We didn't want another classifier. We wanted a reviewer.
      </h2>

      <div style={{
        display: 'grid', gridTemplateColumns: '200px 1fr 1fr', gap: 0,
        background: AC.surface, borderRadius: 10, overflow: 'hidden',
        border: `1px solid ${AC.line}`,
      }}>
        {/* header row */}
        <div style={{ padding: '18px 22px', background: AC.bg, borderBottom: `1px solid ${AC.line}` }}/>
        <div style={{
          padding: '18px 22px', background: AC.bg,
          borderBottom: `1px solid ${AC.line}`, borderLeft: `1px solid ${AC.line}`,
          fontFamily: AF.mono, fontSize: 11, fontWeight: 700,
          letterSpacing: '0.2em', textTransform: 'uppercase', color: AC.ink3,
        }}>Traditional ML</div>
        <div style={{
          padding: '18px 22px', background: AC.bg,
          borderBottom: `1px solid ${AC.line}`, borderLeft: `1px solid ${AC.line}`,
          fontFamily: AF.mono, fontSize: 11, fontWeight: 700,
          letterSpacing: '0.2em', textTransform: 'uppercase', color: AC.teal,
        }}>Claude agent</div>

        {rows.map((r, i) => (
          <React.Fragment key={i}>
            <div style={{
              padding: '22px', borderBottom: i < rows.length - 1 ? `1px solid ${AC.line}` : 'none',
              fontFamily: AF.mono, fontSize: 11, fontWeight: 700,
              letterSpacing: '0.08em', color: AC.ink, display: 'flex', alignItems: 'center',
            }}>
              0{i + 1}
            </div>
            <div style={{
              padding: '22px',
              borderLeft: `1px solid ${AC.line}`,
              borderBottom: i < rows.length - 1 ? `1px solid ${AC.line}` : 'none',
            }}>
              <div style={{
                fontFamily: AF.body, fontSize: 18, fontWeight: 700,
                color: AC.ink, marginBottom: 8,
              }}>{r.legacy}</div>
              <div style={{
                fontFamily: AF.body, fontSize: 14, color: AC.ink2, lineHeight: 1.5,
              }}>{r.legacyDesc}</div>
            </div>
            <div style={{
              padding: '22px', background: AC.tealTint,
              borderLeft: `1px solid ${AC.line}`,
              borderBottom: i < rows.length - 1 ? `1px solid ${AC.line}` : 'none',
            }}>
              <div style={{
                fontFamily: AF.body, fontSize: 18, fontWeight: 700,
                color: AC.tealDark, marginBottom: 8,
              }}>{r.agent}</div>
              <div style={{
                fontFamily: AF.body, fontSize: 14, color: AC.ink, lineHeight: 1.5,
              }}>{r.agentDesc}</div>
            </div>
          </React.Fragment>
        ))}
      </div>
    </section>
  );
}

// ── Tech stack / data pipeline ───────────────────────────────────────
function TechStack() {
  const pipeline = [
    {
      step: '01', title: 'CMS Part B public use files',
      detail: 'Provider-level aggregates, 2013 → 2023',
      tag: 'DATA',
    },
    {
      step: '02', title: 'Normalize · specialty-scope',
      detail: 'Percentile-rank each metric within specialty cohort',
      tag: 'SQL / DUCKDB',
    },
    {
      step: '03', title: 'Rule engine',
      detail: 'Six deterministic checks, each emits status + evidence JSON',
      tag: 'TYPESCRIPT',
    },
    {
      step: '04', title: 'Claude sonnet - briefing',
      detail: 'Reads rule outputs + profile, drafts finding + risk score',
      tag: 'CLAUDE',
    },
    {
      step: '05', title: 'Analyst UI',
      detail: 'One-screen brief · approve · escalate · dismiss',
      tag: 'REACT · VERCEL',
    },
  ];

  return (
    <section style={{ padding: '72px 72px' }}>
      <Eyebrow>§ 05 · Tech stack · data pipeline</Eyebrow>
      <h2 style={{
        fontFamily: AF.body, fontSize: 40, fontWeight: 700,
        letterSpacing: '-0.025em', lineHeight: 1.1, maxWidth: 900,
        margin: '14px 0 44px', color: AC.ink, textWrap: 'balance',
      }}>
        Boring plumbing, interesting reasoning.
      </h2>

      <div style={{ position: 'relative' }}>
        {/* connector line */}
        <div style={{
          position: 'absolute', left: '10%', right: '10%',
          top: 28, height: 1, background: AC.line2, zIndex: 0,
        }}/>
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 20,
          position: 'relative', zIndex: 1,
        }}>
          {pipeline.map((p, i) => (
            <div key={p.step}>
              <div style={{
                width: 56, height: 56, borderRadius: '50%',
                background: AC.surface, border: `2px solid ${i === 3 ? AC.teal : AC.ink}`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontFamily: AF.mono, fontWeight: 700, fontSize: 14,
                color: i === 3 ? AC.teal : AC.ink, margin: '0 auto 16px',
              }}>{p.step}</div>
              <div style={{
                fontFamily: AF.mono, fontSize: 10, fontWeight: 700,
                letterSpacing: '0.18em', color: i === 3 ? AC.teal : AC.ink3,
                textAlign: 'center', marginBottom: 8,
              }}>{p.tag}</div>
              <div style={{
                fontFamily: AF.body, fontSize: 16, fontWeight: 700,
                color: AC.ink, textAlign: 'center', marginBottom: 6,
                lineHeight: 1.25,
              }}>{p.title}</div>
              <div style={{
                fontFamily: AF.body, fontSize: 13, color: AC.ink2,
                textAlign: 'center', lineHeight: 1.4,
              }}>{p.detail}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ── Limitations / honesty ────────────────────────────────────────────
function Limitations() {
  const items = [
    {
      title: 'Flagged ≠ fraudulent',
      body: 'Every escalation is a hypothesis. The agent raises hands; a human decides. High-risk does not mean guilty - it means worth 15 minutes of analyst attention.',
    },
    {
      title: 'Public data is lagged and coarse',
      body: 'CMS public files aggregate to provider-year. We cannot see single-claim anomalies, same-day impossibilities, or beneficiary overlap. The production system would ingest claim-level data.',
    },
    {
      title: 'The LLM can be wrong about narrative',
      body: 'Claude writes the finding, but never picks the score - that comes from the deterministic rule panel. Every claim in the brief can be traced to a rule output.',
    },
    {
      title: 'Specialty baselines shift slowly',
      body: 'A genuinely new treatment looks like a volume spike until the cohort catches up. We surface prior-year activity to help analysts distinguish emerging practice from opportunism.',
    },
  ];

  return (
    <section style={{
      padding: '72px 72px', background: AC.ink, color: '#fff',
    }}>
      <Eyebrow color="#F2F0ED">§ 06 · What this isn't</Eyebrow>
      <h2 style={{
        fontFamily: AF.body, fontSize: 40, fontWeight: 700,
        letterSpacing: '-0.025em', lineHeight: 1.1, maxWidth: 900,
        margin: '14px 0 44px', color: '#fff', textWrap: 'balance',
      }}>
        Honest limitations - because "AI caught fraud" is a load-bearing claim.
      </h2>
      <div style={{
        display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 28,
      }}>
        {items.map((it, i) => (
          <div key={i} style={{
            padding: '24px 26px', background: 'rgba(255,255,255,0.04)',
            borderRadius: 10, border: '1px solid rgba(255,255,255,0.08)',
          }}>
            <div style={{
              fontFamily: AF.mono, fontSize: 11, fontWeight: 700,
              letterSpacing: '0.18em', color: AC.orange, marginBottom: 12,
            }}>⚠ CAVEAT {String(i + 1).padStart(2, '0')}</div>
            <div style={{
              fontFamily: AF.body, fontSize: 20, fontWeight: 700,
              color: '#fff', marginBottom: 10, lineHeight: 1.25,
            }}>{it.title}</div>
            <div style={{
              fontFamily: AF.body, fontSize: 14, color: '#D6D3D1',
              lineHeight: 1.55, textWrap: 'pretty',
            }}>{it.body}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

// ── CTA ──────────────────────────────────────────────────────────────
function CTA() {
  return (
    <section style={{
      padding: '96px 72px', background: AC.surface2, textAlign: 'center',
    }}>
      <Eyebrow>See it run</Eyebrow>
      <h2 style={{
        fontFamily: AF.body, fontSize: 56, fontWeight: 800,
        letterSpacing: '-0.03em', lineHeight: 1.05,
        margin: '16px auto 24px', color: AC.ink, maxWidth: 900,
        textWrap: 'balance',
      }}>
        Watch the agent scan 1.26M providers and flag one in under a minute.
      </h2>
      <p style={{
        fontFamily: AF.body, fontSize: 18, color: AC.ink2,
        margin: '0 auto 36px', maxWidth: 620, lineHeight: 1.55,
      }}>
        The full product demo walks through the cost estimator, out-of-pocket
        distributions, forecast, and fraud agent in about 45 seconds.
      </p>
      <a href="https://allowancemap.vercel.app/" target="_blank" rel="noopener noreferrer" style={{
        display: 'inline-flex', alignItems: 'center', gap: 12,
        padding: '18px 32px', background: AC.ink, color: '#fff',
        fontFamily: AF.body, fontSize: 17, fontWeight: 600,
        borderRadius: 10, textDecoration: 'none',
        boxShadow: '0 10px 30px rgba(0,0,0,0.2)',
      }}>
        <span>See the live demo</span>
        <span style={{ fontFamily: AF.mono }}>→</span>
      </a>

      <div style={{
        marginTop: 56, paddingTop: 32,
        borderTop: `1px solid ${AC.line}`,
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        fontFamily: AF.mono, fontSize: 12, color: AC.ink3,
        letterSpacing: '0.08em', maxWidth: 980, margin: '56px auto 0',
      }}>
        <span>AllowanceMap · Fraud Detection Agent</span>
        <span>@rvedire.com · April 2026</span>
      </div>
    </section>
  );
}

// ── Page ─────────────────────────────────────────────────────────────
function OnePager() {
  return (
    <div style={{
      background: AC.bg, color: AC.ink, fontFamily: AF.body,
      width: 1280,
    }}>
      <Hero/>
      <MedicarePrimer/>
      <ProblemBand/>
      <HowItWorks/>
      <BriefScreenshot/>
      <WhyClaude/>
      <TechStack/>
      <Limitations/>
      <CTA/>
    </div>
  );
}

Object.assign(window, { OnePager });
