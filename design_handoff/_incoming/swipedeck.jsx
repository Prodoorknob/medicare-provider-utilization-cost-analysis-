/* Fraud Detection Agent - LinkedIn swipe deck (8 panels, 1080×1080)
   Shares the warm-light palette + type system from onepager.jsx. */

// Reuse AC / AF defined in onepager.jsx (loaded before this file)

function SwipePanel({ n, total, children, bg = '#FAFAF8', color = AC.ink, borderTop }) {
  return (
    <div style={{
      width: 1080, height: 1080, background: bg, color,
      fontFamily: AF.body, position: 'relative', overflow: 'hidden',
      borderTop: borderTop || 'none',
    }}>
      {children}
      {/* Footer chrome - consistent across panels */}
      <div style={{
        position: 'absolute', bottom: 36, left: 48, right: 48,
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        fontFamily: AF.mono, fontSize: 13, letterSpacing: '0.16em',
        textTransform: 'uppercase',
        color: color === AC.ink ? AC.ink3 : 'rgba(255,255,255,0.55)',
      }}>
        <span>@rvedire.com</span>
        <span style={{
          display: 'flex', alignItems: 'center', gap: 10,
        }}>
          <span>{String(n).padStart(2, '0')} / {String(total).padStart(2, '0')}</span>
          {n < total && (
            <span style={{
              fontSize: 16, letterSpacing: 0, color: color === AC.ink ? AC.teal : AC.tealLight,
            }}>swipe →</span>
          )}
        </span>
      </div>
    </div>
  );
}

function PanelEyebrow({ children, color = AC.teal }) {
  return (
    <div style={{
      fontFamily: AF.mono, fontSize: 14, fontWeight: 700,
      letterSpacing: '0.24em', textTransform: 'uppercase', color,
    }}>{children}</div>
  );
}

// ── Panel 1 - HOOK ───────────────────────────────────────────────────
function P1Hook() {
  const rows = 12, cols = 34;
  const cells = [];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++)
    cells.push({ r, c, flagged: r === 7 && c === 22 });
  return (
    <div style={{ position: 'absolute', inset: 0, background: '#0D0C0B', color: '#fff' }}>
      <div style={{
        position: 'absolute', inset: 0,
        background: 'radial-gradient(ellipse at 28% 30%, rgba(220,38,38,0.18), transparent 55%), radial-gradient(ellipse at 75% 78%, rgba(19,137,172,0.2), transparent 60%)',
      }}/>

      <div style={{ position: 'absolute', top: 48, left: 48, right: 48,
        display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 12,
          fontFamily: AF.mono, fontSize: 14, fontWeight: 700,
          letterSpacing: '0.24em', textTransform: 'uppercase', color: AC.red,
        }}>
          <span style={{
            width: 10, height: 10, borderRadius: '50%', background: AC.red,
            boxShadow: `0 0 14px ${AC.red}`,
          }}/>
          Fraud Detection Agent
        </div>
        <div style={{
          fontFamily: AF.mono, fontSize: 12, color: '#A8A29E',
          letterSpacing: '0.2em', textTransform: 'uppercase',
        }}>CMS 2013–2023</div>
      </div>

      <div style={{
        position: 'absolute', top: 150, left: 48, right: 48,
        fontFamily: AF.body, fontSize: 96, fontWeight: 800,
        letterSpacing: '-0.035em', lineHeight: 0.98, color: '#fff',
      }}>
        How an AI agent<br/>
        reads Medicare<br/>
        like a <span style={{ color: AC.tealLight }}>forensic</span><br/>
        <span style={{ color: AC.tealLight }}>accountant.</span>
      </div>

      <svg viewBox="0 0 1000 180" style={{
        position: 'absolute', bottom: 220, left: 48, right: 48,
        width: 'calc(100% - 96px)', height: 180,
      }}>
        {cells.map((d, i) => {
          const x = 6 + d.c * 29;
          const y = 6 + d.r * 14;
          let c = 'rgba(255,255,255,0.18)';
          if (d.flagged) c = AC.red;
          else if ((i * 13) % 41 === 0) c = AC.orange;
          return <circle key={i} cx={x} cy={y} r={d.flagged ? 5 : 2.2} fill={c}/>;
        })}
        <circle cx={6 + 22 * 29} cy={6 + 7 * 14} r="16"
                fill="none" stroke={AC.red} strokeWidth="1.5" opacity="0.55"/>
      </svg>

      <div style={{
        position: 'absolute', bottom: 100, left: 48, right: 48,
        fontFamily: AF.body, fontSize: 22, color: '#D6D3D1',
        letterSpacing: '0.02em',
      }}>
        A Claude-powered reviewer inside <b style={{ color: '#fff' }}>AllowanceMap</b>.
      </div>
    </div>
  );
}

// ── Panel 2 - WHAT IS MEDICARE? (primer) ─────────────────────────────
function P2Medicare() {
  const parts = [
    { id: 'A', desc: 'Hospital stays, skilled nursing', focus: false },
    { id: 'B', desc: 'Doctor visits, outpatient care, labs', focus: true },
    { id: 'C', desc: 'Medicare Advantage (private bundles)', focus: false },
    { id: 'D', desc: 'Prescription drugs', focus: false },
  ];
  return (
    <>
      <div style={{ position: 'absolute', top: 72, left: 64, right: 64 }}>
        <PanelEyebrow>§ 00 · Primer · what is Medicare?</PanelEyebrow>
        <div style={{
          marginTop: 18, fontFamily: AF.body, fontSize: 60, fontWeight: 800,
          letterSpacing: '-0.03em', lineHeight: 1.02, color: AC.ink,
          textWrap: 'balance',
        }}>
          Federal health insurance for{' '}
          <span style={{ color: AC.teal }}>63 million Americans.</span>
        </div>
        <div style={{
          marginTop: 22, fontFamily: AF.body, fontSize: 22, lineHeight: 1.45,
          color: AC.ink2, maxWidth: 880, textWrap: 'pretty',
        }}>
          The U.S. federal program enacted in <b>1965</b>. Covers Americans <b>65 +</b>,
          plus some younger people with long-term disabilities or end-stage kidney failure.
        </div>
      </div>

      {/* Four parts list */}
      <div style={{
        position: 'absolute', top: 480, left: 64, right: 64,
      }}>
        <div style={{
          fontFamily: AF.mono, fontSize: 12, fontWeight: 700,
          letterSpacing: '0.22em', textTransform: 'uppercase',
          color: AC.teal, marginBottom: 18,
        }}>The four parts</div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
          {parts.map((p) => {
            const isFocus = p.focus;
            return (
              <div key={p.id} style={{
                display: 'flex', alignItems: 'center', gap: 16,
                background: isFocus ? '#FFF7F0' : AC.surface,
                border: `1px solid ${isFocus ? 'rgba(234,88,12,0.35)' : AC.line}`,
                borderLeft: `5px solid ${isFocus ? AC.orange : AC.line2}`,
                borderRadius: 10, padding: '18px 22px',
              }}>
                <div style={{
                  width: 80, height: 44, flexShrink: 0,
                  background: isFocus ? AC.orange : AC.surface2,
                  color: isFocus ? '#fff' : AC.ink,
                  border: `1px solid ${isFocus ? AC.orange : AC.line}`,
                  borderRadius: 6,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontFamily: AF.body, fontSize: 16, fontWeight: 700,
                }}>Part {p.id}</div>
                <div style={{
                  fontFamily: AF.body, fontSize: 17,
                  color: isFocus ? AC.ink : AC.ink2,
                  fontWeight: isFocus ? 600 : 500,
                  lineHeight: 1.3,
                }}>{p.desc}</div>
              </div>
            );
          })}
        </div>

        <div style={{
          marginTop: 18, display: 'flex', alignItems: 'center', gap: 10,
          fontFamily: AF.mono, fontSize: 12, fontWeight: 600,
          letterSpacing: '0.18em', textTransform: 'uppercase', color: AC.orange,
        }}>
          <span>↑ Part B is what we audit</span>
        </div>
      </div>

      {/* Footer stat strip */}
      <div style={{
        position: 'absolute', bottom: 110, left: 64, right: 64,
        paddingTop: 22, borderTop: `1px solid ${AC.line}`,
        display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 24,
      }}>
        {[
          ['~63M', 'Beneficiaries', 'Roughly 1 in 5 Americans'],
          ['80 / 20', 'Part B cost split', 'Medicare pays 80%, patient 20%'],
          ['$900B+', 'Annual spending', 'Largest U.S. health-insurance program'],
        ].map(([big, label, sub]) => (
          <div key={label}>
            <div style={{
              fontFamily: AF.body, fontSize: 38, fontWeight: 800,
              letterSpacing: '-0.035em', lineHeight: 1, color: AC.ink,
              fontVariantNumeric: 'tabular-nums',
            }}>{big}</div>
            <div style={{
              marginTop: 8, fontFamily: AF.body, fontSize: 14, fontWeight: 600,
              color: AC.ink, letterSpacing: '-0.005em',
            }}>{label}</div>
            <div style={{
              marginTop: 2, fontFamily: AF.body, fontSize: 12,
              color: AC.ink2, lineHeight: 1.35,
            }}>{sub}</div>
          </div>
        ))}
      </div>
    </>
  );
}

// ── Panel 3 - THE PROBLEM ────────────────────────────────────────────
function P2Problem() {
  return (
    <>
      <div style={{ position: 'absolute', top: 72, left: 64, right: 64 }}>
        <PanelEyebrow>§ 01 · The problem</PanelEyebrow>
        <div style={{
          marginTop: 18, fontFamily: AF.body, fontSize: 68, fontWeight: 800,
          letterSpacing: '-0.03em', lineHeight: 1.02, color: AC.ink,
          textWrap: 'balance',
        }}>
          The signal is in the data. The problem is <span style={{ color: AC.red }}>scale</span>.
        </div>
      </div>

      <div style={{
        position: 'absolute', top: 440, left: 64, right: 64,
        display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 24,
      }}>
        {[
          ['$60B+', 'Estimated annual\nimproper Medicare payments'],
          ['1.26M', 'Providers billing Part B\nin any given year'],
          ['10K+', 'Distinct procedure codes\nto cross-reference'],
        ].map(([big, label]) => (
          <div key={big} style={{
            paddingTop: 28, borderTop: `3px solid ${AC.ink}`,
          }}>
            <div style={{
              fontFamily: AF.body, fontSize: 72, fontWeight: 800,
              letterSpacing: '-0.04em', lineHeight: 1, color: AC.ink,
            }}>{big}</div>
            <div style={{
              marginTop: 18, fontFamily: AF.body, fontSize: 18,
              color: AC.ink, lineHeight: 1.35, whiteSpace: 'pre-line',
              fontWeight: 500,
            }}>{label}</div>
          </div>
        ))}
      </div>

      <div style={{
        position: 'absolute', bottom: 150, left: 64, right: 64,
        paddingTop: 28, borderTop: `1px solid ${AC.line}`,
        fontFamily: AF.body, fontSize: 22, lineHeight: 1.45,
        color: AC.ink2, maxWidth: 880, textWrap: 'pretty',
      }}>
        A human auditor can't read every row. So we built
        something that can - and writes up what it finds.
      </div>
    </>
  );
}

// ── Panel 3 - SIX RULES ──────────────────────────────────────────────
function P3Rules() {
  const rules = [
    { id: 'VOLUME_SPIKE', q: 'Same doctor. Claims tripled overnight.', color: AC.red },
    { id: 'CHARGE_INFLATION', q: 'Charging $900 for what Medicare pays $80.', color: AC.red },
    { id: 'HIGH_INTENSITY', q: 'One visit. Billed for twelve procedures.', color: AC.red },
    { id: 'OUT_OF_SPECIALTY', q: 'A dentist billing for an MRI.', color: AC.green },
    { id: 'CONCENTRATION', q: 'Every patient has the same exact diagnosis.', color: AC.green },
    { id: 'IMPOSSIBLE_DAY', q: 'Physically present in two cities. Simultaneously.', color: AC.ink3 },
  ];
  return (
    <>
      <div style={{ position: 'absolute', top: 72, left: 64, right: 64 }}>
        <PanelEyebrow>§ 02 · How it works</PanelEyebrow>
        <div style={{
          marginTop: 18, fontFamily: AF.body, fontSize: 64, fontWeight: 800,
          letterSpacing: '-0.03em', lineHeight: 1.02, color: AC.ink,
          textWrap: 'balance',
        }}>
          Six rules. Each a question<br/>
          a human investigator would ask.
        </div>
      </div>

      <div style={{
        position: 'absolute', top: 420, left: 64, right: 64,
        display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16,
      }}>
        {rules.map((r, i) => (
          <div key={r.id} style={{
            background: AC.surface, padding: '22px 24px',
            borderRadius: 10, border: `1px solid ${AC.line}`,
            borderLeft: `5px solid ${r.color}`,
          }}>
            <div style={{
              fontFamily: AF.mono, fontSize: 12, fontWeight: 700,
              letterSpacing: '0.16em', color: AC.ink3,
              marginBottom: 8,
            }}>{String(i + 1).padStart(2, '0')} · {r.id}</div>
            <div style={{
              fontFamily: AF.body, fontSize: 22, fontWeight: 700,
              color: AC.ink, lineHeight: 1.2,
            }}>"{r.q}"</div>
          </div>
        ))}
      </div>

      <div style={{
        position: 'absolute', bottom: 100, left: 64, right: 64,
        fontFamily: AF.mono, fontSize: 13, letterSpacing: '0.14em',
        color: AC.ink3, textTransform: 'uppercase',
      }}>
        <span style={{ color: AC.red }}>■</span> high-signal ·{' '}
        <span style={{ color: AC.green }}>■</span> negative control ·{' '}
        <span style={{ color: AC.ink3 }}>■</span> not evaluable (public data)
      </div>
    </>
  );
}

// ── Panel 4 - CASE (scan animation-ish frame) ────────────────────────
function P4Case() {
  return (
    <div style={{ position: 'absolute', inset: 0, background: '#0D0C0B', color: '#fff' }}>
      <div style={{
        position: 'absolute', inset: 0,
        background: 'radial-gradient(ellipse at 30% 30%, rgba(220,38,38,0.16), transparent 55%)',
      }}/>

      <div style={{ position: 'absolute', top: 72, left: 64, right: 64 }}>
        <PanelEyebrow color={AC.red}>§ 03 · In production</PanelEyebrow>
        <div style={{
          marginTop: 18, fontFamily: AF.body, fontSize: 64, fontWeight: 800,
          letterSpacing: '-0.03em', lineHeight: 1.02, color: '#fff',
          textWrap: 'balance',
        }}>
          One provider. <span style={{ color: AC.red }}>Three rules</span> triggered. Eight seconds.
        </div>
      </div>

      <div style={{
        position: 'absolute', top: 420, left: 64, right: 64,
        display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20,
      }}>
        {[
          ['+9,655%', 'YoY volume spike\non supply codes', AC.red],
          ['25.51×', 'Charge-to-allowed ratio\n(P100 within specialty)', AC.red],
          ['2.71', 'Services per unique\nbeneficiary (P99)', AC.red],
          ['88 / 100', 'Composite risk score\n- CRITICAL band', AC.tealLight],
        ].map(([big, label, color]) => (
          <div key={big} style={{
            background: 'rgba(255,255,255,0.04)', padding: '26px 28px',
            borderRadius: 10, border: '1px solid rgba(255,255,255,0.08)',
            borderLeft: `5px solid ${color}`,
          }}>
            <div style={{
              fontFamily: AF.body, fontSize: 48, fontWeight: 800,
              letterSpacing: '-0.03em', color: color === AC.tealLight ? color : '#fff',
              lineHeight: 1,
            }}>{big}</div>
            <div style={{
              marginTop: 14, fontFamily: AF.body, fontSize: 17,
              color: '#D6D3D1', lineHeight: 1.35, whiteSpace: 'pre-line',
            }}>{label}</div>
          </div>
        ))}
      </div>

      <div style={{
        position: 'absolute', bottom: 104, left: 64, right: 64,
        fontFamily: AF.body, fontSize: 18, color: '#A8A29E',
        lineHeight: 1.5, fontStyle: 'italic',
      }}>
        Identifying details hashed. Figures reflect a real 2023 outlier
        surfaced from CMS public files.
      </div>
    </div>
  );
}

// ── Panel 5 - FINDING (Claude-written blurb) ─────────────────────────
function P5Finding() {
  return (
    <>
      <div style={{ position: 'absolute', top: 72, left: 64, right: 64 }}>
        <PanelEyebrow>§ 04 · What Claude writes</PanelEyebrow>
        <div style={{
          marginTop: 18, fontFamily: AF.body, fontSize: 56, fontWeight: 800,
          letterSpacing: '-0.03em', lineHeight: 1.04, color: AC.ink,
          textWrap: 'balance',
        }}>
          Not a score. A finding an analyst can defend.
        </div>
      </div>

      <div style={{
        position: 'absolute', top: 380, left: 64, right: 64,
        background: AC.surface, padding: '36px 40px',
        borderRadius: 14, border: `1px solid ${AC.line}`,
        borderLeft: `6px solid ${AC.red}`,
        boxShadow: '0 20px 50px rgba(0,0,0,0.06)',
      }}>
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
          marginBottom: 24, paddingBottom: 20, borderBottom: `1px solid ${AC.line}`,
        }}>
          <div>
            <div style={{
              fontFamily: AF.mono, fontSize: 12, fontWeight: 700,
              letterSpacing: '0.22em', textTransform: 'uppercase', color: AC.red,
            }}>Investigation Brief · 2023</div>
            <div style={{
              fontFamily: AF.body, fontSize: 28, fontWeight: 700,
              color: AC.ink, marginTop: 6,
            }}>
              Provider <span style={{ fontFamily: AF.mono, color: AC.teal }}>#a4f··2c1</span>
              <span style={{ fontSize: 18, color: AC.ink3, fontWeight: 400, marginLeft: 10 }}>
                · Emergency Medicine · NE region
              </span>
            </div>
          </div>
          <div style={{
            padding: '10px 18px', borderRadius: 999,
            background: AC.red, color: '#fff',
            fontFamily: AF.mono, fontSize: 13, fontWeight: 700,
            letterSpacing: '0.2em',
          }}>● 88</div>
        </div>
        <div style={{
          fontFamily: AF.body, fontSize: 24, lineHeight: 1.5,
          color: AC.ink, textWrap: 'pretty',
        }}>
          <span style={{
            fontFamily: AF.mono, fontSize: 13, fontWeight: 700,
            letterSpacing: '0.2em', textTransform: 'uppercase', color: AC.ink3,
            marginRight: 12,
          }}>FINDING</span>
          Catastrophic volume spike on supply codes with a <b>25× charge-to-allowed ratio</b> -
          structurally incompatible with hospital-based emergency medicine practice.
          Prior-year dormancy rules out organic growth.
        </div>
      </div>

      <div style={{
        position: 'absolute', bottom: 100, left: 64, right: 64,
        fontFamily: AF.mono, fontSize: 13, letterSpacing: '0.16em',
        color: AC.ink3, textTransform: 'uppercase',
      }}>
        ▲ Every claim in the brief traces back to a deterministic rule output.
      </div>
    </>
  );
}

// ── Panel 6 - WHY CLAUDE ─────────────────────────────────────────────
function P6Why() {
  const rows = [
    ['Black-box classifier', 'A reviewer that writes its reasoning'],
    ['Retrain to change behavior', 'Edit rules, ship in minutes'],
    ['Ranked list of suspects', 'Triaged brief with evidence'],
  ];
  return (
    <>
      <div style={{ position: 'absolute', top: 72, left: 64, right: 64 }}>
        <PanelEyebrow>§ 05 · Why Claude · why an agent</PanelEyebrow>
        <div style={{
          marginTop: 18, fontFamily: AF.body, fontSize: 60, fontWeight: 800,
          letterSpacing: '-0.03em', lineHeight: 1.02, color: AC.ink,
          textWrap: 'balance',
        }}>
          We didn't want another classifier.<br/>
          We wanted a <span style={{ color: AC.teal }}>reviewer</span>.
        </div>
      </div>

      <div style={{
        position: 'absolute', top: 420, left: 64, right: 64,
        background: AC.surface, borderRadius: 12, overflow: 'hidden',
        border: `1px solid ${AC.line}`,
      }}>
        <div style={{
          display: 'grid', gridTemplateColumns: '1fr 1fr',
          background: AC.surface2,
          borderBottom: `1px solid ${AC.line}`,
        }}>
          <div style={{
            padding: '18px 26px', fontFamily: AF.mono, fontSize: 13, fontWeight: 700,
            letterSpacing: '0.22em', textTransform: 'uppercase', color: AC.ink3,
          }}>Traditional ML</div>
          <div style={{
            padding: '18px 26px', fontFamily: AF.mono, fontSize: 13, fontWeight: 700,
            letterSpacing: '0.22em', textTransform: 'uppercase', color: AC.teal,
            borderLeft: `1px solid ${AC.line}`,
          }}>Claude agent</div>
        </div>
        {rows.map(([l, r], i) => (
          <div key={i} style={{
            display: 'grid', gridTemplateColumns: '1fr 1fr',
            borderBottom: i < rows.length - 1 ? `1px solid ${AC.line}` : 'none',
          }}>
            <div style={{
              padding: '26px', fontFamily: AF.body, fontSize: 22, fontWeight: 600,
              color: AC.ink2, lineHeight: 1.3,
            }}>{l}</div>
            <div style={{
              padding: '26px', fontFamily: AF.body, fontSize: 22, fontWeight: 700,
              color: AC.tealDark, lineHeight: 1.3, background: AC.tealTint,
              borderLeft: `1px solid ${AC.line}`,
            }}>{r}</div>
          </div>
        ))}
      </div>
    </>
  );
}

// ── Panel 7 - HONESTY ────────────────────────────────────────────────
function P7Honesty() {
  const items = [
    ['Flagged ≠ fraudulent', 'Every escalation is a hypothesis. Humans decide. High-risk means worth 15 minutes of attention.'],
    ['Public data is lagged + coarse', 'CMS files aggregate to provider-year. Claim-level anomalies need claim-level data.'],
    ['The LLM can be wrong on narrative', 'Claude writes the finding but never picks the score - that comes from the rule panel.'],
  ];
  return (
    <div style={{ position: 'absolute', inset: 0, background: AC.ink, color: '#fff' }}>
      <div style={{ position: 'absolute', top: 72, left: 64, right: 64 }}>
        <PanelEyebrow color={AC.orange}>§ 06 · What this isn't</PanelEyebrow>
        <div style={{
          marginTop: 18, fontFamily: AF.body, fontSize: 60, fontWeight: 800,
          letterSpacing: '-0.03em', lineHeight: 1.02, color: '#fff',
          textWrap: 'balance',
        }}>
          "AI caught fraud" is<br/>
          a load-bearing claim.
        </div>
        <div style={{
          marginTop: 20, fontFamily: AF.body, fontSize: 22,
          color: '#D6D3D1', lineHeight: 1.45, maxWidth: 800,
        }}>
          Three things we're careful to say honestly:
        </div>
      </div>

      <div style={{
        position: 'absolute', top: 500, left: 64, right: 64,
        display: 'grid', gridTemplateColumns: '1fr', gap: 16,
      }}>
        {items.map(([title, body], i) => (
          <div key={i} style={{
            padding: '20px 26px',
            background: 'rgba(255,255,255,0.04)',
            borderRadius: 10, border: '1px solid rgba(255,255,255,0.08)',
            display: 'grid', gridTemplateColumns: '260px 1fr', gap: 24,
            alignItems: 'baseline',
          }}>
            <div style={{
              fontFamily: AF.body, fontSize: 22, fontWeight: 700, color: AC.orange,
              lineHeight: 1.25,
            }}>{title}</div>
            <div style={{
              fontFamily: AF.body, fontSize: 18, color: '#D6D3D1',
              lineHeight: 1.45, textWrap: 'pretty',
            }}>{body}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Panel 8 - CTA ────────────────────────────────────────────────────
function P8CTA() {
  return (
    <>
      <div style={{
        position: 'absolute', top: 140, left: 64, right: 64, textAlign: 'center',
      }}>
        <PanelEyebrow>See it run</PanelEyebrow>
        <div style={{
          marginTop: 28, fontFamily: AF.body, fontSize: 84, fontWeight: 800,
          letterSpacing: '-0.035em', lineHeight: 0.98, color: AC.ink,
          textWrap: 'balance',
        }}>
          1.26M providers.<br/>
          <span style={{ color: AC.red }}>One</span> flagged.<br/>
          Under a minute.
        </div>
      </div>

      <div style={{
        position: 'absolute', top: 620, left: 64, right: 64,
        textAlign: 'center',
      }}>
        <p style={{
          fontFamily: AF.body, fontSize: 22, lineHeight: 1.5,
          color: AC.ink2, margin: '0 auto 40px', maxWidth: 760,
          textWrap: 'pretty',
        }}>
          Watch the full AllowanceMap demo - cost estimator, OOP
          distribution, forecast, and the fraud agent in ~45s.
        </p>
        <a href="https://allowancemap.vercel.app/" target="_blank" rel="noopener noreferrer" data-cta-light style={{
          display: 'inline-flex', alignItems: 'center', gap: 14,
          padding: '22px 40px', background: AC.ink, color: '#fff',
          fontFamily: AF.body, fontSize: 22, fontWeight: 600,
          borderRadius: 12, textDecoration: 'none',
          boxShadow: '0 10px 30px rgba(0,0,0,0.15)',
        }}>
          <span style={{ color: '#fff' }}>See the live demo</span>
          <span style={{ fontFamily: AF.mono, color: '#fff' }}>→</span>
        </a>
        <div style={{
          marginTop: 56,
          fontFamily: AF.mono, fontSize: 13, color: AC.ink3,
          letterSpacing: '0.2em', textTransform: 'uppercase',
        }}>
          Follow for more teardowns →
        </div>
      </div>
    </>
  );
}

// ── Deck ─────────────────────────────────────────────────────────────
function SwipeDeck() {
  const panels = [
    P1Hook, P2Medicare, P2Problem, P3Rules, P4Case, P5Finding, P6Why, P7Honesty, P8CTA,
  ];
  // Darker bg panels - index-based config so footer color adapts
  const darkIdx = new Set([0, 4, 7]); // P1 hook, P4 case, P7 honesty
  return (
    <div style={{
      display: 'grid', gridTemplateColumns: '1080px',
      gap: 40, padding: 0,
    }}>
      {panels.map((P, i) => (
        <SwipePanel
          key={i} n={i + 1} total={panels.length}
          bg={darkIdx.has(i) ? '#0D0C0B' : AC.bg}
          color={darkIdx.has(i) ? '#fff' : AC.ink}
        >
          <P/>
        </SwipePanel>
      ))}
    </div>
  );
}

Object.assign(window, { SwipeDeck, P1Hook, P2Medicare, P2Problem, P3Rules, P4Case, P5Finding, P6Why, P7Honesty, P8CTA, SwipePanel });
