/* One-screen summary card - single screenshot, high-shareability */

function SummaryCard() {
  return (
    <div style={{
      width: 1200, height: 1600, background: AC.bg, color: AC.ink,
      fontFamily: AF.body, padding: '56px 64px', position: 'relative',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        paddingBottom: 22, borderBottom: `2px solid ${AC.ink}`,
      }}>
        <div style={{
          fontFamily: AF.mono, fontSize: 12, fontWeight: 700,
          letterSpacing: '0.24em', textTransform: 'uppercase', color: AC.red,
          display: 'flex', alignItems: 'center', gap: 10,
        }}>
          <Dot color={AC.red} glow/>
          FRAUD DETECTION AGENT
        </div>
        <div style={{
          fontFamily: AF.mono, fontSize: 11, letterSpacing: '0.18em',
          color: AC.ink3, textTransform: 'uppercase',
        }}>ALLOWANCEMAP · CMS 2013-2023</div>
      </div>

      {/* Title */}
      <h1 style={{
        fontFamily: AF.body, fontSize: 72, fontWeight: 800,
        letterSpacing: '-0.035em', lineHeight: 1.0,
        margin: '28px 0 18px', color: AC.ink, textWrap: 'balance',
      }}>
        How an AI agent reads Medicare like a <span style={{ color: AC.teal }}>forensic accountant</span>.
      </h1>
      <p style={{
        fontFamily: AF.body, fontSize: 20, lineHeight: 1.5,
        color: AC.ink2, margin: 0, maxWidth: 920,
      }}>
        A Claude-powered reviewer that scans <b>1.26M providers</b>, flags
        statistical outliers, and hands analysts an evidence-backed brief
        in seconds.
      </p>

      {/* Stats */}
      <div style={{
        marginTop: 40, display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 18,
      }}>
        {[
          ['1.26M', 'providers scanned', AC.ink],
          ['6', 'transparent rule checks', AC.ink],
          ['< 8s', 'to produce a brief', AC.ink],
          ['88/100', 'risk score, example case', AC.red],
        ].map(([big, label, c]) => (
          <div key={big} style={{
            padding: '22px 22px', background: AC.surface, borderRadius: 10,
            border: `1px solid ${AC.line}`,
          }}>
            <div style={{
              fontFamily: AF.body, fontSize: 44, fontWeight: 800,
              letterSpacing: '-0.03em', color: c, lineHeight: 1,
            }}>{big}</div>
            <div style={{
              marginTop: 10, fontFamily: AF.mono, fontSize: 11,
              letterSpacing: '0.12em', color: AC.ink3, textTransform: 'uppercase',
            }}>{label}</div>
          </div>
        ))}
      </div>

      {/* Six rules */}
      <div style={{ marginTop: 36 }}>
        <div style={{
          fontFamily: AF.mono, fontSize: 11, fontWeight: 700,
          letterSpacing: '0.22em', textTransform: 'uppercase', color: AC.teal,
          marginBottom: 14,
        }}>SIX RULES, SIX HUMAN QUESTIONS</div>
        <div style={{
          display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10,
        }}>
          {[
            ['VOLUME_SPIKE', 'Same doctor. Claims tripled overnight.', AC.red],
            ['CHARGE_INFLATION', 'Charging $900 for what Medicare pays $80.', AC.red],
            ['HIGH_INTENSITY', 'One visit. Billed for twelve procedures.', AC.red],
            ['OUT_OF_SPECIALTY', 'A dentist billing for an MRI.', AC.green],
            ['CONCENTRATION', 'Every patient has the same exact diagnosis.', AC.green],
            ['IMPOSSIBLE_DAY', 'Physically present in two cities. Simultaneously.', AC.ink3],
          ].map(([id, q, c]) => (
            <div key={id} style={{
              padding: '14px 16px', background: AC.surface, borderRadius: 8,
              border: `1px solid ${AC.line}`, borderLeft: `4px solid ${c}`,
            }}>
              <div style={{
                fontFamily: AF.mono, fontSize: 10, fontWeight: 700,
                letterSpacing: '0.14em', color: AC.ink3,
              }}>{id}</div>
              <div style={{
                fontFamily: AF.body, fontSize: 16, fontWeight: 600,
                color: AC.ink, marginTop: 4,
              }}>"{q}"</div>
            </div>
          ))}
        </div>
      </div>

      {/* Why Claude summary */}
      <div style={{
        marginTop: 32, padding: '24px 26px',
        background: AC.tealTint, borderRadius: 10,
        borderLeft: `4px solid ${AC.teal}`,
      }}>
        <div style={{
          fontFamily: AF.mono, fontSize: 11, fontWeight: 700,
          letterSpacing: '0.22em', textTransform: 'uppercase', color: AC.teal,
          marginBottom: 10,
        }}>WHY AN AGENT, NOT A CLASSIFIER</div>
        <div style={{
          fontFamily: AF.body, fontSize: 18, color: AC.ink, lineHeight: 1.5,
          textWrap: 'pretty',
        }}>
          A classifier gives you a score. An agent writes the reasoning.
          Claude reads the rule outputs and produces a prose finding an
          analyst can defend in a meeting - with every claim traceable to
          a deterministic rule.
        </div>
      </div>

      {/* Footer */}
      <div style={{
        position: 'absolute', bottom: 40, left: 64, right: 64,
        paddingTop: 20, borderTop: `1px solid ${AC.line}`,
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        fontFamily: AF.mono, fontSize: 12, letterSpacing: '0.14em',
        color: AC.ink2,
      }}>
        <span>@rvedire.com · April 2026</span>
        <span style={{ color: AC.teal }}>see the live demo →</span>
      </div>
    </div>
  );
}

Object.assign(window, { SummaryCard });
