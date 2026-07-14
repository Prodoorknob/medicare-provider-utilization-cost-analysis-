/* All slide components for AllowanceMap deck */

const TOTAL = 19;

// ─────────────────────────────────────────────────────────
// 01. COVER
function SlideCover() {
  return (
    <Slide bg={C.surface} style={{ padding: 0 }}>
      {/* Split composition: left content, right brand mark */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', height: '100%' }}>
        <div style={{
          padding: `${SP.pt}px ${SP.px}px ${SP.pb}px`,
          display: 'flex', flexDirection: 'column', justifyContent: 'space-between',
          background: C.surface,
        }}>
          <div>
            <div style={{
              fontFamily: FONT.mono, fontSize: TYPE.eyebrow, fontWeight: 700,
              letterSpacing: '0.22em', textTransform: 'uppercase', color: C.primary,
            }}>AllowanceMap · v1.0 · April 2026</div>
            <div style={{ width: 72, height: 4, background: C.primary, marginTop: 36 }}/>
          </div>
          <div>
            <div style={{
              fontSize: 120, fontWeight: 900, letterSpacing: '-0.035em',
              lineHeight: 0.98, color: C.text,
            }}>
              Medicare<br/>
              <span style={{ color: C.primary }}>Provider Cost</span><br/>
              Analysis
            </div>
            <div style={{
              fontSize: TYPE.subtitle, color: C.text2, marginTop: 40,
              maxWidth: 820, lineHeight: 1.3,
            }}>
              An end-to-end ML pipeline predicting what Medicare allows
              and what patients pay — built on 103M CMS records, 2013–2023.
            </div>
          </div>
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end',
            fontFamily: FONT.mono, fontSize: TYPE.micro, color: C.text3,
            letterSpacing: '0.14em', textTransform: 'uppercase',
          }}>
            <div>
              <div style={{ color: C.text, fontSize: TYPE.small, fontFamily: FONT.body, fontWeight: 600, letterSpacing: 0, textTransform: 'none', marginBottom: 4 }}>Raj Vedire</div>
              <div>Indiana University · rvedire@iu.edu</div>
            </div>
            <div>allowancemap.vercel.app</div>
          </div>
        </div>
        <div style={{
          background: C.primary, position: 'relative', overflow: 'hidden',
        }}>
          {/* Big concentric rings = mapping / coverage metaphor */}
          <svg viewBox="0 0 600 1080" preserveAspectRatio="xMidYMid slice" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}>
            {[640, 540, 440, 340, 240, 140].map((r, i) => (
              <circle key={i} cx="380" cy="540" r={r}
                fill="none" stroke="rgba(255,255,255,0.12)" strokeWidth="1"/>
            ))}
            <circle cx="380" cy="540" r="80" fill={C.secondaryLight} opacity="0.9"/>
            <circle cx="380" cy="540" r="40" fill={C.accent}/>
          </svg>
          <div style={{
            position: 'absolute', top: 80, left: 60, right: 60,
            color: 'rgba(255,255,255,0.9)', fontFamily: FONT.mono,
            fontSize: TYPE.micro, letterSpacing: '0.18em', textTransform: 'uppercase',
          }}>
            63 state/territory markets · 131 specialties · 11 years
          </div>
          <div style={{
            position: 'absolute', bottom: 80, left: 60, right: 60,
            display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 32,
            color: '#fff',
          }}>
            <div>
              <div style={{ fontFamily: FONT.mono, fontSize: TYPE.metricSm, fontWeight: 700, lineHeight: 1 }}>103M</div>
              <div style={{ fontSize: TYPE.tiny, color: 'rgba(255,255,255,0.7)', marginTop: 8 }}>provider-service records</div>
            </div>
            <div>
              <div style={{ fontFamily: FONT.mono, fontSize: TYPE.metricSm, fontWeight: 700, lineHeight: 1 }}>0.943</div>
              <div style={{ fontSize: TYPE.tiny, color: 'rgba(255,255,255,0.7)', marginTop: 8 }}>Stage 1 test R²</div>
            </div>
          </div>
        </div>
      </div>
    </Slide>
  );
}

// 02. THE PROBLEM
function SlideProblem({ n }) {
  return (
    <Slide>
      <Eyebrow>01 · Context</Eyebrow>
      <Title>Medicare pricing is a <span style={{color:C.primary}}>black box</span> for the people who pay for it.</Title>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 32, marginTop: 24 }}>
        {[
          {
            k: '63M',
            l: 'Medicare beneficiaries in the U.S.',
            d: 'Each facing cost decisions with almost no upfront pricing information at the point of care.',
          },
          {
            k: '~$33',
            l: 'CMS conversion factor per RVU',
            d: 'Published, but buried behind a 3-component RVU formula and 100+ geographic locality adjusters.',
          },
          {
            k: '2 stages',
            l: 'Allowed amount → patient OOP',
            d: 'Most existing tools collapse these into one number. They are two different questions with two different data sources.',
          },
        ].map((x, i) => (
          <div key={i} style={{
            background: C.surface, border: `1px solid ${C.border}`,
            borderRadius: 12, padding: 32, display: 'flex', flexDirection: 'column', gap: 16,
          }}>
            <div style={{ fontFamily: FONT.mono, fontSize: TYPE.metricSm, fontWeight: 700, color: C.primary, lineHeight: 1 }}>{x.k}</div>
            <div style={{ fontSize: TYPE.small, fontWeight: 600, color: C.text }}>{x.l}</div>
            <div style={{ fontSize: TYPE.tiny, color: C.text2, lineHeight: 1.5 }}>{x.d}</div>
          </div>
        ))}
      </div>
      <div style={{
        marginTop: 40, padding: '24px 32px',
        background: C.accentTint, borderLeft: `4px solid ${C.accent}`, borderRadius: 8,
        fontSize: TYPE.small, color: C.text, lineHeight: 1.5,
      }}>
        Allowed-amount pricing follows the formula&nbsp;
        <Mono color={C.accent} size={TYPE.small}>Allowed = [(Work RVU × Work GPCI) + (PE RVU × PE GPCI) + (MP RVU × MP GPCI)] × CF</Mono>
        &nbsp;— a closed-form rule the public never sees.
      </div>
      <Footer n={n} total={TOTAL} label="The problem" />
    </Slide>
  );
}

// 03. OBJECTIVE
function SlideObjective({ n }) {
  return (
    <Slide bg={C.primary}>
      <Eyebrow color="rgba(255,255,255,0.7)">02 · Objective</Eyebrow>
      <div style={{ color: '#fff', fontSize: 80, fontWeight: 800, letterSpacing: '-0.025em', lineHeight: 1.05, maxWidth: 1500 }}>
        Build a public-facing tool that gives Medicare beneficiaries a <span style={{color:C.primaryTint}}>trustworthy, uncertainty-aware</span> cost estimate before care.
      </div>
      <div style={{
        marginTop: 72, display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)',
        gap: 24, color: '#fff',
      }}>
        {[
          ['Predictive',  'Not a lookup — a model that generalizes across specialty × state × procedure.'],
          ['Transparent', 'Every estimate surfaces P10 / P50 / P90 bounds, never a single false-precision point.'],
          ['Two-stage',   'Decouple what Medicare allows (Stage 1) from what the patient actually pays (Stage 2).'],
          ['Forward-looking', 'LSTM + foundation-model ensemble forecasts rates 2024–2026 at specialty resolution.'],
        ].map(([h, b], i) => (
          <div key={i} style={{ borderTop: '2px solid rgba(255,255,255,0.35)', paddingTop: 20 }}>
            <div style={{ fontFamily: FONT.mono, fontSize: TYPE.micro, letterSpacing: '0.16em', textTransform: 'uppercase', color: 'rgba(255,255,255,0.65)', marginBottom: 8 }}>0{i+1}</div>
            <div style={{ fontSize: TYPE.small, fontWeight: 700, color: '#fff', marginBottom: 12 }}>{h}</div>
            <div style={{ fontSize: TYPE.tiny, color: 'rgba(255,255,255,0.78)', lineHeight: 1.5 }}>{b}</div>
          </div>
        ))}
      </div>
      <div style={{
        position: 'absolute', bottom: 36, left: SP.px, right: SP.px,
        display: 'flex', justifyContent: 'space-between',
        fontFamily: FONT.mono, fontSize: TYPE.micro, color: 'rgba(255,255,255,0.6)',
        letterSpacing: '0.12em', textTransform: 'uppercase',
      }}>
        <span>AllowanceMap · Medicare Provider Cost Analysis</span>
        <span>Objective</span>
        <span>{String(n).padStart(2,'0')} / {String(TOTAL).padStart(2,'0')}</span>
      </div>
    </Slide>
  );
}

// 04. DATASET SCOPE
function SlideDataset({ n }) {
  return (
    <Slide>
      <Eyebrow>03 · Dataset scope</Eyebrow>
      <Title>Four CMS datasets, <span style={{color:C.primary}}>103M rows</span>, 11 years.</Title>

      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 40, marginTop: 16 }}>
        <div>
          <div style={{
            border: `1px solid ${C.border}`, borderRadius: 12, overflow: 'hidden',
            background: C.surface,
          }}>
            <div style={{
              display: 'grid', gridTemplateColumns: '2.2fr 0.9fr 1fr 1.5fr',
              padding: '18px 24px', fontFamily: FONT.mono, fontSize: TYPE.micro,
              letterSpacing: '0.14em', textTransform: 'uppercase', color: C.text3,
              borderBottom: `1px solid ${C.border}`, background: C.surface2,
            }}>
              <div>Dataset</div><div>Years</div><div style={{textAlign:'right'}}>Records</div><div>Used for</div>
            </div>
            {[
              ['CMS Medicare Physician & Other Practitioners', '2013–23', '~103M',   'Stage 1 training'],
              ['CMS Medicare Provider Summary (by NPI)',       '2013–23', '~10M NPIs', 'HCC risk-score join'],
              ['MCBS Cost Supplement PUF',                     '2018–23', '~30K/yr', 'Stage 2 OOP distributions'],
              ['MCBS Survey File PUF',                         '2015–23', '~15K/yr', 'Demographic synthesis'],
            ].map((row, i) => (
              <div key={i} style={{
                display: 'grid', gridTemplateColumns: '2.2fr 0.9fr 1fr 1.5fr',
                padding: '20px 24px', alignItems: 'baseline',
                borderBottom: i < 3 ? `1px solid ${C.border}` : 'none',
                fontSize: TYPE.tiny,
              }}>
                <div style={{ color: C.text, fontWeight: 500 }}>{row[0]}</div>
                <div style={{ fontFamily: FONT.mono, color: C.text2 }}>{row[1]}</div>
                <div style={{ fontFamily: FONT.mono, color: C.primary, fontWeight: 600, textAlign: 'right' }}>{row[2]}</div>
                <div style={{ color: C.text2 }}>{row[3]}</div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 24, fontSize: TYPE.tiny, color: C.text3, lineHeight: 1.5 }}>
            All datasets are public. Part B physician/practitioner only — no Part A hospital facility fees,
            no Part D drug claims, no Medicare Advantage encounters.
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <MetricBig value="103M" label="provider-service rows" accent={C.primary} />
          <MetricBig value="131"  label="encoded specialties"     accent={C.primary} />
          <MetricBig value="~6K"  label="unique HCPCS codes"      accent={C.secondary} />
          <MetricBig value="63"   label="state / territory markets" accent={C.secondary} />
        </div>
      </div>
      <Footer n={n} total={TOTAL} label="Dataset scope" />
    </Slide>
  );
}

function MetricBig({ value, label, accent }) {
  return (
    <div style={{
      background: C.surface, border: `1px solid ${C.border}`, borderLeft: `4px solid ${accent}`,
      borderRadius: 10, padding: '18px 22px', display: 'flex',
      alignItems: 'baseline', justifyContent: 'space-between',
    }}>
      <div style={{ fontFamily: FONT.mono, fontSize: 56, fontWeight: 700, color: accent, lineHeight: 1 }}>{value}</div>
      <div style={{ fontSize: TYPE.tiny, color: C.text2, textAlign: 'right', maxWidth: 180 }}>{label}</div>
    </div>
  );
}

// 05. TWO-STAGE PIPELINE
function SlidePipeline({ n }) {
  return (
    <Slide>
      <Eyebrow>04 · Architecture</Eyebrow>
      <Title>Two independent models, <span style={{color:C.primary}}>one composable pipeline</span>.</Title>

      <div style={{ marginTop: 16 }}>
        {/* Pipeline diagram */}
        <div style={{ display: 'grid', gridTemplateColumns: '0.5fr 2fr 0.3fr 2fr 0.3fr 1fr', alignItems: 'center', gap: 20 }}>
          <Node label="User input" sub="specialty, state, HCPCS, POS" tone="neutral" />
          <Arrow />
          <Node label="Stage 1" title="Medicare allowed amount"
                sub="LightGBM V2 no-charge"
                metric="R² 0.9428"
                tone="primary" big />
          <Arrow />
          <Node label="Stage 2" title="Patient out-of-pocket"
                sub="XGB Quantile V1 · P10 / P50 / P90"
                metric="P50 R² 0.40"
                tone="secondary" big />
          <Node label="$" sub="estimate with bounds" tone="accent" />
        </div>

        {/* Second row: forecast */}
        <div style={{ marginTop: 72, padding: '24px 32px', background: C.surface2, borderRadius: 12, border: `1px dashed ${C.border2}` }}>
          <div style={{ display: 'grid', gridTemplateColumns: '0.6fr 2fr 2.5fr', alignItems: 'center', gap: 32 }}>
            <div style={{ fontFamily: FONT.mono, fontSize: TYPE.micro, letterSpacing: '0.14em', textTransform: 'uppercase', color: C.text3 }}>
              Independent track
            </div>
            <div>
              <div style={{ fontSize: TYPE.small, fontWeight: 700, color: C.text }}>Forecast — 2024–2026</div>
              <div style={{ fontSize: TYPE.tiny, color: C.text2, marginTop: 6 }}>LGB Stacker V2_12 · LSTM + Chronos base learners</div>
            </div>
            <div style={{ display: 'flex', gap: 28, fontFamily: FONT.mono }}>
              <Kv k="R²" v="0.8852" c={C.primary}/>
              <Kv k="MAE" v="$8.74" c={C.primary}/>
              <Kv k="RMSE" v="$17.69" c={C.primary}/>
              <Kv k="Groups" v="20,572" c={C.text2}/>
            </div>
          </div>
        </div>

        <div style={{ marginTop: 36, fontSize: TYPE.tiny, color: C.text3, lineHeight: 1.5 }}>
          Stage 1 predicts the per-service Medicare allowed amount. Stage 2 consumes that prediction
          plus beneficiary demographics and returns an OOP distribution. The forecast track is
          architecturally independent and powers the Forecast Explorer page.
        </div>
      </div>
      <Footer n={n} total={TOTAL} label="Two-stage pipeline" />
    </Slide>
  );
}

function Node({ label, title, sub, metric, tone, big }) {
  const colors = {
    neutral:   { bg: C.surface, fg: C.text, bd: C.border2, acc: C.text3 },
    primary:   { bg: C.primaryTint, fg: C.primary, bd: C.primary, acc: C.primary },
    secondary: { bg: C.secondaryTint, fg: C.secondary, bd: C.secondary, acc: C.secondary },
    accent:    { bg: C.accentTint, fg: C.accent, bd: C.accent, acc: C.accent },
  }[tone];
  return (
    <div style={{
      background: colors.bg, border: `1.5px solid ${colors.bd}`,
      borderRadius: 12, padding: big ? 24 : 18,
      minHeight: big ? 160 : 100,
      display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 8,
    }}>
      <div style={{
        fontFamily: FONT.mono, fontSize: TYPE.micro, letterSpacing: '0.14em',
        textTransform: 'uppercase', color: colors.acc, fontWeight: 700,
      }}>{label}</div>
      {title && <div style={{ fontSize: TYPE.small, fontWeight: 700, color: C.text }}>{title}</div>}
      {sub && <div style={{ fontSize: TYPE.tiny, color: C.text2 }}>{sub}</div>}
      {metric && (
        <div style={{ fontFamily: FONT.mono, fontSize: TYPE.small, fontWeight: 700, color: colors.fg, marginTop: 4 }}>{metric}</div>
      )}
    </div>
  );
}
function Arrow() {
  return (
    <div style={{ display: 'flex', justifyContent: 'center' }}>
      <svg width="40" height="20" viewBox="0 0 40 20"><path d="M0 10 L34 10 M28 4 L34 10 L28 16" stroke={C.text3} strokeWidth="1.8" fill="none" strokeLinecap="round" strokeLinejoin="round"/></svg>
    </div>
  );
}
function Kv({ k, v, c }) {
  return (
    <div>
      <div style={{ fontFamily: FONT.mono, fontSize: TYPE.micro, letterSpacing: '0.12em', textTransform: 'uppercase', color: C.text3, marginBottom: 4 }}>{k}</div>
      <div style={{ fontFamily: FONT.mono, fontSize: TYPE.small, fontWeight: 700, color: c }}>{v}</div>
    </div>
  );
}

// 06. STAGE 1 OVERVIEW
function SlideStage1Intro({ n }) {
  return (
    <Slide>
      <Eyebrow>05 · Stage 1</Eyebrow>
      <Title>Predict the <span style={{color:C.primary}}>Medicare allowed amount</span> per service.</Title>

      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 48, marginTop: 16 }}>
        <div>
          <div style={{ fontFamily: FONT.mono, fontSize: TYPE.micro, letterSpacing: '0.14em', textTransform: 'uppercase', color: C.text3, marginBottom: 12 }}>
            Target variable
          </div>
          <div style={{ fontFamily: FONT.mono, fontSize: TYPE.subtitle, color: C.primary, fontWeight: 700, marginBottom: 8 }}>
            Avg_Mdcr_Alowd_Amt
          </div>
          <Body size={TYPE.small} style={{ marginBottom: 32 }}>
            Average dollar amount Medicare allows per group-year (specialty × state × HCPCS × POS).
            Log-transformed (log1p) during training, inverse-transformed at inference.
          </Body>

          <div style={{ fontFamily: FONT.mono, fontSize: TYPE.micro, letterSpacing: '0.14em', textTransform: 'uppercase', color: C.text3, marginBottom: 12 }}>
            Leakage removed
          </div>
          <Body size={TYPE.tiny}>
            <Mono color={C.error} size={TYPE.tiny}>Avg_Mdcr_Pymt_Amt</Mono> and
            &nbsp;<Mono color={C.error} size={TYPE.tiny}>Avg_Mdcr_Stdzd_Amt</Mono>
            &nbsp;are algebraically derived from the target and excluded from every model.
          </Body>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 28 }}>
            <div style={{ fontFamily: FONT.mono, fontSize: TYPE.micro, letterSpacing: '0.14em', textTransform: 'uppercase', color: C.text3, marginBottom: 6 }}>Eval</div>
            <div style={{ fontSize: TYPE.tiny, color: C.text2, lineHeight: 1.6 }}>
              80/20 random split · seed 42 · individual group-HCPCS rows (126.8M in V2)
            </div>
          </div>
          <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 28 }}>
            <div style={{ fontFamily: FONT.mono, fontSize: TYPE.micro, letterSpacing: '0.14em', textTransform: 'uppercase', color: C.text3, marginBottom: 6 }}>Compute</div>
            <div style={{ fontSize: TYPE.tiny, color: C.text2, lineHeight: 1.6 }}>
              Colab Pro A100 · ~4 hrs per full run · MLflow-logged
            </div>
          </div>
          <div style={{ background: C.primaryTint, border: `1px solid ${C.primary}`, borderRadius: 12, padding: 28 }}>
            <div style={{ fontFamily: FONT.mono, fontSize: TYPE.micro, letterSpacing: '0.14em', textTransform: 'uppercase', color: C.primary, marginBottom: 6 }}>Production model</div>
            <div style={{ fontFamily: FONT.mono, fontSize: TYPE.subtitle, fontWeight: 700, color: C.primary }}>LightGBM V2 no-charge</div>
            <div style={{ fontSize: TYPE.tiny, color: C.text2, marginTop: 4 }}>deployed on Railway · serves the homepage estimator</div>
          </div>
        </div>
      </div>
      <Footer n={n} total={TOTAL} label="Stage 1 · allowed amount" />
    </Slide>
  );
}

// 07. STAGE 1 FEATURES
function SlideStage1Features({ n }) {
  const features = [
    ['Rndrng_Prvdr_Type_idx',     'Encoded provider specialty', 'cat'],
    ['Rndrng_Prvdr_State_Abrvtn_idx', 'Encoded state',          'cat'],
    ['HCPCS_Cd_idx',              'Encoded HCPCS code (~6K)',   'cat'],
    ['hcpcs_bucket',              'Clinical category 0–5',      'cat'],
    ['place_of_srvc_flag',        'Facility (1) vs office (0)', 'bin'],
    ['Bene_Avg_Risk_Scre',        'NPI-level HCC risk score',   'num'],
    ['log_srvcs',                 'log1p(Tot_Srvcs)',           'num'],
    ['log_benes',                 'log1p(Tot_Benes)',           'num'],
    ['Avg_Sbmtd_Chrg',            'Submitted charge',           'num', 'dropped'],
    ['srvcs_per_bene',            'Services per beneficiary',   'num'],
    ['specialty_bucket',          'Coarse specialty grouping',  'cat'],
    ['pos_bucket',                'Place-of-service bucket',    'cat'],
    ['hcpcs_target_enc',          'HCPCS target encoding',      'num'],
  ];
  const tone = { cat: C.primary, num: C.secondary, bin: C.accent };
  return (
    <Slide>
      <Eyebrow>06 · Stage 1 features</Eyebrow>
      <Title>13 features — <span style={{color:C.primary}}>12 in production</span> (charge dropped).</Title>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, marginTop: 16 }}>
        {features.map(([name, desc, kind, flag], i) => (
          <div key={i} style={{
            display: 'grid', gridTemplateColumns: '28px 1fr 1fr 90px',
            alignItems: 'center', padding: '14px 18px',
            background: flag ? C.accentTint : C.surface,
            border: `1px solid ${flag ? C.accent : C.border}`,
            borderRadius: 8, gap: 16,
          }}>
            <div style={{ fontFamily: FONT.mono, fontSize: TYPE.micro, color: C.text3, fontWeight: 700 }}>{String(i+1).padStart(2,'0')}</div>
            <div style={{ fontFamily: FONT.mono, fontSize: TYPE.tiny, color: tone[kind], fontWeight: 600 }}>{name}</div>
            <div style={{ fontSize: TYPE.tiny, color: C.text2 }}>{desc}</div>
            <div style={{ textAlign: 'right' }}>
              {flag
                ? <span style={{ fontFamily: FONT.mono, fontSize: 14, color: C.accent, background: '#fff', border: `1px solid ${C.accent}`, padding: '3px 10px', borderRadius: 99, letterSpacing: '0.08em', textTransform: 'uppercase', fontWeight: 700 }}>no-chg</span>
                : <span style={{ fontFamily: FONT.mono, fontSize: 14, color: C.text3, letterSpacing: '0.08em', textTransform: 'uppercase' }}>{kind}</span>}
            </div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 28, display: 'flex', gap: 40, fontSize: TYPE.tiny, color: C.text3, fontFamily: FONT.mono, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
        <span><span style={{color:C.primary}}>■</span> categorical</span>
        <span><span style={{color:C.secondary}}>■</span> numerical</span>
        <span><span style={{color:C.accent}}>■</span> binary</span>
        <span style={{ color: C.text2, fontFamily: FONT.body, textTransform: 'none', letterSpacing: 0 }}>
          Amber rows = dropped in the deployed no-charge variant to avoid requiring charge at inference time.
        </span>
      </div>
      <Footer n={n} total={TOTAL} label="Stage 1 feature set" />
    </Slide>
  );
}

// 08. STAGE 1 LEADERBOARD
function SlideStage1Leaderboard({ n }) {
  const rows = [
    ['LightGBM V2 (full)',          6.73,  13.59, 0.9575, 'best metric · charge required'],
    ['Ensemble V2 (5-fold stack)',  6.68,  13.50, 0.9580, '+0.0005 over LGB · 13hr compute · not deployed'],
    ['XGBoost V2',                  7.73,  15.43, 0.9452, 'fallback'],
    ['CatBoost V2',                 10.88, 20.10, 0.9070, 'weakest · kept for ensemble diversity'],
    ['LightGBM V2 no-charge',       7.70,  15.77, 0.9428, 'PRODUCTION — serves live API', true],
    ['XGBoost V2 no-charge',        8.66,  17.19, 0.9319, ''],
    ['CatBoost V2 no-charge',       12.10, 22.34, 0.8849, ''],
  ];
  return (
    <Slide>
      <Eyebrow>07 · Stage 1 results</Eyebrow>
      <Title>Full 126.8M-row training. <span style={{color:C.primary}}>LightGBM wins both variants.</span></Title>

      <div style={{
        border: `1px solid ${C.border}`, borderRadius: 12, overflow: 'hidden',
        background: C.surface, marginTop: 16,
      }}>
        <div style={{
          display: 'grid', gridTemplateColumns: '3fr 1fr 1fr 1fr 2.6fr',
          padding: '18px 28px', fontFamily: FONT.mono, fontSize: TYPE.micro,
          letterSpacing: '0.14em', textTransform: 'uppercase', color: C.text3,
          borderBottom: `1px solid ${C.border}`, background: C.surface2,
        }}>
          <div>Model</div>
          <div style={{textAlign:'right'}}>Test MAE</div>
          <div style={{textAlign:'right'}}>Test RMSE</div>
          <div style={{textAlign:'right'}}>Test R²</div>
          <div>Notes</div>
        </div>
        {rows.map(([m, mae, rmse, r2, note, best], i) => (
          <div key={i} style={{
            display: 'grid', gridTemplateColumns: '3fr 1fr 1fr 1fr 2.6fr',
            padding: '18px 28px', alignItems: 'baseline',
            borderBottom: i < rows.length - 1 ? `1px solid ${C.border}` : 'none',
            background: best ? C.primaryTint : 'transparent',
            fontSize: TYPE.tiny,
          }}>
            <div style={{ color: best ? C.primary : C.text, fontWeight: best ? 800 : 500, fontSize: TYPE.small }}>
              {best && <span style={{ marginRight: 10, color: C.primary }}>★</span>}
              {m}
              {best && <span style={{ marginLeft: 12, fontFamily: FONT.mono, fontSize: 14, color: '#fff', background: C.primary, padding: '3px 10px', borderRadius: 99, letterSpacing: '0.1em' }}>PRODUCTION</span>}
            </div>
            <div style={{ fontFamily: FONT.mono, textAlign: 'right', color: best ? C.primary : C.text, fontWeight: best ? 700 : 500 }}>${mae.toFixed(2)}</div>
            <div style={{ fontFamily: FONT.mono, textAlign: 'right', color: best ? C.primary : C.text, fontWeight: best ? 700 : 500 }}>${rmse.toFixed(2)}</div>
            <div style={{ fontFamily: FONT.mono, textAlign: 'right', color: best ? C.primary : C.text, fontWeight: 700 }}>{r2.toFixed(4)}</div>
            <div style={{ color: C.text2, fontSize: TYPE.micro }}>{note}</div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 32, padding: '20px 28px', background: C.surface2, borderRadius: 10, fontSize: TYPE.tiny, color: C.text2, lineHeight: 1.6 }}>
        <strong style={{color:C.text}}>Why no-charge ships instead of the absolute-best model:</strong>&nbsp;
        the Railway API serves user estimates where the submitted charge is not known at request time.
        The 0.015 R² gap is a small price for a usable production endpoint.
      </div>
      <Footer n={n} total={TOTAL} label="Stage 1 leaderboard" />
    </Slide>
  );
}

// 09. NO-CHARGE ABLATION
function SlideNoCharge({ n }) {
  return (
    <Slide>
      <Eyebrow>08 · Ablation</Eyebrow>
      <Title>Dropping <Mono color={C.accent} size={TYPE.title} weight={800}>Avg_Sbmtd_Chrg</Mono> only costs us 0.015 R².</Title>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 48, marginTop: 16 }}>
        <div>
          <Body size={TYPE.small} style={{ marginBottom: 40 }}>
            In V1, submitted charge dominated at <strong style={{color:C.primary}}>61.8% feature importance</strong> — a near-leaky
            signal that made evaluation optimistic. The V2 no-charge ablation forces the model to
            learn from procedure identity, provider features, and utilization patterns alone.
          </Body>
          <Body size={TYPE.small} color={C.text}>
            Result: the model is <strong>genuinely predictive</strong> — not a charge-repeater — and
            therefore safe to deploy at a user-facing endpoint where the charge is not known in advance.
          </Body>
        </div>

        <div>
          {/* Visual: two bars comparing full vs no-charge R² */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 28 }}>
            <RbarRow label="LightGBM V2 (full)"       r2={0.9575} color={C.primary} />
            <RbarRow label="LightGBM V2 no-charge"   r2={0.9428} color={C.primary} highlighted />
            <RbarRow label="XGBoost V2 (full)"        r2={0.9452} color={C.secondary} />
            <RbarRow label="XGBoost V2 no-charge"     r2={0.9319} color={C.secondary} />
            <RbarRow label="CatBoost V2 (full)"       r2={0.9070} color={C.text3} />
            <RbarRow label="CatBoost V2 no-charge"    r2={0.8849} color={C.text3} />
          </div>
          <div style={{ marginTop: 24, fontFamily: FONT.mono, fontSize: TYPE.micro, color: C.text3, letterSpacing: '0.12em', textTransform: 'uppercase' }}>
            Test R² · 0.85 &nbsp;——&nbsp; 0.96
          </div>
        </div>
      </div>
      <Footer n={n} total={TOTAL} label="The no-charge ablation" />
    </Slide>
  );
}
function RbarRow({ label, r2, color, highlighted }) {
  const min = 0.85, max = 0.96;
  const pct = ((r2 - min) / (max - min)) * 100;
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
        <div style={{ fontSize: TYPE.tiny, color: C.text, fontWeight: highlighted ? 700 : 500 }}>{label}</div>
        <Mono color={color} size={TYPE.small} weight={700}>{r2.toFixed(4)}</Mono>
      </div>
      <div style={{ height: 14, background: C.surface2, borderRadius: 99, overflow: 'hidden', position: 'relative' }}>
        <div style={{
          height: '100%', width: pct + '%', background: color,
          borderRadius: 99,
          boxShadow: highlighted ? `0 0 0 3px ${C.primaryTint}` : 'none',
        }}/>
      </div>
    </div>
  );
}

// 10. FEATURE IMPORTANCE
function SlideFeatureImportance({ n }) {
  const data = [
    ['HCPCS_Cd_idx',           0.311, 'procedure identity'],
    ['hcpcs_target_enc',       0.218, 'procedure target encoding'],
    ['Avg_Sbmtd_Chrg',         0.145, 'submitted charge (dropped)', true],
    ['specialty_bucket',       0.082, 'coarse specialty'],
    ['Rndrng_Prvdr_Type_idx',  0.061, 'provider specialty'],
    ['log_srvcs',              0.055, 'log1p volume'],
    ['Bene_Avg_Risk_Scre',     0.044, 'HCC risk score'],
    ['Rndrng_Prvdr_State_Abrvtn_idx', 0.036, 'state'],
    ['place_of_srvc_flag',     0.028, 'facility/office'],
    ['log_benes',              0.020, 'log1p beneficiaries'],
  ];
  const max = 0.32;
  return (
    <Slide>
      <Eyebrow>09 · Feature importance</Eyebrow>
      <Title>Procedure identity carries the signal — <span style={{color:C.primary}}>not charge</span>.</Title>

      <div style={{ display: 'grid', gridTemplateColumns: '1.6fr 1fr', gap: 48, marginTop: 16 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {data.map(([name, imp, desc, dropped], i) => (
            <div key={i} style={{ display: 'grid', gridTemplateColumns: '280px 1fr 80px', alignItems: 'center', gap: 18 }}>
              <Mono color={dropped ? C.accent : C.primary} size={TYPE.tiny}>{name}</Mono>
              <div style={{ height: 20, background: C.surface2, borderRadius: 4, overflow: 'hidden' }}>
                <div style={{
                  height: '100%', width: `${(imp/max)*100}%`,
                  background: dropped ? C.accent : C.primary,
                  borderRadius: 4,
                }}/>
              </div>
              <Mono color={dropped ? C.accent : C.primary} size={TYPE.tiny} weight={700}>{(imp*100).toFixed(1)}%</Mono>
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 24 }}>
          <div style={{ fontFamily: FONT.mono, fontSize: TYPE.micro, letterSpacing: '0.14em', textTransform: 'uppercase', color: C.text3 }}>Takeaway</div>
          <Body size={TYPE.small} color={C.text}>
            The top two features —&nbsp;<Mono color={C.primary} size={TYPE.small}>HCPCS_Cd_idx</Mono>&nbsp;
            and&nbsp;<Mono color={C.primary} size={TYPE.small}>hcpcs_target_enc</Mono>&nbsp;—
            account for <strong>53%</strong> of gain.
          </Body>
          <Body size={TYPE.tiny}>
            Both encode the same thing: the procedure identity. The model has effectively learned
            a per-HCPCS price baseline, then adjusts for specialty, risk, and geography.
          </Body>
          <div style={{ padding: '16px 20px', background: C.accentTint, borderLeft: `3px solid ${C.accent}`, borderRadius: 8, fontSize: TYPE.tiny, color: C.text2, lineHeight: 1.5 }}>
            Next lever: swap submitted charge for the underlying&nbsp;<Mono color={C.accent} size={TYPE.tiny}>RVU × GPCI × CF</Mono>&nbsp;
            inputs from the MPFS fee schedule. Turns the model from "learn the fee schedule"
            into "learn deviations from the fee schedule."
          </div>
        </div>
      </div>
      <Footer n={n} total={TOTAL} label="Feature importance · LGB no-charge" />
    </Slide>
  );
}

// 11. STAGE 2 INTRO
function SlideStage2({ n }) {
  return (
    <Slide>
      <Eyebrow color={C.secondary}>10 · Stage 2</Eyebrow>
      <Title>Predict patient <span style={{color:C.secondary}}>out-of-pocket</span> at three quantiles.</Title>

      <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1fr', gap: 48, marginTop: 16 }}>
        <div>
          <Body size={TYPE.small} style={{ marginBottom: 32 }}>
            Stage 1's output becomes a Stage 2 input. The quantile regressor returns P10 / P50 / P90
            bounds conditioned on beneficiary context — never a single point estimate.
          </Body>

          {/* Stage 2 result card mock */}
          <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderLeft: `4px solid ${C.secondary}`, borderRadius: 12, padding: 28, boxShadow: '0 4px 16px rgba(0,0,0,0.06)' }}>
            <div style={{ fontFamily: FONT.mono, fontSize: TYPE.micro, fontWeight: 700, letterSpacing: '0.14em', textTransform: 'uppercase', color: C.secondary, marginBottom: 16 }}>
              Stage 2 · Patient Out-of-Pocket
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
              {[['Best · P10', '$84', C.secondaryLight], ['Typical · P50', '$217', C.secondary], ['High end · P90', '$492', C.secondaryDark]].map(([k, v, col], i) => (
                <div key={i} style={{ textAlign: 'center', padding: '14px 8px', borderLeft: i > 0 ? `1px solid ${C.border}` : 'none' }}>
                  <div style={{ fontFamily: FONT.mono, fontSize: TYPE.micro, letterSpacing: '0.12em', color: C.text3, textTransform: 'uppercase', marginBottom: 8 }}>{k}</div>
                  <div style={{ fontFamily: FONT.mono, fontSize: TYPE.metricSm, fontWeight: 700, color: col, lineHeight: 1 }}>{v}</div>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 20, padding: '12px 16px', background: C.accentTint, borderLeft: `3px solid ${C.accent}`, borderRadius: 8, fontSize: TYPE.micro, color: C.text2, lineHeight: 1.5 }}>
              Estimated from synthetic MCBS-derived data. Actual costs vary with plan type, deductible status, geography.
            </div>
          </div>
        </div>

        <div>
          <div style={{ fontFamily: FONT.mono, fontSize: TYPE.micro, letterSpacing: '0.14em', textTransform: 'uppercase', color: C.text3, marginBottom: 14 }}>
            Stage 2 feature set · 12 features
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            {[
              ['Avg_Mdcr_Alowd_Amt','stage 1 output',true],
              ['Bene_Avg_Risk_Scre','provider'],
              ['specialty_bucket','gold'],
              ['hcpcs_bucket','gold'],
              ['place_of_srvc_flag','gold'],
              ['census_region','MCBS'],
              ['age_band','MCBS'],
              ['sex','MCBS'],
              ['income_band','MCBS'],
              ['n_chronic_conditions','MCBS'],
              ['dual_eligible','Medicaid'],
              ['has_supplemental_insurance','Medigap'],
            ].map(([f, src, link], i) => (
              <div key={i} style={{
                padding: '10px 14px', borderRadius: 8,
                background: link ? C.primaryTint : C.surface,
                border: `1px solid ${link ? C.primary : C.border}`,
              }}>
                <div style={{ fontFamily: FONT.mono, fontSize: 13, color: link ? C.primary : C.secondary, fontWeight: 600 }}>{f}</div>
                <div style={{ fontSize: 14, color: C.text3, marginTop: 2 }}>{src}</div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 18, fontSize: TYPE.micro, color: C.text3, lineHeight: 1.6 }}>
            Highlighted row is the Stage 1 prediction — the connector between the two stages.
          </div>
        </div>
      </div>
      <Footer n={n} total={TOTAL} label="Stage 2 · patient OOP" />
    </Slide>
  );
}

// 12. STAGE 2 RESULTS
function SlideStage2Results({ n }) {
  const rows = [
    ['XGB Quantile V1',      9.78,  18.28, 0.400, '50.0%', '90.0%', 'PRODUCTION', true],
    ['CatBoost Monotonic V2', 10.55, 21.34, 0.173, '—',    '—',     'constraints fought synthetic data'],
    ['CatBoost Zero-Inflated V2', 11.95, 24.15, -0.054, '—', '—',   'gate+reg errors compound'],
  ];
  return (
    <Slide>
      <Eyebrow color={C.secondary}>11 · Stage 2 results</Eyebrow>
      <Title>V1 wins — and the reason is <span style={{color:C.secondary}}>the data, not the architecture</span>.</Title>

      <div style={{
        border: `1px solid ${C.border}`, borderRadius: 12, overflow: 'hidden',
        background: C.surface, marginTop: 16,
      }}>
        <div style={{
          display: 'grid', gridTemplateColumns: '2.4fr 1fr 1fr 1fr 1fr 1fr 2fr',
          padding: '16px 28px', fontFamily: FONT.mono, fontSize: TYPE.micro,
          letterSpacing: '0.14em', textTransform: 'uppercase', color: C.text3,
          borderBottom: `1px solid ${C.border}`, background: C.surface2,
        }}>
          <div>Model</div>
          <div style={{textAlign:'right'}}>P50 MAE</div>
          <div style={{textAlign:'right'}}>P50 RMSE</div>
          <div style={{textAlign:'right'}}>P50 R²</div>
          <div style={{textAlign:'right'}}>P50 cov</div>
          <div style={{textAlign:'right'}}>P90 cov</div>
          <div>Status</div>
        </div>
        {rows.map(([m, mae, rmse, r2, p50, p90, note, best], i) => (
          <div key={i} style={{
            display: 'grid', gridTemplateColumns: '2.4fr 1fr 1fr 1fr 1fr 1fr 2fr',
            padding: '18px 28px', alignItems: 'baseline',
            borderBottom: i < rows.length - 1 ? `1px solid ${C.border}` : 'none',
            background: best ? C.secondaryTint : 'transparent',
            fontSize: TYPE.tiny,
          }}>
            <div style={{ color: best ? C.secondary : C.text, fontWeight: best ? 800 : 500, fontSize: TYPE.small }}>
              {best && <span style={{ marginRight: 10 }}>★</span>}{m}
            </div>
            <Mono color={best ? C.secondary : C.text} size={TYPE.tiny} weight={best ? 700 : 500} style={{ textAlign: 'right', display: 'block' }}>${mae.toFixed(2)}</Mono>
            <Mono color={best ? C.secondary : C.text} size={TYPE.tiny} weight={best ? 700 : 500} style={{ textAlign: 'right', display: 'block' }}>${rmse.toFixed(2)}</Mono>
            <Mono color={best ? C.secondary : C.text} size={TYPE.tiny} weight={700} style={{ textAlign: 'right', display: 'block' }}>{r2.toFixed(3)}</Mono>
            <Mono color={C.text2} size={TYPE.tiny} style={{ textAlign: 'right', display: 'block' }}>{p50}</Mono>
            <Mono color={C.text2} size={TYPE.tiny} style={{ textAlign: 'right', display: 'block' }}>{p90}</Mono>
            <div style={{ color: best ? C.secondary : C.text2, fontWeight: best ? 700 : 400 }}>
              {best
                ? <span style={{ fontFamily: FONT.mono, fontSize: 14, color: '#fff', background: C.secondary, padding: '3px 10px', borderRadius: 99, letterSpacing: '0.1em' }}>PRODUCTION</span>
                : note}
            </div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 32, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
        <div style={{ padding: '20px 24px', background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12 }}>
          <div style={{ fontFamily: FONT.mono, fontSize: TYPE.micro, letterSpacing: '0.14em', textTransform: 'uppercase', color: C.text3, marginBottom: 8 }}>V2 monotonic — why it lost</div>
          <Body size={TYPE.tiny}>
            Five domain constraints (allowed↑→OOP↑, income↑→OOP↑, chronic↑→OOP↑, dual↓OOP, suppl↓OOP)
            plus CQR calibration. The constraints are correct a priori — but they fought the synthetic
            distribution, which does not perfectly obey them.
          </Body>
        </div>
        <div style={{ padding: '20px 24px', background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12 }}>
          <div style={{ fontFamily: FONT.mono, fontSize: TYPE.micro, letterSpacing: '0.14em', textTransform: 'uppercase', color: C.text3, marginBottom: 8 }}>Interpretation</div>
          <Body size={TYPE.tiny}>
            On real MCBS Limited Data Set OOP, monotonicity would likely help. Synthetic data makes
            architectural innovation look worse than it is. Don't over-engineer on proxy data.
          </Body>
        </div>
      </div>
      <Footer n={n} total={TOTAL} label="Stage 2 leaderboard" />
    </Slide>
  );
}

// 13. FORECAST INTRO
function SlideForecast({ n }) {
  return (
    <Slide>
      <Eyebrow>12 · Forecast track</Eyebrow>
      <Title>Project specialty-level <span style={{color:C.primary}}>allowed amounts through 2026</span>.</Title>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 48, marginTop: 16 }}>
        <div>
          <Body size={TYPE.small} style={{ marginBottom: 24 }}>
            Independent from the Stage 1/2 pipeline. Targets per-group mean allowed amount for
            (specialty × hcpcs_bucket × state), forecasted 2024–2026 with a 2022–2023 holdout.
          </Body>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
            <Stat k="Groups forecasted" v="20,572" c={C.primary}/>
            <Stat k="Forecast years"    v="2024–26" c={C.primary}/>
            <Stat k="Eval holdout"       v="2022–23" c={C.text2}/>
            <Stat k="Eval rows"          v="32,481"  c={C.text2}/>
          </div>

          <div style={{ marginTop: 28, padding: '20px 24px', background: C.accentTint, borderLeft: `3px solid ${C.accent}`, borderRadius: 8, fontSize: TYPE.tiny, color: C.text2, lineHeight: 1.6 }}>
            <strong style={{color:C.text}}>Caveat on R² comparison:</strong>&nbsp;
            Stage 1 R² (0.9575) is over individual HCPCS rows. Forecast R² (0.8852) is over
            group-year means — a smoother, aggregated surface. Do not compare them head-to-head.
          </div>
        </div>

        {/* Mini sparkline visual: history + forecast with band */}
        <div>
          <div style={{ fontFamily: FONT.mono, fontSize: TYPE.micro, letterSpacing: '0.14em', textTransform: 'uppercase', color: C.text3, marginBottom: 10 }}>
            Example: Cardiology specialty
          </div>
          <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 24 }}>
            <svg viewBox="0 0 780 400" style={{ width: '100%', height: 'auto' }}>
              {/* axis */}
              <line x1="60" y1="350" x2="760" y2="350" stroke={C.border2} strokeWidth="1"/>
              <line x1="60" y1="40" x2="60" y2="350" stroke={C.border2} strokeWidth="1"/>
              {/* grid */}
              {[100, 175, 250, 325].map((y, i) => (
                <line key={i} x1="60" y1={y} x2="760" y2={y} stroke={C.border} strokeDasharray="3 4"/>
              ))}
              {/* history line 2013-2023 */}
              {(() => {
                const hist = [[0,230],[1,220],[2,210],[3,200],[4,195],[5,185],[6,180],[7,175],[8,170],[9,162],[10,158]];
                const fc = [[10,158],[11,145],[12,138],[13,136]];
                const years = 14;
                const x = i => 60 + (i/(years-1)) * 700;
                const y = v => v;
                const histPath = hist.map((p,i)=>`${i?'L':'M'}${x(p[0])} ${y(p[1])}`).join(' ');
                const fcPath   = fc.map((p,i)=>`${i?'L':'M'}${x(p[0])} ${y(p[1])}`).join(' ');
                const band = fc.map(p=>[p[0],p[1]-22,p[1]+28]);
                const bandTop = band.map((p,i)=>`${i?'L':'M'}${x(p[0])} ${p[1]}`).join(' ');
                const bandBot = [...band].reverse().map((p,i)=>`L${x(p[0])} ${p[2]}`).join(' ');
                return (
                  <g>
                    {/* confidence band */}
                    <path d={`${bandTop} ${bandBot} Z`} fill={C.primary} opacity="0.14"/>
                    {/* history solid */}
                    <path d={histPath} stroke={C.primary} strokeWidth="3" fill="none" strokeLinecap="round"/>
                    {/* forecast dashed */}
                    <path d={fcPath} stroke={C.primary} strokeWidth="3" fill="none" strokeDasharray="6 6" strokeLinecap="round"/>
                    {hist.map((p,i)=><circle key={'h'+i} cx={x(p[0])} cy={p[1]} r="4" fill={C.primary}/>)}
                    {fc.slice(1).map((p,i)=><circle key={'f'+i} cx={x(p[0])} cy={p[1]} r="4" fill="#fff" stroke={C.primary} strokeWidth="2"/>)}
                    {/* vertical separator at 2023 */}
                    <line x1={x(10)} y1="40" x2={x(10)} y2="350" stroke={C.text3} strokeDasharray="2 4"/>
                  </g>
                );
              })()}
              {/* x labels */}
              {['2013','','2015','','2017','','2019','','2021','','2023','2024','2025','2026'].map((lbl, i) => (
                <text key={i} x={60 + (i/13)*700} y="378" fontSize="14" fill={C.text3} fontFamily={FONT.mono} textAnchor="middle">{lbl}</text>
              ))}
              {/* legend */}
              <g transform="translate(480,55)">
                <line x1="0" y1="6" x2="24" y2="6" stroke={C.primary} strokeWidth="3"/>
                <text x="30" y="10" fontSize="14" fill={C.text2} fontFamily={FONT.body}>Historical</text>
                <line x1="130" y1="6" x2="154" y2="6" stroke={C.primary} strokeWidth="3" strokeDasharray="6 6"/>
                <text x="160" y="10" fontSize="14" fill={C.text2} fontFamily={FONT.body}>Forecast</text>
                <rect x="250" y="0" width="24" height="12" fill={C.primary} opacity="0.14"/>
                <text x="280" y="10" fontSize="14" fill={C.text2} fontFamily={FONT.body}>P10–P90 band</text>
              </g>
            </svg>
          </div>
        </div>
      </div>
      <Footer n={n} total={TOTAL} label="Forecast track · 2024–2026" />
    </Slide>
  );
}
function Stat({ k, v, c }) {
  return (
    <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: '16px 20px' }}>
      <div style={{ fontFamily: FONT.mono, fontSize: TYPE.micro, letterSpacing: '0.12em', textTransform: 'uppercase', color: C.text3, marginBottom: 6 }}>{k}</div>
      <div style={{ fontFamily: FONT.mono, fontSize: 40, fontWeight: 700, color: c }}>{v}</div>
    </div>
  );
}

// 14. FORECAST LEADERBOARD
function SlideForecastLeaderboard({ n }) {
  const rows = [
    ['LSTM V1 (fair · autoregressive)',   9.82, 18.91, 0.8689, '2-layer PyTorch · static embeds'],
    ['Chronos-Bolt (cpi_cf_deflated)',    9.39, 19.71, 0.8576, 'foundation model · zero-shot'],
    ['Multivariate TFT V2_13',            9.23, 18.79, 0.8691, '5 obs + 3 known-future covariates'],
    ['LGB Stacker V2_12',                 8.74, 17.69, 0.8852, 'LSTM + Chronos + history feats', true],
  ];
  return (
    <Slide>
      <Eyebrow>13 · Forecast leaderboard</Eyebrow>
      <Title>Stacking beats every single model — <span style={{color:C.primary}}>by 0.016 R²</span>.</Title>

      <div style={{
        border: `1px solid ${C.border}`, borderRadius: 12, overflow: 'hidden',
        background: C.surface, marginTop: 16,
      }}>
        <div style={{
          display: 'grid', gridTemplateColumns: '2.6fr 1fr 1fr 1fr 2.4fr',
          padding: '18px 28px', fontFamily: FONT.mono, fontSize: TYPE.micro,
          letterSpacing: '0.14em', textTransform: 'uppercase', color: C.text3,
          borderBottom: `1px solid ${C.border}`, background: C.surface2,
        }}>
          <div>Model</div>
          <div style={{textAlign:'right'}}>MAE</div>
          <div style={{textAlign:'right'}}>RMSE</div>
          <div style={{textAlign:'right'}}>R²</div>
          <div>Role</div>
        </div>
        {rows.map(([m, mae, rmse, r2, note, best], i) => (
          <div key={i} style={{
            display: 'grid', gridTemplateColumns: '2.6fr 1fr 1fr 1fr 2.4fr',
            padding: '22px 28px', alignItems: 'baseline',
            borderBottom: i < rows.length - 1 ? `1px solid ${C.border}` : 'none',
            background: best ? C.primaryTint : 'transparent',
            fontSize: TYPE.small,
          }}>
            <div style={{ color: best ? C.primary : C.text, fontWeight: best ? 800 : 500 }}>
              {best && <span style={{ marginRight: 10 }}>★</span>}{m}
            </div>
            <Mono color={best ? C.primary : C.text} size={TYPE.small} weight={best ? 700 : 500} style={{ textAlign: 'right', display: 'block' }}>${mae.toFixed(2)}</Mono>
            <Mono color={best ? C.primary : C.text} size={TYPE.small} weight={best ? 700 : 500} style={{ textAlign: 'right', display: 'block' }}>${rmse.toFixed(2)}</Mono>
            <Mono color={best ? C.primary : C.text} size={TYPE.small} weight={700} style={{ textAlign: 'right', display: 'block' }}>{r2.toFixed(4)}</Mono>
            <div style={{ color: C.text2, fontSize: TYPE.tiny }}>
              {best && <span style={{ fontFamily: FONT.mono, fontSize: 14, color: '#fff', background: C.primary, padding: '3px 10px', borderRadius: 99, letterSpacing: '0.1em', marginRight: 10 }}>PRODUCTION</span>}
              {note}
            </div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 28, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
        <div style={{ padding: '18px 24px', background: C.surface2, borderRadius: 10, fontSize: TYPE.tiny, color: C.text2, lineHeight: 1.6 }}>
          <strong style={{color:C.text}}>Signal ceiling ≈ 0.885.</strong>&nbsp;
          At annual resolution with 11 timesteps per group, attention-based architectures (TFT)
          tie the LSTM baseline. The next lever is quarterly ingestion — not a bigger model.
        </div>
        <div style={{ padding: '18px 24px', background: C.surface2, borderRadius: 10, fontSize: TYPE.tiny, color: C.text2, lineHeight: 1.6 }}>
          <strong style={{color:C.text}}>Stacker lift is capped by error diversity.</strong>&nbsp;
          Under fair evaluation, base-model RMSE/MAE ratios sit in 1.9–2.1 — near-Gaussian,
          with limited orthogonal error. Still, +0.016 R² ships.
        </div>
      </div>
      <Footer n={n} total={TOTAL} label="Forecast leaderboard (2022–23 holdout)" />
    </Slide>
  );
}

// 15. STACKER ARCHITECTURE
function SlideStacker({ n }) {
  const feats = [
    ['lstm_pred',                     70.48, 'LSTM autoregressive prediction'],
    ['last_history_value',            15.97, 'persistence anchor (≤ 2021)'],
    ['chronos_pred',                   3.63, 'Chronos cpi_cf_deflated median'],
    ['history_mean',                   2.67, 'mean of history'],
    ['Rndrng_Prvdr_Type_idx',          1.91, 'specialty (categorical)'],
    ['history_cv',                     1.62, 'volatility conditioning'],
    ['history_trend',                  1.18, 'linear trend slope'],
    ['Rndrng_Prvdr_State_Abrvtn_idx',  1.16, 'state (categorical)'],
    ['n_history_years',                0.77, 'context length'],
    ['hcpcs_bucket',                   0.31, 'clinical bucket'],
    ['forecast_year',                  0.27, '2022/2023 flag'],
    ['cpi_factor',                     0.02, 'dead — baked into Chronos'],
    ['cf_factor',                      0.00, 'dead'],
  ];
  return (
    <Slide>
      <Eyebrow>14 · Stacker architecture</Eyebrow>
      <Title>LightGBM learns <span style={{color:C.primary}}>when to trust each base model</span>.</Title>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.3fr', gap: 48, marginTop: 16 }}>
        <div>
          {/* Diagram: two base models → meta → forecast */}
          <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 28 }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                <BaseCard title="LSTM V1" sub="PyTorch · A100 · 5 min retrain" metric="AR rollout from ≤ 2021" />
                <BaseCard title="Chronos-Bolt" sub="foundation model · zero-shot" metric="CPI + CF deflated" />
              </div>
              <ArrowDown />
              <div style={{
                background: C.primaryTint, border: `2px solid ${C.primary}`,
                borderRadius: 10, padding: '18px 22px',
              }}>
                <div style={{ fontFamily: FONT.mono, fontSize: TYPE.micro, letterSpacing: '0.14em', textTransform: 'uppercase', color: C.primary, fontWeight: 700, marginBottom: 6 }}>Meta-learner</div>
                <div style={{ fontSize: TYPE.small, fontWeight: 700, color: C.text }}>LightGBM · 13 features · 1000 rounds</div>
                <div style={{ fontSize: TYPE.tiny, color: C.text2, marginTop: 4 }}>5-fold GroupKFold CV · early stopping</div>
              </div>
              <ArrowDown />
              <div style={{
                background: C.primary, borderRadius: 10, padding: '16px 22px',
                color: '#fff', display: 'flex', alignItems: 'baseline', justifyContent: 'space-between',
              }}>
                <div style={{ fontSize: TYPE.small, fontWeight: 700 }}>2024 – 2026 forecast</div>
                <Mono color="#fff" size={TYPE.small} weight={700}>R² 0.8852</Mono>
              </div>
            </div>
          </div>
          <div style={{ marginTop: 20, fontSize: TYPE.tiny, color: C.text3, lineHeight: 1.5 }}>
            GroupKFold on (ptype, bucket, state) prevents leakage where a group's 2022 observation
            could sit in train and its 2023 in test.
          </div>
        </div>

        <div>
          <div style={{ fontFamily: FONT.mono, fontSize: TYPE.micro, letterSpacing: '0.14em', textTransform: 'uppercase', color: C.text3, marginBottom: 14 }}>
            Stacker feature importance (gain %)
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {feats.map(([name, imp, desc], i) => (
              <div key={i} style={{ display: 'grid', gridTemplateColumns: '240px 1fr 70px', alignItems: 'center', gap: 14 }}>
                <Mono color={imp > 5 ? C.primary : imp > 1 ? C.text2 : C.text3} size={14} weight={600}>{name}</Mono>
                <div style={{ height: 14, background: C.surface2, borderRadius: 3, overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${(imp/75)*100}%`, background: imp > 5 ? C.primary : imp > 1 ? C.secondary : C.muted, borderRadius: 3 }}/>
                </div>
                <Mono color={imp > 5 ? C.primary : C.text2} size={14} weight={700} style={{ textAlign: 'right', display: 'block' }}>{imp.toFixed(2)}%</Mono>
              </div>
            ))}
          </div>
        </div>
      </div>
      <Footer n={n} total={TOTAL} label="LGB Stacker V2_12" />
    </Slide>
  );
}
function BaseCard({ title, sub, metric }) {
  return (
    <div style={{ background: C.surface2, border: `1px solid ${C.border}`, borderRadius: 10, padding: '14px 18px' }}>
      <div style={{ fontSize: TYPE.tiny, fontWeight: 700, color: C.text }}>{title}</div>
      <div style={{ fontSize: TYPE.micro, color: C.text3, marginTop: 2 }}>{sub}</div>
      <Mono color={C.primary} size={13} weight={600} style={{ display: 'block', marginTop: 8 }}>{metric}</Mono>
    </div>
  );
}
function ArrowDown() {
  return (
    <div style={{ display: 'flex', justifyContent: 'center' }}>
      <svg width="20" height="28" viewBox="0 0 20 28"><path d="M10 2 L10 22 M4 16 L10 22 L16 16" stroke={C.text3} strokeWidth="1.8" fill="none" strokeLinecap="round" strokeLinejoin="round"/></svg>
    </div>
  );
}

// 16. THE BUG
function SlideBug({ n }) {
  return (
    <Slide bg={C.surface2}>
      <Eyebrow color={C.error}>15 · Measurement integrity</Eyebrow>
      <Title>A <span style={{color:C.error}}>teacher-forcing bug</span> inflated the LSTM baseline for three phases.</Title>

      <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1fr', gap: 48, marginTop: 16 }}>
        <div>
          <Body size={TYPE.small} style={{ marginBottom: 24 }}>
            The original <Mono color={C.error} size={TYPE.small}>evaluate()</Mono> function fed the true value at each
            position during evaluation — so when predicting 2023, the model saw the TRUE 2022
            value. At inference, it must feed its own predictions back autoregressively.
          </Body>
          <Body size={TYPE.small}>
            Re-running with a fair autoregressive rollout produced a substantially honest baseline.
            All Phase 8 numbers use the corrected metric.
          </Body>
        </div>

        <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 28 }}>
          <div style={{ fontFamily: FONT.mono, fontSize: TYPE.micro, letterSpacing: '0.14em', textTransform: 'uppercase', color: C.text3, marginBottom: 18 }}>
            LSTM V1 baseline · before vs after
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
            <CompareRow k="R²" before="0.886" after="0.8689" delta="−0.017" />
            <CompareRow k="MAE" before="$8.84" after="$9.82" delta="+$0.98" />
            <CompareRow k="RMSE" before="$36.42" after="$18.91" delta="−$17.51" bigdelta />
          </div>
          <div style={{ marginTop: 22, padding: '14px 18px', background: C.accentTint, borderLeft: `3px solid ${C.accent}`, borderRadius: 8, fontSize: TYPE.micro, color: C.text2, lineHeight: 1.6 }}>
            <strong style={{color:C.text}}>The RMSE swing is load-bearing.</strong> The original number
            made it look like Chronos had half the RMSE of LSTM — an artifact, not a real property.
            Under fair eval, all base models have RMSE/MAE ratios near 2.0.
          </div>
        </div>
      </div>
      <Footer n={n} total={TOTAL} label="The teacher-forcing bug" />
    </Slide>
  );
}
function CompareRow({ k, before, after, delta, bigdelta }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '80px 1fr 1fr 1fr', alignItems: 'baseline', gap: 12 }}>
      <Mono color={C.text3} size={TYPE.tiny} weight={700}>{k}</Mono>
      <div>
        <div style={{ fontFamily: FONT.mono, fontSize: TYPE.micro, letterSpacing: '0.12em', color: C.text3, textTransform: 'uppercase', marginBottom: 4 }}>teacher-forced</div>
        <Mono color={C.text2} size={TYPE.subtitle} weight={700} style={{ textDecoration: 'line-through', textDecorationColor: C.text3 }}>{before}</Mono>
      </div>
      <div>
        <div style={{ fontFamily: FONT.mono, fontSize: TYPE.micro, letterSpacing: '0.12em', color: C.secondary, textTransform: 'uppercase', marginBottom: 4 }}>autoregressive (fair)</div>
        <Mono color={C.secondary} size={TYPE.subtitle} weight={700}>{after}</Mono>
      </div>
      <Mono color={bigdelta ? C.error : C.text2} size={TYPE.small} weight={700}>{delta}</Mono>
    </div>
  );
}

// 17. NEGATIVE RESULTS
function SlideNegatives({ n }) {
  const items = [
    ['Charge-ratio derived series',  'R² 0.1937', 'divide target by submitted charge · back-transform amplifies error', C.error],
    ['Volume-normalized series',     'R² −540.24', 'catastrophic — multiplicative error compounding', C.error],
    ['CPI-only deflation',           'R² 0.8473', 'worse than raw · CPI over-corrects vs. fee schedule', C.accent],
    ['Sequestration reversal',       'R² 0.8528', 'uniform ~2% reversal adds noise, exposes no signal', C.accent],
    ['Risk-adjusted derived series', 'R² 0.8563', 'identical to raw · risk score imputed ~1.0', C.text3],
    ['Multivariate TFT',             'R² 0.8691', 'ties LSTM · covariates move in lockstep at annual res', C.accent],
    ['Univariate TFT',               'R² 0.846',  'starved input · architecture wasn\'t the problem', C.text3],
    ['Hierarchical reconciliation',  'no-op',      'bottom-up forecasts already coherent',         C.text3],
  ];
  return (
    <Slide>
      <Eyebrow>16 · Negative results</Eyebrow>
      <Title>Eight things that didn't work — <span style={{color:C.primary}}>and why each failure was informative</span>.</Title>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
        {items.map(([title, metric, why, col], i) => (
          <div key={i} style={{
            background: C.surface, border: `1px solid ${C.border}`, borderLeft: `3px solid ${col}`,
            borderRadius: 8, padding: '18px 22px',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
              <div style={{ fontSize: TYPE.small, fontWeight: 700, color: C.text }}>{title}</div>
              <Mono color={col} size={TYPE.tiny} weight={700}>{metric}</Mono>
            </div>
            <div style={{ fontSize: TYPE.micro, color: C.text2, lineHeight: 1.5 }}>{why}</div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 24, padding: '18px 24px', background: C.primaryTint, borderLeft: `3px solid ${C.primary}`, borderRadius: 8, fontSize: TYPE.tiny, color: C.text2, lineHeight: 1.6 }}>
        <strong style={{color:C.primary}}>The pattern:</strong>&nbsp;
        derived-target tricks and clever decompositions consistently lose at annual resolution.
        The base target series already carries most of the signal the data supports.
      </div>
      <Footer n={n} total={TOTAL} label="Negative results" />
    </Slide>
  );
}

// 18. PRODUCT / APP
function SlideProduct({ n }) {
  return (
    <Slide bg={C.surface}>
      <Eyebrow>17 · Product</Eyebrow>
      <Title>The models ship behind <span style={{color:C.primary}}>three pages</span>.</Title>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 24, marginTop: 16 }}>
        {/* Page 1: Estimator */}
        <PageCard
          badge="Primary"
          badgeBg={C.primary}
          route="/"
          title="Cost Estimator"
          body="Homepage IS the tool. Stage 1 form on the left, Stage 2 results on the right."
          preview={<EstimatorMock />}
        />
        {/* Page 2: Forecast */}
        <PageCard
          badge="Secondary"
          badgeBg={C.secondary}
          route="/forecast"
          title="Forecast Explorer"
          body="LSTM-powered specialty-level projections 2024–2026 with P10 / P90 bounds."
          preview={<ForecastMock />}
        />
        {/* Page 3: About */}
        <PageCard
          badge="Research"
          badgeBg={C.accent}
          route="/about"
          title="About & Methodology"
          body="Tabbed methodology: data, models, pipeline. This deck sources from here."
          preview={<AboutMock />}
        />
      </div>

      <div style={{ marginTop: 32, padding: '18px 28px', background: C.surface2, borderRadius: 10, display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', fontSize: TYPE.tiny, color: C.text2 }}>
        <div><strong style={{color:C.text}}>Stack:</strong> Next.js 16 · MUI v9 · Recharts 3.8 · Supabase · Railway · Vercel</div>
        <Mono color={C.primary} size={TYPE.tiny}>allowancemap.vercel.app</Mono>
      </div>
      <Footer n={n} total={TOTAL} label="The AllowanceMap web app" />
    </Slide>
  );
}

function PageCard({ badge, badgeBg, route, title, body, preview }) {
  return (
    <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <div style={{ background: badgeBg, padding: '10px 18px' }}>
        <span style={{ fontFamily: FONT.mono, fontSize: 13, fontWeight: 700, color: 'rgba(255,255,255,0.85)', letterSpacing: '0.12em', textTransform: 'uppercase' }}>{badge}</span>
      </div>
      <div style={{ padding: '16px 20px 8px', display: 'flex', flexDirection: 'column', gap: 6 }}>
        <Mono color={C.primary} size={TYPE.tiny}>{route}</Mono>
        <div style={{ fontSize: TYPE.small, fontWeight: 700, color: C.text }}>{title}</div>
        <div style={{ fontSize: TYPE.micro, color: C.text2, lineHeight: 1.5, marginBottom: 8 }}>{body}</div>
      </div>
      <div style={{ flex: 1, background: C.surface2, padding: 14, borderTop: `1px solid ${C.border}` }}>
        {preview}
      </div>
    </div>
  );
}

function MiniBar({ bg = '#fff', h = 16, w = '100%', r = 4, style = {} }) {
  return <div style={{ background: bg, height: h, width: w, borderRadius: r, ...style }} />;
}

function EstimatorMock() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <Mono color={C.text3} size={11} style={{ letterSpacing: '0.14em', textTransform: 'uppercase' }}>Stage 1 · inputs</Mono>
      <MiniBar h={26} bg="#fff" style={{ border: `1px solid ${C.border2}` }}/>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        <MiniBar h={26} bg="#fff" style={{ border: `1px solid ${C.border2}` }}/>
        <MiniBar h={26} bg="#fff" style={{ border: `1px solid ${C.border2}` }}/>
      </div>
      <MiniBar h={30} bg={C.primary} style={{ marginTop: 6 }}/>
      <div style={{ marginTop: 10, padding: 10, background: '#fff', borderLeft: `3px solid ${C.primary}`, borderRadius: 4 }}>
        <Mono color={C.primary} size={11} style={{ letterSpacing: '0.14em', textTransform: 'uppercase' }}>Stage 1 result</Mono>
        <Mono color={C.primary} size={28} weight={700} style={{ display: 'block', marginTop: 3 }}>$1,247</Mono>
        <div style={{ fontSize: 10, color: C.text3 }}>median allowed</div>
      </div>
    </div>
  );
}
function ForecastMock() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <Mono color={C.text3} size={11} style={{ letterSpacing: '0.14em', textTransform: 'uppercase' }}>Specialty selector</Mono>
      <MiniBar h={26} bg="#fff" style={{ border: `1px solid ${C.border2}` }}/>
      <svg viewBox="0 0 260 120" style={{ background: '#fff', borderRadius: 4, padding: 6, border: `1px solid ${C.border}`, marginTop: 6 }}>
        <path d="M10 80 L40 70 L70 60 L100 55 L130 48 L160 45 L190 42 L220 38 L250 35"
              fill="none" stroke={C.primary} strokeWidth="2"/>
        <path d="M160 45 L190 42 L220 38 L250 35"
              fill="none" stroke={C.primary} strokeWidth="2" strokeDasharray="4 4"/>
        <path d="M160 30 L190 24 L220 18 L250 16 L250 48 L220 50 L190 54 L160 60 Z"
              fill={C.primary} opacity="0.12"/>
      </svg>
    </div>
  );
}
function AboutMock() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', gap: 4 }}>
        {['Overview','Data','Models','Pipeline'].map((t, i) => (
          <div key={i} style={{ fontSize: 9, padding: '3px 7px', background: i === 2 ? C.primary : '#fff', color: i === 2 ? '#fff' : C.text2, border: `1px solid ${C.border}`, borderRadius: 3, fontFamily: FONT.mono, letterSpacing: '0.08em', textTransform: 'uppercase' }}>{t}</div>
        ))}
      </div>
      <div style={{ background: '#fff', borderRadius: 4, padding: 8, border: `1px solid ${C.border}`, marginTop: 4 }}>
        <Mono color={C.text3} size={9} style={{ letterSpacing: '0.14em', textTransform: 'uppercase' }}>Feature importance</Mono>
        {[0.31, 0.22, 0.15, 0.08, 0.06].map((v, i) => (
          <div key={i} style={{ display: 'grid', gridTemplateColumns: '40px 1fr', gap: 4, alignItems: 'center', marginTop: 4 }}>
            <div style={{ background: C.surface2, height: 6, borderRadius: 2 }}/>
            <div style={{ background: C.primary, height: 6, width: `${v*100*2}%`, borderRadius: 2 }}/>
          </div>
        ))}
      </div>
    </div>
  );
}

// 19. ROADMAP
function SlideRoadmap({ n }) {
  return (
    <Slide>
      <Eyebrow>18 · What's next</Eyebrow>
      <Title>Three levers, ranked by <span style={{color:C.primary}}>expected lift vs. effort</span>.</Title>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 18, marginTop: 16 }}>
        {[
          {
            pri: 'HIGH',
            priCol: C.primary,
            title: 'Integrate Medicare Physician Fee Schedule (MPFS)',
            lift: 'Replace charge with RVU × GPCI × CF',
            effort: 'Medium · free · 11 CSV joins',
            body: 'Transforms the model from "learn the fee schedule" into "learn deviations from the fee schedule." Publishable contribution; charge dependency eliminated.',
          },
          {
            pri: 'HIGH',
            priCol: C.primary,
            title: 'Quarterly data ingestion',
            lift: 'Forecast R² 0.89 → 0.91–0.93',
            effort: 'High · 2–4 weeks of data eng',
            body: 'Only known lever above the 0.885 signal ceiling. 44 timesteps/group instead of 11 unlocks CNN / TCN / PatchTST and meaningful Chronos fine-tuning.',
          },
          {
            pri: 'MED',
            priCol: C.secondary,
            title: 'Real MCBS Limited Data Set for Stage 2',
            lift: 'Revives monotonic constraints',
            effort: 'Medium · $600/module · 6–8wk DUA',
            body: 'Replaces synthetic OOP with actual patient-level linkage. V2 CatBoost-monotonic likely becomes competitive because real distributions obey the constraints synthetic data violates.',
          },
          {
            pri: 'MED',
            priCol: C.secondary,
            title: 'Canonicalize specialty names in the silver layer',
            lift: 'Fixes Cardiology, Colorectal, Oral Surgery splits',
            effort: 'Low · 1 day of compute',
            body: 'CMS renames specialties year-to-year; LabelEncoder currently creates duplicate indices with disjoint coverage. Affects forecasts for ~3 confirmed specialties + unaudited more.',
          },
          {
            pri: 'LOW',
            priCol: C.text3,
            title: 'Quantile stacker variant for real P10/P50/P90 bounds',
            lift: 'Genuine uncertainty UI',
            effort: 'Low · 2 hours · 20-line edit',
            body: 'Currently the stacker is point-only; bounds collapse to the mean. Three LightGBM boosters at α={0.1, 0.5, 0.9} deliver real quantile forecasts.',
          },
        ].map((r, i) => (
          <div key={i} style={{
            display: 'grid', gridTemplateColumns: '70px 2fr 1.5fr 3fr',
            alignItems: 'center', gap: 24, padding: '18px 22px',
            background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10,
          }}>
            <span style={{
              fontFamily: FONT.mono, fontSize: 13, fontWeight: 700, color: '#fff',
              background: r.priCol, padding: '4px 10px', borderRadius: 99,
              letterSpacing: '0.12em', textAlign: 'center',
            }}>{r.pri}</span>
            <div>
              <div style={{ fontSize: TYPE.small, fontWeight: 700, color: C.text }}>{r.title}</div>
              <Mono color={r.priCol} size={13} style={{ display: 'block', marginTop: 4 }}>{r.lift}</Mono>
            </div>
            <Mono color={C.text3} size={13}>{r.effort}</Mono>
            <div style={{ fontSize: TYPE.micro, color: C.text2, lineHeight: 1.5 }}>{r.body}</div>
          </div>
        ))}
      </div>
      <Footer n={n} total={TOTAL} label="Roadmap" />
    </Slide>
  );
}

// ─────────────────────────────────────────────────────────
function Deck() {
  return (
    <>
      <SlideCover />
      <SlideProblem n={2} />
      <SlideObjective n={3} />
      <SlideDataset n={4} />
      <SlidePipeline n={5} />
      <SlideStage1Intro n={6} />
      <SlideStage1Features n={7} />
      <SlideStage1Leaderboard n={8} />
      <SlideNoCharge n={9} />
      <SlideFeatureImportance n={10} />
      <SlideStage2 n={11} />
      <SlideStage2Results n={12} />
      <SlideForecast n={13} />
      <SlideForecastLeaderboard n={14} />
      <SlideStacker n={15} />
      <SlideBug n={16} />
      <SlideNegatives n={17} />
      <SlideProduct n={18} />
      <SlideRoadmap n={19} />
    </>
  );
}

window.Deck = Deck;
