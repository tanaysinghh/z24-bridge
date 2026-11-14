import { AlertTriangle, ArrowRight, CheckCircle2, Info, XCircle } from 'lucide-react'
import type { Prediction, Tier } from '../api'
import { pct } from '../api'
import { DsiScale, Histogram, LineChart } from './charts'
import { Bar, Big, Card, Label, Note, Pill } from './ui'

const TIER_COLORS = ['text-tier-0', 'text-tier-1', 'text-tier-2', 'text-tier-3', 'text-tier-4']
const GROUP_ORDER = ['Healthy', 'Settlement', 'Foundation', 'Hinge', 'Tendon']

function Verdict({ ok, yes, no }: { ok: boolean; yes: string; no: string }) {
  return ok ? (
    <Pill tone="good">
      <CheckCircle2 size={13} aria-hidden="true" />
      {yes}
    </Pill>
  ) : (
    <Pill tone="warn">
      <XCircle size={13} aria-hidden="true" />
      {no}
    </Pill>
  )
}

function Hero({ r }: { r: Prediction }) {
  const p = r.prediction
  const t = r.truth
  return (
    <Card accent className="grid gap-6 lg:grid-cols-[1.6fr_1fr_1fr]">
      <div>
        <Label>Predicted scenario · {p.method}</Label>
        <div className="mt-2 text-3xl font-bold leading-tight tracking-tight text-ink sm:text-4xl">{p.scenario}</div>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <Pill tone="accent">{p.group} group</Pill>
          <Pill>Scenario {p.label}</Pill>
          {t && <Verdict ok={t.label === p.label} yes="Matches ground truth" no={`Ground truth: ${t.scenario}`} />}
        </div>
        {t && (
          <Note>
            Built-in test recording, setup {t.setup}. True scenario {t.label} ({t.scenario}).
          </Note>
        )}
      </div>
      <div>
        <Label>Confidence</Label>
        <Big tone="accent" size="xl">
          {pct(p.probability, 0)}
        </Big>
        <Note>Ensemble probability over {r.input.n_windows} windows of 20 s</Note>
      </div>
      <div>
        <Label>Model agreement</Label>
        <Big size="xl">
          {r.agreement.models_agreeing}
          <span className="text-faint">/{r.agreement.models_total}</span>
        </Big>
        <Note>base models pick the same scenario{r.agreement.hierarchy_agrees ? '; the hierarchical classifier agrees' : '; the hierarchical classifier differs'}</Note>
      </div>
    </Card>
  )
}

function Severity({ r, tiers }: { r: Prediction; tiers: Tier[] }) {
  const s = r.severity
  const tier = s.tier
  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <Card>
        <Label>Damage severity index</Label>
        <Big>{s.dsi.expected.toFixed(2)}</Big>
        <Note>
          90% interval {s.dsi.lower.toFixed(2)} to {s.dsi.upper.toFixed(2)} on a 0 (intact) to 1 (most severe Z24 state) scale
        </Note>
        <div className="mt-4">
          <DsiScale value={s.dsi.expected} lower={s.dsi.lower} upper={s.dsi.upper} tiers={tiers} />
        </div>
      </Card>
      <Card>
        <Label>Remaining useful life, model-based</Label>
        <Big>
          {r.rul.median.toFixed(0)}
          <span className="ml-1 text-2xl font-semibold text-muted">years</span>
        </Big>
        <Note>
          {Math.round(r.rul.interval * 100)}% interval {r.rul.lower.toFixed(0)} to {r.rul.upper.toFixed(0)} years (Monte Carlo, {r.rul.params.n_samples.toLocaleString()} draws)
        </Note>
        <div className="mt-4">
          <Histogram edges={r.rul.histogram.edges} counts={r.rul.histogram.counts} lower={r.rul.lower} upper={r.rul.upper} median={r.rul.median} />
        </div>
      </Card>
      <Card>
        <Label>Recommended action</Label>
        <div className={`mt-2 text-3xl font-bold tracking-tight sm:text-4xl ${TIER_COLORS[tier.index]}`}>{tier.name}</div>
        <div className="mt-1 text-sm font-semibold text-ink-2">{tier.urgency}</div>
        <p className="mt-3 text-base text-ink-2">{tier.action}</p>
        <div className="mt-4 space-y-1.5">
          {Object.entries(s.tier_probabilities).map(([name, v]) => (
            <div key={name} className="grid grid-cols-[5.5rem_1fr_3rem] items-center gap-2 text-sm">
              <span className="text-muted">{name}</span>
              <Bar value={v} tone={name === tier.name ? 'accent' : 'muted'} label={`${name} tier probability`} />
              <span className="tnum text-right text-ink-2">{pct(v, 0)}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}

function Hierarchy({ r }: { r: Prediction }) {
  const h = r.hierarchy
  const groups = GROUP_ORDER.map((g) => [g, h.group.probabilities[g] ?? 0] as const)
  return (
    <Card>
      <div className="flex flex-wrap items-center gap-3">
        <Label>Hierarchical classifier</Label>
        <Verdict ok={r.agreement.hierarchy_agrees} yes="Agrees with the ensemble" no="Differs from the ensemble" />
      </div>
      <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center">
        <div>
          <div className="text-sm text-muted">Stage 1 · damage group</div>
          <div className="text-2xl font-bold text-ink">{h.group.name}</div>
          <div className="tnum text-sm text-muted">{pct(h.group.probability)}</div>
        </div>
        <ArrowRight className="hidden text-faint sm:block" aria-hidden="true" />
        <div>
          <div className="text-sm text-muted">Stage 2 · scenario within {h.group.name}</div>
          <div className="text-2xl font-bold text-ink">{h.scenario.name}</div>
          <div className="tnum text-sm text-muted">{pct(h.scenario.probability_within_group)} within the group</div>
        </div>
      </div>
      <p className="mt-2 text-sm text-muted">{h.group.description}.</p>
      <div className="mt-5 grid gap-6 md:grid-cols-2">
        <div className="space-y-2">
          <div className="text-sm font-semibold text-ink-2">Group probabilities</div>
          {groups.map(([g, v]) => (
            <div key={g} className="grid grid-cols-[6.5rem_1fr_3.5rem] items-center gap-2 text-sm">
              <span className={g === h.group.name ? 'font-semibold text-ink' : 'text-muted'}>{g}</span>
              <Bar value={v} tone={g === h.group.name ? 'accent' : 'muted'} label={`${g} group probability`} />
              <span className="tnum text-right text-ink-2">{pct(v)}</span>
            </div>
          ))}
        </div>
        <div className="space-y-2">
          <div className="text-sm font-semibold text-ink-2">Specialist output</div>
          {h.scenario.within_group.map((s) => (
            <div key={s.label} className="grid grid-cols-[minmax(0,10rem)_1fr_3.5rem] items-center gap-2 text-sm">
              <span className={`truncate ${s.label === h.scenario.label ? 'font-semibold text-ink' : 'text-muted'}`} title={s.name}>
                {s.name}
              </span>
              <Bar value={s.probability} tone={s.label === h.scenario.label ? 'accent' : 'muted'} label={`${s.name} probability`} />
              <span className="tnum text-right text-ink-2">{pct(s.probability)}</span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  )
}

function Agreement({ r }: { r: Prediction }) {
  return (
    <Card>
      <Label>Which models agreed</Label>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[34rem] text-left text-[0.97rem]">
          <thead>
            <tr className="border-b border-line text-xs font-semibold text-muted">
              <th className="py-2 pr-3 font-semibold">Model</th>
              <th className="py-2 pr-3 font-semibold">Predicts</th>
              <th className="py-2 pr-3 text-right font-semibold">Its confidence</th>
              <th className="py-2 pr-3 text-right font-semibold">P(ensemble class)</th>
              <th className="py-2 font-semibold">Verdict</th>
            </tr>
          </thead>
          <tbody>
            {r.agreement.models.map((m) => (
              <tr key={m.key} className="border-b border-line last:border-0">
                <td className="py-3 pr-3 font-semibold text-ink">{m.name}</td>
                <td className="py-3 pr-3 text-ink-2">{m.scenario}</td>
                <td className="tnum py-3 pr-3 text-right text-ink-2">{pct(m.probability)}</td>
                <td className="tnum py-3 pr-3 text-right text-ink-2">{pct(m.probability_of_ensemble_class)}</td>
                <td className="py-3">
                  {m.agrees ? (
                    <Verdict ok yes="Agrees" no="" />
                  ) : m.agrees_group ? (
                    <Pill>Same group, other stage</Pill>
                  ) : (
                    <Verdict ok={false} yes="" no="Disagrees" />
                  )}
                </td>
              </tr>
            ))}
            <tr>
              <td className="py-3 pr-3 font-semibold text-ink">Hierarchical</td>
              <td className="py-3 pr-3 text-ink-2">{r.hierarchy.scenario.name}</td>
              <td className="tnum py-3 pr-3 text-right text-ink-2">{pct(r.hierarchy.group.probability * r.hierarchy.scenario.probability_within_group)}</td>
              <td className="py-3 pr-3 text-right text-faint">n/a</td>
              <td className="py-3">
                {r.agreement.hierarchy_agrees ? <Verdict ok yes="Agrees" no="" /> : r.agreement.hierarchy_group_agrees ? <Pill>Same group, other stage</Pill> : <Verdict ok={false} yes="" no="Disagrees" />}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function Probabilities({ r }: { r: Prediction }) {
  const top = r.prediction.label
  return (
    <Card>
      <Label>Ensemble probability for every scenario</Label>
      <div className="mt-4 grid gap-x-10 gap-y-5 md:grid-cols-2">
        {GROUP_ORDER.map((g) => (
          <div key={g}>
            <div className="mb-1.5 text-sm font-semibold text-ink-2">{g}</div>
            <div className="space-y-1.5">
              {r.prediction.probabilities
                .filter((c) => c.group === g)
                .map((c) => (
                  <div key={c.label} className="grid grid-cols-[minmax(0,12rem)_1fr_3.5rem] items-center gap-2 text-sm">
                    <span className={`truncate ${c.label === top ? 'font-semibold text-ink' : 'text-muted'}`} title={c.name}>
                      <span className="tnum mr-1.5 text-faint">{String(c.label).padStart(2, '0')}</span>
                      {c.name}
                    </span>
                    <Bar value={c.probability} tone={c.label === top ? 'accent' : 'muted'} label={`${c.name} probability`} />
                    <span className="tnum text-right text-ink-2">{pct(c.probability)}</span>
                  </div>
                ))}
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}

function Signal({ r }: { r: Prediction }) {
  const mode = r.spectrum.first_mode_hz
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card>
        <Label>Band-passed signal · sensor 22</Label>
        <div className="mt-3">
          <LineChart x={r.signal.t} y={r.signal.y} xLabel="Time (s)" yLabel="Acceleration (raw units)" ariaLabel="Band-passed acceleration signal over time" />
        </div>
        <Note>0.5 to 30 Hz Butterworth band-pass, the same filter used in training.</Note>
      </Card>
      <Card>
        <Label>Power spectrum</Label>
        <div className="mt-3">
          <LineChart
            x={r.spectrum.f}
            y={r.spectrum.db}
            xLabel="Frequency (Hz)"
            yLabel="Power (dB rel. peak)"
            ariaLabel="Welch power spectrum of the signal"
            marker={mode ? { x: mode, label: `${mode.toFixed(2)} Hz` } : null}
            xTicks={[0, 5, 10, 15, 20, 25, 30]}
            fill
          />
        </div>
        <Note>Marker: strongest peak between 3 and 4.6 Hz, the first vertical bending mode (about 3.9 Hz on the intact bridge).</Note>
      </Card>
    </div>
  )
}

function Windows({ r }: { r: Prediction }) {
  return (
    <Card>
      <Label>Window-by-window decision</Label>
      <div className="mt-3 flex flex-wrap gap-2">
        {r.windows.map((w) => (
          <div key={w.start_s} className={`rounded-md border px-3 py-2 text-sm ${w.label === r.prediction.label ? 'border-[#c9dbea] bg-accent-soft' : 'border-line bg-surface-2'}`}>
            <div className="tnum text-xs text-muted">
              {w.start_s.toFixed(0)}-{(w.start_s + 20).toFixed(0)} s
            </div>
            <div className="font-semibold text-ink">{w.scenario}</div>
            <div className="tnum text-xs text-muted">{pct(w.probability, 0)}</div>
          </div>
        ))}
      </div>
      <Note>
        Input: {r.input.source ?? 'upload'} · {r.input.n_channels} channel{r.input.n_channels > 1 ? 's' : ''} · {r.input.duration_s.toFixed(0)} s at {r.input.fs_input} Hz · computed in {r.elapsed_ms.toFixed(0)} ms
      </Note>
      {r.input.warnings.length > 0 && (
        <ul className="mt-2 space-y-1 text-sm text-ink-2">
          {r.input.warnings.map((w) => (
            <li key={w} className="flex items-start gap-1.5">
              <Info size={15} className="mt-0.5 shrink-0 text-accent" aria-hidden="true" />
              {w}
            </li>
          ))}
        </ul>
      )}
    </Card>
  )
}

export function Results({ r, tiers }: { r: Prediction; tiers: Tier[] }) {
  return (
    <div className="space-y-4" aria-live="polite">
      <Hero r={r} />
      <Severity r={r} tiers={tiers} />
      <div className="flex items-start gap-2 rounded-lg border border-[#efd3c8] bg-warn-soft px-4 py-3 text-sm text-ink-2">
        <AlertTriangle size={17} className="mt-0.5 shrink-0 text-warn" aria-hidden="true" />
        <span>{r.rul.disclaimer}</span>
      </div>
      <Hierarchy r={r} />
      <Agreement r={r} />
      <Probabilities r={r} />
      <Signal r={r} />
      <Windows r={r} />
    </div>
  )
}
