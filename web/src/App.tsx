import { useCallback, useEffect, useRef, useState } from 'react'
import { AlertCircle, ExternalLink, RotateCcw } from 'lucide-react'
import { api, ApiError, API_URL, DASHBOARD_URL, REPO_URL } from './api'
import type { Metrics, Prediction, SampleIndex, Tier } from './api'
import { InputPanel } from './components/InputPanel'
import { Performance } from './components/Performance'
import { Results } from './components/Results'
import { Section, Skeleton } from './components/ui'

const TIERS: Tier[] = [
  { index: 0, name: 'Healthy', lower: 0, upper: 0.15, action: '', urgency: '' },
  { index: 1, name: 'Minor', lower: 0.15, upper: 0.4, action: '', urgency: '' },
  { index: 2, name: 'Moderate', lower: 0.4, upper: 0.6, action: '', urgency: '' },
  { index: 3, name: 'Severe', lower: 0.6, upper: 0.8, action: '', urgency: '' },
  { index: 4, name: 'Critical', lower: 0.8, upper: 1.0001, action: '', urgency: '' },
]

type Last = { kind: 'sample'; id: string } | { kind: 'file'; file: File; fs?: number }

export default function App() {
  const [samples, setSamples] = useState<SampleIndex | null>(null)
  const [samplesError, setSamplesError] = useState<string | null>(null)
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [metricsError, setMetricsError] = useState<string | null>(null)
  const [result, setResult] = useState<Prediction | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const last = useRef<Last | null>(null)
  const resultsRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api.samples().then(setSamples, (e: ApiError) => setSamplesError(e.message))
    api.metrics().then(setMetrics, (e: ApiError) => setMetricsError(e.message))
  }, [])

  const run = useCallback(async (job: Last) => {
    last.current = job
    setBusy(true)
    setError(null)
    try {
      const r = job.kind === 'sample' ? await api.predictSample(job.id) : await api.predictFile(job.file, job.fs)
      setResult(r)
      requestAnimationFrame(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong.')
    } finally {
      setBusy(false)
    }
  }, [])

  return (
    <div className="min-h-dvh">
      <a href="#diagnose" className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded focus:bg-surface focus:px-3 focus:py-2">
        Skip to the diagnosis tool
      </a>
      <header className="sticky top-0 z-40 border-b border-line bg-surface/95 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <a href="#top" className="flex items-center gap-2 font-bold text-ink">
            <img src="/favicon.svg" alt="" width={24} height={24} />
            <span className="hidden min-[400px]:inline">VibraDiagnose</span>
          </a>
          <nav aria-label="Main" className="flex items-center gap-3 text-[0.85rem] font-semibold text-muted sm:gap-6 sm:text-sm">
            <a href="#diagnose" className="hover:text-ink">
              Diagnose
            </a>
            <a href="#performance" className="hover:text-ink">
              Results
            </a>
            <a href="#method" className="hidden hover:text-ink sm:inline">
              Method
            </a>
            <a href={DASHBOARD_URL} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 hover:text-ink">
              Dashboard <ExternalLink size={13} aria-hidden="true" className="hidden sm:block" />
            </a>
          </nav>
        </div>
      </header>

      <main id="top" className="mx-auto max-w-6xl px-4 pb-24 sm:px-6">
        <div className="pt-12 sm:pt-16">
          <div className="text-sm font-semibold text-muted">Z24 Bridge structural health monitoring</div>
          <h1 className="mt-3 text-4xl font-bold leading-[1.05] tracking-tight text-ink sm:text-6xl">Diagnose a bridge recording</h1>
          <p className="mt-4 max-w-3xl text-lg text-ink-2 sm:text-xl">
            Identify which of the 17 Z24 damage scenarios a vibration recording comes from, then estimate its damage severity, a model-based remaining useful life and the recommended maintenance action.
          </p>
        </div>

        <Section id="diagnose" title="Input" sub="Pick a recording from the held-out test setups, or upload your own accelerometer file.">
          <InputPanel samples={samples} samplesError={samplesError} busy={busy} onSample={(id) => run({ kind: 'sample', id })} onFile={(file, fs) => run({ kind: 'file', file, fs })} />
          {error && (
            <div role="alert" className="mt-4 flex flex-col gap-3 rounded-lg border border-[#efd3c8] bg-warn-soft px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-start gap-2 text-ink">
                <AlertCircle size={18} className="mt-0.5 shrink-0 text-warn" aria-hidden="true" />
                <span>{error}</span>
              </div>
              {last.current && (
                <button type="button" onClick={() => last.current && run(last.current)} className="inline-flex min-h-11 cursor-pointer items-center gap-2 self-start rounded-md border border-line bg-surface px-4 text-sm font-semibold text-ink-2 hover:border-accent hover:text-accent sm:self-auto">
                  <RotateCcw size={15} aria-hidden="true" /> Try again
                </button>
              )}
            </div>
          )}
        </Section>

        <div ref={resultsRef} className="scroll-mt-20">
          {busy && !result && (
            <div className="mt-10 space-y-4" aria-label="Loading results">
              <Skeleton className="h-40" />
              <div className="grid gap-4 lg:grid-cols-3">
                <Skeleton className="h-56" />
                <Skeleton className="h-56" />
                <Skeleton className="h-56" />
              </div>
            </div>
          )}
          {result && (
            <Section title="Diagnosis" sub={busy ? 'Updating...' : undefined}>
              <div className={busy ? 'opacity-50 transition-opacity duration-200' : 'transition-opacity duration-200'}>
                <Results r={result} tiers={TIERS} />
              </div>
            </Section>
          )}
        </div>

        <Section id="performance" title="How accurate is it" sub="Final results of every model on recordings from sensor setups never seen during training or tuning.">
          <Performance metrics={metrics} error={metricsError} />
        </Section>

        <Section id="method" title="Method">
          <div className="grid gap-8 md:grid-cols-3">
            {[
              ['Signal', 'One fixed reference accelerometer (Z24 channel 22) or the five reference sensors 22 to 26, band-passed 0.5 to 30 Hz and cut into 20 s windows with 50% overlap, each z-scored.'],
              ['Models', 'WaveNet, InceptionTime and MiniRocket, combined by a logistic-regression stacking ensemble fitted on out-of-fold predictions from the training setups. A separate hierarchical classifier predicts the damage group, then the scenario.'],
              ['Severity and RUL', 'Each scenario has a defined damage severity index from 0 to 1. Class probabilities become a DSI distribution, and a power-law degradation model turns it into a remaining-useful-life distribution by Monte Carlo.'],
            ].map(([t, d]) => (
              <div key={t} className="border-t border-line pt-4">
                <div className="text-lg font-bold text-ink">{t}</div>
                <p className="mt-2 text-base text-muted">{d}</p>
              </div>
            ))}
          </div>
          <div className="mt-8 rounded-lg border border-line bg-surface-2 px-5 py-4 text-sm text-ink-2">
            <b className="text-ink">Limitations.</b> The Z24 bridge has no ground-truth lifespan data, so the remaining useful life is a prognostic estimate from the assumed degradation model, not a validated measurement. The classifier was trained on one bridge with 9 sensor setups; neighbouring damage stages (for example 2 versus 4 ruptured tendons) are often confused, and the first reference state is frequently mistaken for a late damage state, most likely because of temperature effects.
          </div>
        </Section>
      </main>

      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-6xl flex-col gap-2 px-4 py-6 text-sm text-muted sm:flex-row sm:justify-between sm:px-6">
          <span>VibraDiagnose · KU Leuven Z24 Progressive Damage Tests</span>
          <span className="flex gap-4">
            <a href={REPO_URL} target="_blank" rel="noreferrer" className="hover:text-ink">
              Source code
            </a>
            <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer" className="hover:text-ink">
              API docs
            </a>
            <a href={DASHBOARD_URL} target="_blank" rel="noreferrer" className="hover:text-ink">
              Project dashboard
            </a>
          </span>
        </div>
      </footer>
    </div>
  )
}
