import { useRef, useState } from 'react'
import type { ChangeEvent, DragEvent, FormEvent } from 'react'
import { Database, FileUp, Loader2, Play, Upload } from 'lucide-react'
import type { SampleIndex } from '../api'

const MAX_BYTES = 10 * 1024 * 1024
const ACCEPT = ['.npy', '.csv', '.txt', '.mat']

type Props = {
  samples: SampleIndex | null
  samplesError: string | null
  busy: boolean
  onSample: (id: string) => void
  onFile: (file: File, fs?: number) => void
}

export function InputPanel({ samples, samplesError, busy, onSample, onFile }: Props) {
  const [mode, setMode] = useState<'sample' | 'upload'>('sample')
  const [sampleId, setSampleId] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [fs, setFs] = useState('100')
  const [fileError, setFileError] = useState<string | null>(null)
  const [drag, setDrag] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const selected = sampleId || samples?.samples[0]?.id || ''

  const pick = (f: File | null) => {
    setFileError(null)
    if (!f) return setFile(null)
    const name = f.name.toLowerCase()
    if (!ACCEPT.some((ext) => name.endsWith(ext))) {
      setFile(null)
      return setFileError(`"${f.name}" is not a supported file. Use .npy, .csv or .mat.`)
    }
    if (f.size > MAX_BYTES) {
      setFile(null)
      return setFileError(`"${f.name}" is ${(f.size / 1e6).toFixed(1)} MB. The limit is 10 MB.`)
    }
    setFile(f)
  }

  const submitUpload = (e: FormEvent) => {
    e.preventDefault()
    if (!file) return setFileError('Choose a file first.')
    const rate = fs.trim() === '' ? undefined : Number(fs)
    if (rate !== undefined && (!Number.isFinite(rate) || rate < 50 || rate > 2000)) return setFileError('Sampling rate must be a number between 50 and 2000 Hz.')
    onFile(file, rate)
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDrag(false)
    pick(e.dataTransfer.files?.[0] ?? null)
  }

  const tab = (key: 'sample' | 'upload', label: string, Icon: typeof Database) => (
    <button
      type="button"
      role="tab"
      aria-selected={mode === key}
      onClick={() => setMode(key)}
      className={`inline-flex min-h-11 cursor-pointer items-center gap-2 border-b-2 px-1 text-[0.98rem] font-semibold transition-colors duration-150 ${mode === key ? 'border-accent text-ink' : 'border-transparent text-muted hover:text-ink'}`}
    >
      <Icon size={17} aria-hidden="true" />
      {label}
    </button>
  )

  return (
    <div className="rounded-lg border border-line bg-surface">
      <div role="tablist" aria-label="Input source" className="flex gap-6 border-b border-line px-5 sm:px-6">
        {tab('sample', 'Built-in test recording', Database)}
        {tab('upload', 'Upload a file', Upload)}
      </div>
      <div className="p-5 sm:p-6">
        {mode === 'sample' ? (
          <form
            className="flex flex-col gap-4 sm:flex-row sm:items-end"
            onSubmit={(e) => {
              e.preventDefault()
              if (selected) onSample(selected)
            }}
          >
            <div className="flex-1">
              <label htmlFor="sample" className="text-sm font-semibold text-ink-2">
                Recording from the held-out test setups (8 and 9)
              </label>
              <select
                id="sample"
                value={selected}
                onChange={(e: ChangeEvent<HTMLSelectElement>) => setSampleId(e.target.value)}
                disabled={!samples}
                className="mt-1.5 block min-h-11 w-full cursor-pointer rounded-md border border-line bg-surface px-3 text-base text-ink disabled:cursor-not-allowed disabled:opacity-50"
              >
                {samples?.samples.map((s) => (
                  <option key={s.id} value={s.id}>
                    {String(s.label).padStart(2, '0')} · {s.scenario} · setup {s.setup}
                  </option>
                ))}
              </select>
              <p className="mt-1.5 text-sm text-muted">{samplesError ? <span className="text-warn">{samplesError}</span> : 'One 60 s recording from the five reference sensors, never used for training or tuning.'}</p>
            </div>
            <button
              type="submit"
              disabled={busy || !selected}
              className="inline-flex min-h-11 cursor-pointer items-center justify-center gap-2 rounded-md bg-accent px-5 font-semibold text-white transition-colors duration-150 hover:bg-[#275d86] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {busy ? <Loader2 size={18} className="animate-spin" aria-hidden="true" /> : <Play size={18} aria-hidden="true" />}
              {busy ? 'Running models' : 'Diagnose'}
            </button>
          </form>
        ) : (
          <form onSubmit={submitUpload} className="flex flex-col gap-4">
            <div
              onDragOver={(e) => {
                e.preventDefault()
                setDrag(true)
              }}
              onDragLeave={() => setDrag(false)}
              onDrop={onDrop}
              className={`flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-4 py-8 text-center transition-colors duration-150 ${drag ? 'border-accent bg-accent-soft' : 'border-line bg-surface-2'}`}
            >
              <FileUp size={26} className="text-accent" aria-hidden="true" />
              <div className="text-base font-semibold text-ink">{file ? file.name : 'Drop a sensor file here'}</div>
              <div className="text-sm text-muted">{file ? `${(file.size / 1024).toFixed(0)} KB` : '.npy, .csv or .mat · up to 10 MB · at least 20 s of signal'}</div>
              <button type="button" onClick={() => inputRef.current?.click()} className="mt-2 min-h-11 cursor-pointer rounded-md border border-line bg-surface px-4 text-sm font-semibold text-ink-2 hover:border-accent hover:text-accent">
                Browse files
              </button>
              <input ref={inputRef} id="file" type="file" accept={ACCEPT.join(',')} className="sr-only" aria-label="Sensor file" onChange={(e) => pick(e.target.files?.[0] ?? null)} />
            </div>
            <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
              <div className="sm:w-56">
                <label htmlFor="fs" className="text-sm font-semibold text-ink-2">
                  Sampling rate (Hz)
                </label>
                <input id="fs" inputMode="decimal" value={fs} onChange={(e) => setFs(e.target.value)} className="tnum mt-1.5 block min-h-11 w-full rounded-md border border-line px-3 text-base" aria-describedby="fs-help" />
                <p id="fs-help" className="mt-1.5 text-sm text-muted">
                  Leave empty to use a CSV time column, else 100 Hz.
                </p>
              </div>
              <div className="flex-1 text-sm text-muted sm:pb-7">Accepted layouts: the 5 Z24 reference sensors (channels 22 to 26, in order) or all 27 channels, as rows or columns. A CSV may start with a time column.</div>
              <button
                type="submit"
                disabled={busy || !file}
                className="inline-flex min-h-11 cursor-pointer items-center justify-center gap-2 self-start rounded-md bg-accent px-5 font-semibold text-white transition-colors duration-150 hover:bg-[#275d86] disabled:cursor-not-allowed disabled:opacity-50 sm:mb-7 sm:self-auto"
              >
                {busy ? <Loader2 size={18} className="animate-spin" aria-hidden="true" /> : <Play size={18} aria-hidden="true" />}
                {busy ? 'Running models' : 'Diagnose file'}
              </button>
            </div>
            {fileError && (
              <p role="alert" className="text-sm font-semibold text-warn">
                {fileError}
              </p>
            )}
            <p className="text-sm text-muted">
              Example raw files from the test set:{' '}
              <a className="font-semibold text-accent underline-offset-2 hover:underline" href="/samples/z24_test_setup8_scenario00_ch22-26.csv" download>
                reference (CSV)
              </a>
              ,{' '}
              <a className="font-semibold text-accent underline-offset-2 hover:underline" href="/samples/z24_test_setup8_scenario04_ch22-26.csv" download>
                pier −80 mm (CSV)
              </a>
              ,{' '}
              <a className="font-semibold text-accent underline-offset-2 hover:underline" href="/samples/z24_test_setup8_scenario16_ch22-26.npy" download>
                6 tendons ruptured (5-channel NPY)
              </a>
              .
            </p>
          </form>
        )}
      </div>
    </div>
  )
}
