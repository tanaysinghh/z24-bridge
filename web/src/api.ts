export const API_URL = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '') || 'http://localhost:8000'
export const DASHBOARD_URL = 'https://vibradiagnose.streamlit.app'
export const REPO_URL = 'https://github.com/tanaysinghh/z24-bridge'

export type ClassProb = { label: number; name: string; group: string; probability: number }

export type ModelVote = {
  key: string
  name: string
  label: number
  scenario: string
  group: string
  probability: number
  probability_of_ensemble_class: number
  agrees: boolean
  agrees_group: boolean
  probabilities: number[]
}

export type Tier = { index: number; name: string; lower: number; upper: number; action: string; urgency: string }

export type Prediction = {
  input: {
    source: string | null
    n_channels: number
    channels: number[]
    fs_input: number
    fs_model: number
    duration_s: number
    n_windows: number
    warnings: string[]
  }
  prediction: { label: number; scenario: string; group: string; probability: number; method: string; probabilities: ClassProb[] }
  agreement: { models_agreeing: number; models_total: number; hierarchy_agrees: boolean; hierarchy_group_agrees: boolean; models: ModelVote[] }
  hierarchy: {
    group: { name: string; probability: number; description: string; probabilities: Record<string, number> }
    scenario: { label: number; name: string; probability_within_group: number; within_group: { label: number; name: string; probability: number }[] }
  }
  severity: {
    dsi: { expected: number; mean: number; median: number; lower: number; upper: number }
    defined_dsi_of_prediction: number
    tier: Tier
    tier_probabilities: Record<string, number>
  }
  rul: {
    mean: number
    median: number
    lower: number
    upper: number
    interval: number
    histogram: { edges: number[]; counts: number[] }
    params: Record<string, number>
    disclaimer: string
  }
  windows: { start_s: number; label: number; scenario: string; probability: number }[]
  signal: { t: number[]; y: number[]; fs: number }
  spectrum: { f: number[]; db: number[]; first_mode_hz: number | null }
  truth?: { label: number; scenario: string; group: string; setup: number; defined_dsi: number }
  elapsed_ms: number
}

export type Sample = {
  id: string
  row: number
  label: number
  scenario: string
  group: string
  setup: number
  segment_id: number
  duration_s: number
  expected: { label: number; scenario: string; probability: number }
}

export type SampleIndex = { channels: number[]; note: string; samples: Sample[] }

export type Pair = [number, number]

export type Metrics = {
  generated: string
  base_models: Record<string, Record<string, Pair>> | null
  ensemble: { primary: string; summary: Record<string, Record<string, Pair>> } | null
  hierarchical: { summary: Record<string, Record<string, Pair>>; stage2: Record<string, Record<string, Pair>> } | null
  rul_validation: Record<string, Pair> | null
}

export class ApiError extends Error {
  code: string
  constructor(message: string, code = 'error') {
    super(message)
    this.code = code
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (res.ok) return (await res.json()) as T
  let message = `Request failed with status ${res.status}.`
  let code = 'http_error'
  try {
    const body = await res.json()
    if (typeof body.detail === 'string') message = body.detail
    if (typeof body.code === 'string') code = body.code
  } catch {
    code = 'http_error'
  }
  throw new ApiError(message, code)
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${API_URL}${path}`, init)
  } catch {
    throw new ApiError('Could not reach the VibraDiagnose API. Check your connection and try again.', 'network')
  }
  return handle<T>(res)
}

export const api = {
  samples: () => request<SampleIndex>('/api/samples'),
  metrics: () => request<Metrics>('/api/metrics'),
  predictSample: (id: string) => request<Prediction>(`/api/samples/${encodeURIComponent(id)}/prediction`),
  predictFile: (file: File, fs?: number) => {
    const form = new FormData()
    form.append('file', file)
    if (fs !== undefined && Number.isFinite(fs)) form.append('fs', String(fs))
    return request<Prediction>('/api/predict', { method: 'POST', body: form })
  },
}

export const pct = (x: number, digits = 1) => `${(100 * x).toFixed(digits)}%`
