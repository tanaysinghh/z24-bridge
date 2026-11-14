import type { Metrics, Pair } from '../api'
import { Big, Card, Label, Note, Skeleton } from './ui'

const ROWS: { key: string; name: string; source: 'base' | 'ensemble' | 'hier' }[] = [
  { key: 'wavenet', name: 'WaveNet', source: 'base' },
  { key: 'inceptiontime', name: 'InceptionTime', source: 'base' },
  { key: 'minirocket', name: 'MiniRocket', source: 'base' },
  { key: 'simple_average', name: 'Simple average', source: 'ensemble' },
  { key: 'val_weighted_average', name: 'Validation-weighted average', source: 'ensemble' },
  { key: 'stacking', name: 'Stacking (logistic regression)', source: 'ensemble' },
  { key: 'end_to_end_hard', name: 'Hierarchical, end to end', source: 'hier' },
]

const f = (p?: Pair) => (p ? `${p[0].toFixed(1)}%` : 'n/a')
const sd = (p?: Pair) => (p ? `± ${p[1].toFixed(1)}` : '')

export function Performance({ metrics, error }: { metrics: Metrics | null; error: string | null }) {
  if (error) return <p className="text-warn">{error}</p>
  if (!metrics) return <Skeleton className="h-64" />
  const get = (row: (typeof ROWS)[number]) => {
    if (row.source === 'base') return metrics.base_models?.[row.key]
    if (row.source === 'ensemble') return metrics.ensemble?.summary[row.key]
    return metrics.hierarchical?.summary[row.key]
  }
  const primary = metrics.ensemble?.primary
  const best = ROWS.map((r) => ({ r, v: get(r)?.test_segment_acc?.[0] ?? -1 })).sort((a, b) => b.v - a.v)[0]
  const stage1 = metrics.hierarchical?.summary.stage1_dedicated?.test_segment_acc
  const rho = metrics.rul_validation?.test_spearman_rho
  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-3">
        <Card accent>
          <Label>Best 17-class result · {best.r.name}</Label>
          <Big tone="accent">{f(get(best.r)?.test_segment_acc)}</Big>
          <Note>per 60 s test recording, chance is 5.9%</Note>
        </Card>
        <Card>
          <Label>Damage group, Stage 1</Label>
          <Big>{f(stage1)}</Big>
          <Note>5 groups on test recordings, chance is 20%</Note>
        </Card>
        <Card>
          <Label>Severity ordering</Label>
          <Big>{rho ? rho[0].toFixed(2) : 'n/a'}</Big>
          <Note>Spearman correlation of predicted vs defined DSI on test recordings</Note>
        </Card>
      </div>
      <Card>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[36rem] text-right text-[0.97rem]">
            <thead>
              <tr className="border-b border-line text-xs font-semibold text-muted">
                <th className="py-2 text-left font-semibold">Method</th>
                <th className="py-2 font-semibold">Validation, recording</th>
                <th className="py-2 font-semibold">Test, window</th>
                <th className="py-2 font-semibold">Test, recording</th>
              </tr>
            </thead>
            <tbody>
              {ROWS.map((row) => {
                const v = get(row)
                const isPrimary = row.key === primary
                return (
                  <tr key={row.key} className={`border-b border-line last:border-0 ${isPrimary ? 'font-semibold text-ink' : 'text-ink-2'}`}>
                    <td className="py-3 text-left">
                      {row.name}
                      {isPrimary && <span className="ml-2 rounded border border-[#c9dbea] px-1.5 py-0.5 text-[0.7rem] font-semibold text-accent">DEPLOYED</span>}
                    </td>
                    <td className="tnum py-3">
                      {f(v?.val_segment_acc)} <span className="text-xs text-faint">{sd(v?.val_segment_acc)}</span>
                    </td>
                    <td className="tnum py-3">
                      {f(v?.test_window_acc)} <span className="text-xs text-faint">{sd(v?.test_window_acc)}</span>
                    </td>
                    <td className="tnum py-3">
                      {f(v?.test_segment_acc)} <span className="text-xs text-faint">{sd(v?.test_segment_acc)}</span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <Note>
          Mean ± std over three seeds. Train on setups 1-6, validation on setup 7, test on setups 8-9. The test setups were never used to tune, select or fit anything. Numbers come from the committed result files, generated {metrics.generated}.
        </Note>
      </Card>
    </div>
  )
}
