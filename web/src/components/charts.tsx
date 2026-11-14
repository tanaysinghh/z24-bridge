import { useEffect, useMemo, useRef, useState } from 'react'

function useWidth(fallback: number) {
  const ref = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(fallback)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const update = () => setWidth(Math.max(240, Math.round(el.getBoundingClientRect().width)))
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, width] as const
}

function fmtTick(v: number) {
  const a = Math.abs(v)
  if (a === 0) return '0'
  if (a >= 100) return v.toFixed(0)
  if (a < 0.01) return v.toExponential(0)
  return Number(v.toPrecision(2)).toString()
}

type LineProps = {
  x: number[]
  y: number[]
  height?: number
  xLabel: string
  yLabel: string
  marker?: { x: number; label: string } | null
  ariaLabel: string
  xTicks?: number[]
  fill?: boolean
}

const PAD = { l: 52, r: 12, t: 12, b: 34 }

function niceTicks(min: number, max: number, count = 5) {
  const span = max - min || 1
  const step = Math.pow(10, Math.floor(Math.log10(span / count)))
  const err = (count * step) / span
  const mult = err <= 0.15 ? 10 : err <= 0.35 ? 5 : err <= 0.75 ? 2 : 1
  const s = step * mult
  const ticks = []
  for (let v = Math.ceil(min / s) * s; v <= max + 1e-9; v += s) ticks.push(Number(v.toFixed(6)))
  return ticks
}

export function LineChart({ x, y, height = 220, xLabel, yLabel, marker, ariaLabel, xTicks, fill = false }: LineProps) {
  const [ref, W] = useWidth(280)
  const geo = useMemo(() => {
    const xmin = Math.min(...x)
    const xmax = Math.max(...x)
    const ymin = Math.min(...y)
    const ymax = Math.max(...y)
    const ih = height - PAD.t - PAD.b
    const iw = W - PAD.l - PAD.r
    const sx = (v: number) => PAD.l + ((v - xmin) / (xmax - xmin || 1)) * iw
    const sy = (v: number) => PAD.t + (1 - (v - ymin) / (ymax - ymin || 1)) * ih
    const d = x.map((xv, i) => `${i ? 'L' : 'M'}${sx(xv).toFixed(1)},${sy(y[i]).toFixed(1)}`).join('')
    const area = `${d}L${sx(xmax).toFixed(1)},${(PAD.t + ih).toFixed(1)}L${sx(xmin).toFixed(1)},${(PAD.t + ih).toFixed(1)}Z`
    return { sx, sy, d, area, xt: xTicks ?? niceTicks(xmin, xmax, W < 420 ? 4 : 6), yt: niceTicks(ymin, ymax, 4), ih }
  }, [x, y, height, xTicks, W])

  return (
    <div ref={ref} className="w-full min-w-0 overflow-hidden">
    <svg width={W} height={height} viewBox={`0 0 ${W} ${height}`} className="block" role="img" aria-label={ariaLabel}>
      {geo.yt.map((v) => (
        <g key={`y${v}`}>
          <line x1={PAD.l} x2={W - PAD.r} y1={geo.sy(v)} y2={geo.sy(v)} stroke="#e3e8ee" />
          <text x={PAD.l - 6} y={geo.sy(v) + 4} textAnchor="end" fontSize="11" fill="#5f6b78" className="tnum">
            {fmtTick(v)}
          </text>
        </g>
      ))}
      {geo.xt.map((v) => (
        <text key={`x${v}`} x={geo.sx(v)} y={height - PAD.b + 16} textAnchor="middle" fontSize="11" fill="#5f6b78" className="tnum">
          {v}
        </text>
      ))}
      {fill && <path d={geo.area} fill="#2f6f9f" opacity="0.08" />}
      <path d={geo.d} fill="none" stroke="#2f6f9f" strokeWidth="1.2" strokeLinejoin="round" />
      {marker && (
        <g>
          <line x1={geo.sx(marker.x)} x2={geo.sx(marker.x)} y1={PAD.t} y2={PAD.t + geo.ih} stroke="#b5512f" strokeDasharray="4 3" />
          <text x={geo.sx(marker.x) + 6} y={PAD.t + 12} fontSize="11" fill="#b5512f" fontWeight="600">
            {marker.label}
          </text>
        </g>
      )}
      <text x={(PAD.l + W - PAD.r) / 2} y={height - 4} textAnchor="middle" fontSize="11" fill="#5f6b78">
        {xLabel}
      </text>
      <text x={12} y={PAD.t + geo.ih / 2} textAnchor="middle" fontSize="11" fill="#5f6b78" transform={`rotate(-90 12 ${PAD.t + geo.ih / 2})`}>
        {yLabel}
      </text>
    </svg>
    </div>
  )
}

export function Histogram({ edges, counts, lower, upper, median }: { edges: number[]; counts: number[]; lower: number; upper: number; median: number }) {
  const [ref, W] = useWidth(260)
  const height = 110
  const max = Math.max(...counts, 1)
  const xmax = edges[edges.length - 1]
  const sx = (v: number) => 4 + (v / xmax) * (W - 8)
  const bw = (W - 8) / counts.length
  return (
    <div ref={ref} className="w-full min-w-0 overflow-hidden">
    <svg width={W} height={height + 22} viewBox={`0 0 ${W} ${height + 22}`} className="block" role="img" aria-label={`Monte Carlo distribution of remaining useful life, median ${median.toFixed(0)} years`}>
      {counts.map((c, i) => {
        const inside = edges[i + 1] >= lower && edges[i] <= upper
        const h = (c / max) * height
        return <rect key={i} x={4 + i * bw + 1} y={height - h} width={Math.max(bw - 2, 1)} height={h} rx="2" fill={inside ? '#2f6f9f' : '#c9d6e2'} />
      })}
      <line x1={sx(median)} x2={sx(median)} y1={0} y2={height} stroke="#1f2933" strokeWidth="1.5" />
      {[0, xmax / 2, xmax].map((v) => (
        <text key={v} x={Math.min(Math.max(sx(v), 14), W - 14)} y={height + 16} textAnchor="middle" fontSize="11" fill="#5f6b78" className="tnum">
          {v.toFixed(0)} y
        </text>
      ))}
    </svg>
    </div>
  )
}

export function DsiScale({ value, lower, upper, tiers }: { value: number; lower: number; upper: number; tiers: { name: string; lower: number; upper: number }[] }) {
  const [ref, W] = useWidth(260)
  const colors = ['#2f7a5c', '#6a9a3a', '#b08a16', '#c26a2c', '#b5362f']
  const H = 46
  const sx = (v: number) => 4 + Math.max(0, Math.min(1, v)) * (W - 8)
  return (
    <div ref={ref} className="w-full min-w-0 overflow-hidden">
    <svg width={W} height={H + 24} viewBox={`0 0 ${W} ${H + 24}`} className="block" role="img" aria-label={`Damage severity index ${value.toFixed(2)} on a 0 to 1 scale`}>
      {tiers.map((t, i) => (
        <g key={t.name}>
          <rect x={sx(t.lower)} y={18} width={sx(Math.min(t.upper, 1)) - sx(t.lower) - 2} height={10} rx="2" fill={colors[i]} opacity="0.85" />
          <text x={(sx(t.lower) + sx(Math.min(t.upper, 1))) / 2} y={H + 16} textAnchor="middle" fontSize="11" fill="#5f6b78">
            {W < 330 && t.name.length > 6 ? t.name.slice(0, 3) + '.' : t.name}
          </text>
        </g>
      ))}
      <rect x={sx(lower)} y={14} width={Math.max(sx(upper) - sx(lower), 2)} height={18} rx="3" fill="none" stroke="#1f2933" strokeWidth="1.2" strokeDasharray="3 2" />
      <path d={`M${sx(value) - 7},2 L${sx(value) + 7},2 L${sx(value)},13 Z`} fill="#1f2933" />
      {[0, 0.5, 1].map((v) => (
        <text key={v} x={Math.min(Math.max(sx(v), 8), W - 8)} y={H} textAnchor="middle" fontSize="11" fill="#8b97a5" className="tnum">
          {v.toFixed(1)}
        </text>
      ))}
    </svg>
    </div>
  )
}
