import type { ReactNode } from 'react'

export function Section({ id, title, sub, children }: { id?: string; title: string; sub?: ReactNode; children: ReactNode }) {
  return (
    <section id={id} className="scroll-mt-20 pt-12">
      <h2 className="text-2xl font-bold tracking-tight text-ink sm:text-3xl">{title}</h2>
      {sub && <p className="mt-2 max-w-3xl text-base text-muted sm:text-lg">{sub}</p>}
      <div className="mt-6">{children}</div>
    </section>
  )
}

export function Card({ children, className = '', accent = false }: { children: ReactNode; className?: string; accent?: boolean }) {
  return (
    <div className={`min-w-0 rounded-lg border border-line bg-surface p-5 sm:p-6 ${accent ? 'border-l-[3px] border-l-accent' : ''} ${className}`}>{children}</div>
  )
}

export function Label({ children }: { children: ReactNode }) {
  return <div className="text-xs font-semibold uppercase tracking-[0.08em] text-muted">{children}</div>
}

export function Big({ children, tone = 'ink', size = 'lg' }: { children: ReactNode; tone?: 'ink' | 'accent'; size?: 'xl' | 'lg' | 'md' }) {
  const sizes = { xl: 'text-5xl sm:text-6xl', lg: 'text-4xl sm:text-5xl', md: 'text-3xl sm:text-4xl' }
  return <div className={`tnum mt-2 font-bold leading-none tracking-tight ${sizes[size]} ${tone === 'accent' ? 'text-accent' : 'text-ink'}`}>{children}</div>
}

export function Note({ children }: { children: ReactNode }) {
  return <div className="mt-2 text-sm text-muted">{children}</div>
}

export function Pill({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'accent' | 'good' | 'warn' }) {
  const tones = {
    neutral: 'border-line bg-surface-2 text-ink-2',
    accent: 'border-[#c9dbea] bg-accent-soft text-accent',
    good: 'border-[#cfe5da] bg-good-soft text-good',
    warn: 'border-[#efd3c8] bg-warn-soft text-warn',
  }
  return <span className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-semibold ${tones[tone]}`}>{children}</span>
}

export function Bar({ value, tone = 'accent', label }: { value: number; tone?: 'accent' | 'muted' | 'good' | 'warn'; label: string }) {
  const tones = { accent: 'bg-accent', muted: 'bg-[#b7c7d6]', good: 'bg-good', warn: 'bg-warn' }
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-surface-2" role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(value * 100)}>
      <div className={`h-full rounded-full ${tones[tone]} transition-[width] duration-300`} style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
    </div>
  )
}

export function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`animate-pulse rounded-lg bg-surface-2 ${className}`} />
}
