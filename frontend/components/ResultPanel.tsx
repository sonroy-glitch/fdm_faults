import { formatPercent } from "@/lib/api";
import type { Prediction } from "@/lib/types";

export default function ResultPanel({ result }: { result: Prediction }) {
  const [top, ...rest] = result.probabilities;

  return (
    <section
      aria-live="polite"
      className="rounded-xl border border-edge bg-panel/60 p-6"
    >
      <p className="text-xs uppercase tracking-[0.18em] text-muted">
        Most likely fault
      </p>
      <div className="mt-2 flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h2 className="text-2xl font-semibold tracking-tight">{top.label}</h2>
        <span className="text-sm font-medium text-accent">
          {formatPercent(top.confidence)} confident
        </span>
      </div>

      <ul className="mt-6 space-y-3">
        {[top, ...rest].map((entry, index) => (
          <li key={entry.label}>
            <div className="flex items-baseline justify-between text-sm">
              <span className={index === 0 ? "text-slate-100" : "text-muted"}>
                {entry.label}
              </span>
              <span className="tabular-nums text-muted">
                {formatPercent(entry.confidence)}
              </span>
            </div>
            <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-edge">
              <div
                className={index === 0 ? "h-full bg-accent" : "h-full bg-muted/50"}
                style={{ width: `${Math.max(entry.confidence * 100, 0.5)}%` }}
              />
            </div>
          </li>
        ))}
      </ul>

      <p className="mt-6 text-xs text-muted">
        {result.engine} · {result.latency_ms.toFixed(0)} ms
      </p>
    </section>
  );
}
