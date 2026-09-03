import { cn } from "@/lib/cn";
import type { PredictionLabel } from "@/lib/types";

const LABEL_META: Record<
  PredictionLabel,
  { text: string; color: string; bar: string; tint: string }
> = {
  real: { text: "Real", color: "text-success", bar: "bg-success", tint: "bg-success-tint" },
  fake: { text: "Deepfake", color: "text-danger", bar: "bg-danger", tint: "bg-danger-tint" },
  abstain: {
    text: "Uncertain",
    color: "text-warning",
    bar: "bg-warning",
    tint: "bg-warning-tint",
  },
};

export function ConfidenceGauge({
  label,
  confidence,
  fakeProbability,
}: {
  label: PredictionLabel;
  confidence: number;
  fakeProbability: number;
}) {
  const meta = LABEL_META[label];
  const realPct = (1 - fakeProbability) * 100;
  const fakePct = fakeProbability * 100;

  return (
    <div>
      <div className={cn("inline-flex items-center gap-2 rounded px-2.5 py-1", meta.tint)}>
        <span className={cn("h-2 w-2 rounded-full", meta.bar)} />
        <span
          data-slot="verdict"
          className={cn("font-display text-[13px] font-semibold uppercase tracking-wide", meta.color)}
        >
          {meta.text}
        </span>
      </div>
      <div className="mono-value mt-2 text-[13px] text-muted-foreground">
        {(confidence * 100).toFixed(1)}% confidence
      </div>

      <div className="mt-5 space-y-2.5">
        <ProbabilityRow label="Real" pct={realPct} barClass="bg-success" />
        <ProbabilityRow label="Deepfake" pct={fakePct} barClass="bg-danger" />
      </div>
    </div>
  );
}

function ProbabilityRow({ label, pct, barClass }: { label: string; pct: number; barClass: string }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-16 shrink-0 text-[12px] text-foreground-secondary">{label}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
        <div
          className={cn("h-full rounded-full transition-all", barClass)}
          style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
        />
      </div>
      <span className="mono-value w-12 shrink-0 text-right text-[12px] text-foreground">
        {pct.toFixed(1)}%
      </span>
    </div>
  );
}
