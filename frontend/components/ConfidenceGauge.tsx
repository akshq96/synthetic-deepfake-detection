import { cn } from "@/lib/cn";
import type { PredictionLabel } from "@/lib/types";
import { AlertTriangle, HelpCircle, ShieldCheck } from "lucide-react";

const LABEL_META: Record<
  PredictionLabel,
  { text: string; color: string; bar: string; tint: string; icon: React.ElementType; description: string }
> = {
  real: {
    text: "Real",
    color: "text-success",
    bar: "bg-success",
    tint: "bg-success-tint",
    icon: ShieldCheck,
    description: "No significant evidence of manipulation detected.",
  },
  fake: {
    text: "Deepfake",
    color: "text-danger",
    bar: "bg-danger",
    tint: "bg-danger-tint",
    icon: AlertTriangle,
    description: "Strong evidence of manipulation detected in the analyzed media.",
  },
  abstain: {
    text: "Uncertain",
    color: "text-warning",
    bar: "bg-warning",
    tint: "bg-warning-tint",
    icon: HelpCircle,
    description: "Below the confidence threshold for a reliable verdict either way.",
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
  const Icon = meta.icon;
  const realPct = (1 - fakeProbability) * 100;
  const fakePct = fakeProbability * 100;

  return (
    <div>
      <div className="flex items-start gap-3">
        <span className={cn("flex h-11 w-11 shrink-0 items-center justify-center rounded-full", meta.tint)}>
          <Icon className={cn("h-5 w-5", meta.color)} />
        </span>
        <div>
          <div
            data-slot="verdict"
            className={cn("font-display text-[19px] font-bold uppercase tracking-wide", meta.color)}
          >
            {meta.text}
          </div>
          <div className="mono-value mt-0.5 text-[13px] text-muted-foreground">
            {(confidence * 100).toFixed(1)}% confidence
          </div>
        </div>
      </div>

      <p className="mt-3 text-[13px] leading-relaxed text-foreground-secondary">{meta.description}</p>

      <div className="mt-5">
        <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-muted">
          <div className="h-full bg-success transition-all" style={{ width: `${Math.max(0, Math.min(100, realPct))}%` }} />
          <div className="h-full bg-danger transition-all" style={{ width: `${Math.max(0, Math.min(100, fakePct))}%` }} />
        </div>
        <div className="mono-value mt-2 flex items-center justify-between text-[12.5px]">
          <span className="text-success">{realPct.toFixed(1)}% Real</span>
          <span className="text-danger">{fakePct.toFixed(1)}% Deepfake</span>
        </div>
      </div>
    </div>
  );
}
