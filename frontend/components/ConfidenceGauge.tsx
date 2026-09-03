import { Badge } from "@/components/ui/Badge";
import { cn } from "@/lib/cn";
import type { PredictionLabel } from "@/lib/types";

const LABEL_META: Record<PredictionLabel, { text: string; variant: "success" | "danger" | "warning"; bar: string }> = {
  real: { text: "Real", variant: "success", bar: "bg-success" },
  fake: { text: "Fake", variant: "danger", bar: "bg-danger" },
  abstain: { text: "Uncertain / Abstained", variant: "warning", bar: "bg-warning" },
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

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <Badge variant={meta.variant}>{meta.text}</Badge>
        <span className="text-sm text-muted-foreground">
          {(confidence * 100).toFixed(1)}% confidence
        </span>
      </div>

      <div>
        <div className="mb-1 flex justify-between text-[11px] text-muted-foreground">
          <span>Real</span>
          <span>Fake probability: {(fakeProbability * 100).toFixed(1)}%</span>
        </div>
        <div className="h-2.5 w-full overflow-hidden rounded-full bg-muted">
          <div
            className={cn("h-full rounded-full transition-all", meta.bar)}
            style={{ width: `${Math.min(100, Math.max(0, fakeProbability * 100))}%` }}
          />
        </div>
      </div>
    </div>
  );
}
