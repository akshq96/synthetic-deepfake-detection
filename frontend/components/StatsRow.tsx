import { Clock, Frame, Percent, Users } from "lucide-react";

interface Stat {
  icon: React.ElementType;
  label: string;
  value: string;
}

// Each stat renders only when the backend actually computed it — a null
// field (e.g. faces-detected for a video, or any of these for an old
// prediction from before this feature existed) is omitted rather than
// shown as a fabricated placeholder.
export function StatsRow({
  framesAnalyzed,
  framesTotal,
  facesDetected,
  avgConfidence,
  processingTimeMs,
}: {
  framesAnalyzed: number | null;
  framesTotal: number | null;
  facesDetected: number | null;
  avgConfidence: number | null;
  processingTimeMs: number | null;
}) {
  const stats: Stat[] = [];

  if (framesAnalyzed != null) {
    stats.push({
      icon: Frame,
      label: "Frames Analyzed",
      value: framesTotal != null ? `${framesAnalyzed} / ${framesTotal}` : String(framesAnalyzed),
    });
  }
  if (facesDetected != null) {
    stats.push({ icon: Users, label: "Faces Detected", value: String(facesDetected) });
  }
  if (avgConfidence != null) {
    stats.push({ icon: Percent, label: "Confidence", value: `${(avgConfidence * 100).toFixed(1)}%` });
  }
  if (processingTimeMs != null) {
    const seconds = processingTimeMs / 1000;
    stats.push({
      icon: Clock,
      label: "Processing Time",
      value: seconds >= 1 ? `${seconds.toFixed(1)} sec` : `${Math.round(processingTimeMs)} ms`,
    });
  }

  if (stats.length === 0) return null;

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {stats.map((stat, i) => (
        <div
          key={stat.label}
          style={{ animationDelay: `${i * 60}ms` }}
          className="animate-fade-slide-up flex items-center gap-2.5 rounded-lg border border-border px-3.5 py-3"
        >
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary-tint text-primary">
            <stat.icon className="h-4 w-4" />
          </span>
          <div className="min-w-0">
            <div className="meta-label truncate text-[9.5px]">{stat.label}</div>
            <div className="mono-value truncate text-[14px] font-medium text-foreground">{stat.value}</div>
          </div>
        </div>
      ))}
    </div>
  );
}
