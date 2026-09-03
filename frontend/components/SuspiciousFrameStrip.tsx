"use client";

import { HeatmapOverlay } from "@/components/HeatmapOverlay";
import type { SuspiciousFrameOut } from "@/lib/types";
import { useState } from "react";

export function SuspiciousFrameStrip({ frames }: { frames: SuspiciousFrameOut[] }) {
  const [selected, setSelected] = useState(0);

  if (frames.length === 0) {
    return (
      <div className="rounded-lg bg-muted p-4 text-center text-xs text-muted-foreground">
        No suspicious frames flagged for this video.
      </div>
    );
  }

  const active = frames[selected];

  return (
    <div className="space-y-3">
      <HeatmapOverlay
        heatmapPath={active.heatmap_path}
        alt={`Suspicious frame ${active.frame_index}`}
        className="aspect-video w-full"
      />
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>
          Frame {active.frame_index} · t={active.timestamp.toFixed(2)}s · track {active.track_id}
        </span>
        <span className="font-medium text-danger">
          {(active.fake_probability * 100).toFixed(1)}% fake
        </span>
      </div>

      <div className="flex gap-2 overflow-x-auto pb-1">
        {frames.map((frame, i) => (
          <button
            key={`${frame.track_id}-${frame.frame_index}`}
            onClick={() => setSelected(i)}
            className={`shrink-0 rounded-md border px-2.5 py-1.5 text-[11px] transition-colors ${
              i === selected
                ? "border-primary bg-primary/10 text-primary"
                : "border-border text-muted-foreground hover:bg-muted"
            }`}
          >
            #{frame.frame_index}
          </button>
        ))}
      </div>
    </div>
  );
}
