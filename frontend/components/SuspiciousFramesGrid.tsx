"use client";

import { cn } from "@/lib/cn";
import { staticUrl } from "@/lib/api-client";
import type { SuspiciousFrameOut } from "@/lib/types";

// Replaces the old text-pill frame strip with actual image thumbnails —
// selecting one feeds GradCamPanel above via the lifted selectedIndex state
// (owned by the Detect page, so both stay in sync).
export function SuspiciousFramesGrid({
  frames,
  selectedIndex,
  onSelect,
}: {
  frames: SuspiciousFrameOut[];
  selectedIndex: number;
  onSelect: (index: number) => void;
}) {
  if (frames.length === 0) {
    return (
      <div className="rounded-lg bg-muted p-4 text-center text-[12px] text-muted-foreground">
        No suspicious frames flagged for this video.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-5 gap-2">
      {frames.map((frame, i) => {
        const thumbnail = frame.heatmap_path ?? frame.original_path;
        return (
          <button
            key={`${frame.track_id}-${frame.frame_index}`}
            type="button"
            onClick={() => onSelect(i)}
            className={cn(
              "group relative aspect-square overflow-hidden rounded-md border-2 transition-colors",
              i === selectedIndex ? "border-primary" : "border-transparent hover:border-border-strong"
            )}
          >
            {thumbnail ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={staticUrl(thumbnail)} alt={`Frame ${frame.frame_index}`} className="h-full w-full object-cover" />
            ) : (
              <div className="flex h-full w-full items-center justify-center bg-muted text-[9px] text-muted-foreground">
                —
              </div>
            )}
            <span className="absolute bottom-0 left-0 right-0 bg-black/60 px-1 py-0.5 text-center text-[9.5px] font-medium text-white">
              {(frame.fake_probability * 100).toFixed(0)}%
            </span>
            <span className="mono-value absolute left-1 top-1 rounded bg-black/60 px-1 text-[8.5px] text-white">
              {frame.timestamp.toFixed(2)}s
            </span>
          </button>
        );
      })}
    </div>
  );
}
