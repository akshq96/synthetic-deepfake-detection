"use client";

import { staticUrl } from "@/lib/api-client";
import { X, ZoomIn } from "lucide-react";
import { useState } from "react";

export interface EvidenceFrame {
  originalPath: string | null;
  heatmapOnlyPath: string | null;
  overlayPath: string | null;
}

// The Grad-CAM three-panel view: Original / Heatmap / Overlay side by side
// for one frame (a single image prediction's own evidence, or whichever
// suspicious video frame is currently selected in SuspiciousFramesGrid).
export function GradCamPanel({ frame }: { frame: EvidenceFrame | null }) {
  const [zoomed, setZoomed] = useState<string | null>(null);

  if (!frame || (!frame.originalPath && !frame.heatmapOnlyPath && !frame.overlayPath)) {
    return (
      <div className="flex h-40 items-center justify-center rounded-lg bg-muted text-[12px] text-muted-foreground">
        No heatmap available
      </div>
    );
  }

  return (
    <>
      <div className="grid grid-cols-3 gap-2">
        <Panel label="Original" path={frame.originalPath} onZoom={setZoomed} />
        <Panel label="Heatmap" path={frame.heatmapOnlyPath} onZoom={setZoomed} />
        <Panel label="Overlay" path={frame.overlayPath} onZoom={setZoomed} />
      </div>
      <p className="mt-3 text-[12px] leading-relaxed text-muted-foreground">
        The heatmap highlights regions that influenced the model&apos;s decision. Warmer colors indicate a
        stronger contribution toward the prediction.
      </p>

      {zoomed && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-6"
          onClick={() => setZoomed(null)}
        >
          <button
            type="button"
            className="absolute right-6 top-6 rounded-full bg-white/10 p-2 text-white hover:bg-white/20"
            onClick={() => setZoomed(null)}
          >
            <X className="h-5 w-5" />
          </button>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={zoomed} alt="" className="max-h-full max-w-full rounded-lg object-contain" />
        </div>
      )}
    </>
  );
}

function Panel({ label, path, onZoom }: { label: string; path: string | null; onZoom: (url: string) => void }) {
  return (
    <div>
      <div className="meta-label mb-1.5 text-center text-[9px]">{label}</div>
      {path ? (
        <button
          type="button"
          onClick={() => onZoom(staticUrl(path))}
          className="group relative block aspect-square w-full overflow-hidden rounded-md border border-border"
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={staticUrl(path)} alt={label} className="h-full w-full object-cover" />
          <div className="absolute inset-0 flex items-center justify-center bg-black/0 opacity-0 transition-opacity group-hover:bg-black/30 group-hover:opacity-100">
            <ZoomIn className="h-4 w-4 text-white" />
          </div>
        </button>
      ) : (
        <div className="flex aspect-square items-center justify-center rounded-md border border-border bg-muted text-[10px] text-muted-foreground">
          —
        </div>
      )}
    </div>
  );
}
