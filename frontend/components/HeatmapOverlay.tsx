"use client";

// The backend pre-renders the heatmap alpha-blended onto the original crop
// (ml/xai/overlay.py) and serves it as a single static PNG — so this
// component's job is just to present that image well (click-to-zoom lightbox),
// not to re-composite layers client-side.

import { staticUrl } from "@/lib/api-client";
import { useState } from "react";
import { X, ZoomIn } from "lucide-react";

export function HeatmapOverlay({
  heatmapPath,
  alt = "Explainability heatmap",
  className,
}: {
  heatmapPath: string | null;
  alt?: string;
  className?: string;
}) {
  const [zoomed, setZoomed] = useState(false);

  if (!heatmapPath) {
    return (
      <div className="flex h-40 items-center justify-center rounded-lg bg-muted text-xs text-muted-foreground">
        No heatmap available
      </div>
    );
  }

  const url = staticUrl(heatmapPath);

  return (
    <>
      <button
        type="button"
        onClick={() => setZoomed(true)}
        className={`group relative overflow-hidden rounded-lg border border-border ${className ?? ""}`}
      >
        {/* eslint-disable-next-line @next/next/no-img-element -- backend-served, dynamic path; no next/image optimization needed for a local dev tool */}
        <img src={url} alt={alt} className="h-full w-full object-cover" />
        <div className="absolute inset-0 flex items-center justify-center bg-black/0 opacity-0 transition-opacity group-hover:bg-black/30 group-hover:opacity-100">
          <ZoomIn className="h-5 w-5 text-white" />
        </div>
      </button>

      {zoomed && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-6"
          onClick={() => setZoomed(false)}
        >
          <button
            type="button"
            className="absolute right-6 top-6 rounded-full bg-white/10 p-2 text-white hover:bg-white/20"
            onClick={() => setZoomed(false)}
          >
            <X className="h-5 w-5" />
          </button>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={url} alt={alt} className="max-h-full max-w-full rounded-lg object-contain" />
        </div>
      )}
    </>
  );
}
