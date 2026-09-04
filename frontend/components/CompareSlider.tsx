"use client";

import { useCallback, useRef, useState } from "react";

// A draggable before/after divider over two stacked images of identical
// size. Hand-rolled rather than a library dependency — the interaction is
// small (clamp a pointer x-position to a percentage, clip the top image).
export function CompareSlider({
  beforeSrc,
  afterSrc,
  beforeLabel = "Original",
  afterLabel = "Analyzed",
}: {
  beforeSrc: string;
  afterSrc: string;
  beforeLabel?: string;
  afterLabel?: string;
}) {
  const [pct, setPct] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const updateFromClientX = useCallback((clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const next = ((clientX - rect.left) / rect.width) * 100;
    setPct(Math.max(0, Math.min(100, next)));
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative aspect-video w-full select-none overflow-hidden rounded-lg border border-border bg-black"
      onMouseDown={(e) => {
        dragging.current = true;
        updateFromClientX(e.clientX);
      }}
      onMouseMove={(e) => {
        if (dragging.current) updateFromClientX(e.clientX);
      }}
      onMouseUp={() => (dragging.current = false)}
      onMouseLeave={() => (dragging.current = false)}
      onTouchStart={(e) => updateFromClientX(e.touches[0].clientX)}
      onTouchMove={(e) => updateFromClientX(e.touches[0].clientX)}
    >
      {/* eslint-disable-next-line @next/next/no-img-element -- local blob/static URLs, no next/image optimization needed */}
      <img src={afterSrc} alt={afterLabel} className="absolute inset-0 h-full w-full object-contain" draggable={false} />
      {/* Full-size copy of the "before" image, clipped to the left pct% —
          avoids measuring container width in JS to keep a shrunken nested
          image from compressing on first paint. */}
      <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - pct}% 0 0)` }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={beforeSrc} alt={beforeLabel} className="absolute inset-0 h-full w-full object-contain" draggable={false} />
      </div>

      <span className="absolute left-3 top-3 rounded bg-black/60 px-2 py-1 text-[11px] font-medium text-white">
        {beforeLabel}
      </span>
      <span className="absolute right-3 top-3 rounded bg-primary px-2 py-1 text-[11px] font-medium text-primary-foreground">
        {afterLabel}
      </span>

      <div
        className="absolute inset-y-0 w-0.5 cursor-ew-resize bg-white"
        style={{ left: `${pct}%` }}
      >
        <div className="absolute left-1/2 top-1/2 flex h-8 w-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-white shadow-md">
          <div className="flex gap-0.5">
            <div className="h-3 w-0.5 rounded-full bg-muted-foreground" />
            <div className="h-3 w-0.5 rounded-full bg-muted-foreground" />
          </div>
        </div>
      </div>
    </div>
  );
}
