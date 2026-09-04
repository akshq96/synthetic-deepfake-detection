"use client";

import { Button } from "@/components/ui/Button";
import { Spinner } from "@/components/ui/Spinner";
import { api, staticUrl } from "@/lib/api-client";
import type { PredictionOut } from "@/lib/types";
import { useMutation } from "@tanstack/react-query";
import { Check, Download, FileText, RotateCcw, Share2 } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

export function QuickActions({ result, onRunNew }: { result: PredictionOut; onRunNew: () => void }) {
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      <GenerateReportAction predictionId={result.id} />
      <ShareAction result={result} />
      <ExportFramesAction result={result} />
      <Button variant="secondary" className="w-full" onClick={onRunNew}>
        <RotateCcw className="h-4 w-4" />
        Run new analysis
      </Button>
    </div>
  );
}

function GenerateReportAction({ predictionId }: { predictionId: string }) {
  const [reportId, setReportId] = useState<string | null>(null);
  const mutation = useMutation({
    mutationFn: () => api.generateReport(predictionId),
    onSuccess: (report) => setReportId(report.id),
  });

  if (reportId) {
    return (
      <Link href={`/reports/${reportId}`}>
        <Button variant="secondary" className="w-full">
          <FileText className="h-4 w-4" />
          View report
        </Button>
      </Link>
    );
  }

  return (
    <Button variant="secondary" className="w-full" onClick={() => mutation.mutate()} disabled={mutation.isPending}>
      {mutation.isPending ? <Spinner className="h-4 w-4" /> : <FileText className="h-4 w-4" />}
      Download report
    </Button>
  );
}

function ShareAction({ result }: { result: PredictionOut }) {
  const [copied, setCopied] = useState(false);

  async function share() {
    const text = `AegisTrace detection result: ${result.label} (${(result.confidence * 100).toFixed(1)}% confidence)`;
    const url = window.location.href;
    if (navigator.share) {
      try {
        await navigator.share({ title: "AegisTrace detection result", text, url });
      } catch {
        // user cancelled the native share sheet — not an error
      }
      return;
    }
    try {
      await navigator.clipboard.writeText(`${text} — ${url}`);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // clipboard unavailable (e.g. insecure context) — silently no-op rather than throw
    }
  }

  return (
    <Button variant="secondary" className="w-full" onClick={share}>
      {copied ? <Check className="h-4 w-4" /> : <Share2 className="h-4 w-4" />}
      {copied ? "Link copied" : "Share result"}
    </Button>
  );
}

function ExportFramesAction({ result }: { result: PredictionOut }) {
  const [exporting, setExporting] = useState(false);

  const paths = [
    result.heatmap_path,
    result.original_path,
    result.heatmap_only_path,
    ...result.suspicious_frames.flatMap((f) => [f.heatmap_path, f.original_path, f.heatmap_only_path]),
  ].filter((p): p is string => Boolean(p));

  async function exportFrames() {
    if (paths.length === 0) return;
    setExporting(true);
    try {
      const JSZip = (await import("jszip")).default;
      const zip = new JSZip();
      await Promise.all(
        paths.map(async (path, i) => {
          const response = await fetch(staticUrl(path));
          const blob = await response.blob();
          zip.file(path.split("/").pop() ?? `frame-${i}.png`, blob);
        })
      );
      const zipBlob = await zip.generateAsync({ type: "blob" });
      const url = URL.createObjectURL(zipBlob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `aegistrace-frames-${result.id.slice(0, 8)}.zip`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setExporting(false);
    }
  }

  return (
    <Button variant="secondary" className="w-full" onClick={exportFrames} disabled={exporting || paths.length === 0}>
      {exporting ? <Spinner className="h-4 w-4" /> : <Download className="h-4 w-4" />}
      Export frames
    </Button>
  );
}
