"use client";

import { ConfidenceGauge } from "@/components/ConfidenceGauge";
import { HeatmapOverlay } from "@/components/HeatmapOverlay";
import { PageHeader } from "@/components/PageHeader";
import { ErrorState } from "@/components/QueryState";
import { Section } from "@/components/ui/Section";
import { SuspiciousFrameStrip } from "@/components/SuspiciousFrameStrip";
import { Button } from "@/components/ui/Button";
import { Spinner } from "@/components/ui/Spinner";
import { UploadWidget } from "@/components/UploadWidget";
import { cn } from "@/lib/cn";
import { api } from "@/lib/api-client";
import type { PredictionOut } from "@/lib/types";
import { useMutation } from "@tanstack/react-query";
import { FileText, ImageIcon, ScanSearch, ShieldCheck, ArrowRight } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

const IMAGE_FORMATS = "JPG · PNG · WEBP";
const VIDEO_FORMATS = "MP4 · AVI · MOV · MKV · WEBM";

const WORKFLOW = [
  { icon: ImageIcon, text: "Upload a media file" },
  { icon: ScanSearch, text: "The model scores it, frame by frame for video" },
  { icon: ShieldCheck, text: "Get a calibrated verdict + visual evidence" },
];

export default function DetectPage() {
  const [mode, setMode] = useState<"image" | "video">("image");
  const [result, setResult] = useState<PredictionOut | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const mutation = useMutation({
    mutationFn: (file: File) => (mode === "image" ? api.detectImage(file) : api.detectVideo(file)),
    onSuccess: (data) => setResult(data),
  });

  function handleFile(file: File) {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(file));
    mutation.mutate(file);
  }

  function reset(newMode: "image" | "video") {
    setMode(newMode);
    mutation.reset();
    setResult(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
  }

  return (
    <div>
      <PageHeader
        eyebrow="Analysis Workspace"
        title="Detection"
        description="Upload an image or video to run it through the currently configured detector."
      />

      <div className="mb-6 flex gap-1 border-b border-border">
        {(["image", "video"] as const).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => reset(m)}
            className={cn(
              "-mb-px border-b-2 px-1 py-2.5 text-[13px] font-medium capitalize transition-colors",
              mode === m
                ? "border-primary text-foreground"
                : "border-transparent text-muted-foreground hover:text-foreground"
            )}
          >
            {m}
          </button>
        ))}
      </div>

      {!result && !mutation.isPending && (
        <div className="grid gap-8 lg:grid-cols-[1.3fr_1fr]">
          <UploadWidget
            key={mode}
            accept={mode === "image" ? "image/*" : "video/*"}
            disabled={mutation.isPending}
            label={mode === "image" ? "Upload an image" : "Upload a video"}
            hint={mode === "image" ? IMAGE_FORMATS : VIDEO_FORMATS}
            onFileSelected={handleFile}
          />
          <div>
            <div className="meta-label mb-3 text-[10px]">What happens next</div>
            <ol className="space-y-3.5">
              {WORKFLOW.map((step, i) => (
                <li key={step.text} className="flex items-start gap-3">
                  <span className="mono-value mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-border-strong text-[10px] text-muted-foreground">
                    {i + 1}
                  </span>
                  <span className="text-[13px] text-foreground-secondary">{step.text}</span>
                </li>
              ))}
            </ol>
          </div>
        </div>
      )}

      {mutation.isPending && (
        <div className="flex items-center gap-2.5 rounded-lg border border-border bg-muted/50 px-5 py-5 text-[13px] text-muted-foreground">
          <Spinner />
          Running detection — video takes longer (frame sampling + face tracking)…
        </div>
      )}

      {mutation.isError && <ErrorState error={mutation.error} onRetry={() => setResult(null)} />}

      {result && previewUrl && (
        <div>
          <Section title="Media">
            <div className="overflow-hidden rounded-lg border border-border bg-muted/30">
              {result.input_type === "video" ? (
                <video src={previewUrl} controls className="max-h-[420px] w-full bg-black" />
              ) : (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={previewUrl} alt="Uploaded media" className="max-h-[420px] w-full object-contain" />
              )}
            </div>
          </Section>

          <div className="grid gap-8 border-t border-border pt-6 lg:grid-cols-[1fr_1fr]">
            <div>
              <div className="meta-label mb-3 text-[10px]">Model Evidence</div>
              {result.input_type === "video" ? (
                <SuspiciousFrameStrip frames={result.suspicious_frames} />
              ) : (
                <HeatmapOverlay heatmapPath={result.heatmap_path} className="aspect-video w-full" />
              )}
            </div>

            <div>
              <div className="meta-label mb-3 text-[10px]">Analysis</div>
              <ConfidenceGauge
                label={result.label}
                confidence={result.confidence}
                fakeProbability={result.fake_probability}
              />
              <dl className="mono-value mt-5 space-y-1.5 border-t border-border pt-4 text-[11.5px] text-muted-foreground">
                <div className="flex justify-between">
                  <dt>Model</dt>
                  <dd className="text-foreground">{result.model_name}</dd>
                </div>
                <div className="flex justify-between">
                  <dt>Input type</dt>
                  <dd className="text-foreground">{result.input_type}</dd>
                </div>
                {result.abstained && (
                  <div className="flex justify-between">
                    <dt>Note</dt>
                    <dd className="text-warning">Below confidence threshold — abstained</dd>
                  </div>
                )}
              </dl>
              <GenerateReportButton predictionId={result.id} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function GenerateReportButton({ predictionId }: { predictionId: string }) {
  const [reportId, setReportId] = useState<string | null>(null);
  const mutation = useMutation({
    mutationFn: () => api.generateReport(predictionId),
    onSuccess: (report) => setReportId(report.id),
  });

  if (reportId) {
    return (
      <Link href={`/reports/${reportId}`} className="mt-5 block">
        <Button variant="secondary" className="w-full">
          <FileText className="h-4 w-4" />
          View forensic report
          <ArrowRight className="h-3.5 w-3.5" />
        </Button>
      </Link>
    );
  }

  return (
    <Button
      variant="secondary"
      className="mt-5 w-full"
      onClick={() => mutation.mutate()}
      disabled={mutation.isPending}
    >
      {mutation.isPending ? <Spinner className="h-4 w-4" /> : <FileText className="h-4 w-4" />}
      Generate forensic report
    </Button>
  );
}
