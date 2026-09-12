"use client";

import { CompareSlider } from "@/components/CompareSlider";
import { ConfidenceGauge } from "@/components/ConfidenceGauge";
import { GradCamPanel, type EvidenceFrame } from "@/components/GradCamPanel";
import { LineMetricChart } from "@/components/MetricChart";
import { ModelSelector } from "@/components/ModelSelector";
import { PageHeader } from "@/components/PageHeader";
import { QuickActions } from "@/components/QuickActions";
import { ErrorState } from "@/components/QueryState";
import { SampleGallery } from "@/components/SampleGallery";
import { StatsRow } from "@/components/StatsRow";
import { SuspiciousFramesGrid } from "@/components/SuspiciousFramesGrid";
import { Section } from "@/components/ui/Section";
import { Spinner } from "@/components/ui/Spinner";
import { UploadWidget } from "@/components/UploadWidget";
import { api, staticUrl } from "@/lib/api-client";
import { cn } from "@/lib/cn";
import type { PredictionOut } from "@/lib/types";
import { useMutation } from "@tanstack/react-query";
import { ImageIcon, ScanSearch, ShieldCheck } from "lucide-react";
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
  const [fileInfo, setFileInfo] = useState<{ name: string; size: number } | null>(null);
  const [modelRunId, setModelRunId] = useState<string | null>(null);
  const [selectedFrameIndex, setSelectedFrameIndex] = useState(0);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const mutation = useMutation({
    mutationFn: (file: File) =>
      mode === "image" ? api.detectImage(file, modelRunId) : api.detectVideo(file, modelRunId),
    onSuccess: (data) => {
      setResult(data);
      setSelectedFrameIndex(0);
    },
  });

  function handleFile(file: File) {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(file));
    setFileInfo({ name: file.name, size: file.size });
    mutation.mutate(file);
  }

  function reset(newMode: "image" | "video" = mode) {
    setMode(newMode);
    mutation.reset();
    setResult(null);
    setSelectedFrameIndex(0);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setFileInfo(null);
  }

  const activeSuspiciousFrame = result?.input_type === "video" ? result.suspicious_frames[selectedFrameIndex] : undefined;

  const activeFrame: EvidenceFrame | null = !result
    ? null
    : result.input_type === "image"
      ? {
          originalPath: result.original_path,
          heatmapOnlyPath: result.heatmap_only_path,
          overlayPath: result.heatmap_path,
        }
      : activeSuspiciousFrame
        ? {
            originalPath: activeSuspiciousFrame.original_path,
            heatmapOnlyPath: activeSuspiciousFrame.heatmap_only_path,
            overlayPath: activeSuspiciousFrame.heatmap_path,
          }
        : null;

  const overlayForSlider = result
    ? result.input_type === "image"
      ? result.heatmap_path
      : (activeSuspiciousFrame?.heatmap_path ?? null)
    : null;

  return (
    <div>
      <PageHeader
        eyebrow="Analysis Workspace"
        title="Detection"
        description="Upload an image or video to run it through the currently configured detector."
        action={<ModelSelector value={modelRunId} onChange={setModelRunId} />}
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
          <div className="space-y-5">
            <UploadWidget
              key={mode}
              accept={mode === "image" ? "image/*" : "video/*"}
              disabled={mutation.isPending}
              label={mode === "image" ? "Upload an image" : "Upload a video"}
              hint={mode === "image" ? IMAGE_FORMATS : VIDEO_FORMATS}
              onFileSelected={handleFile}
            />
            {mode === "image" && <SampleGallery disabled={mutation.isPending} onSelect={handleFile} />}
          </div>
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
        <div className="animate-fade-slide-up space-y-6">
          <Section
            title="Media"
            action={
              <button
                type="button"
                onClick={() => reset()}
                className="text-[12px] font-medium text-primary hover:underline"
              >
                Change file
              </button>
            }
          >
            {fileInfo && (
              <div className="mono-value mb-3 text-[11.5px] text-muted-foreground">
                {fileInfo.name} · {(fileInfo.size / (1024 * 1024)).toFixed(1)} MB
                {result.video_width && result.video_height ? ` · ${result.video_width}×${result.video_height}` : ""}
                {result.duration_seconds != null ? ` · ${result.duration_seconds.toFixed(1)}s` : ""}
              </div>
            )}
            {overlayForSlider ? (
              <CompareSlider beforeSrc={previewUrl} afterSrc={staticUrl(overlayForSlider)} />
            ) : (
              <div className="overflow-hidden rounded-lg border border-border bg-muted/30">
                {result.input_type === "video" ? (
                  <video src={previewUrl} controls className="max-h-[420px] w-full bg-black" />
                ) : (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={previewUrl} alt="Uploaded media" className="max-h-[420px] w-full object-contain" />
                )}
              </div>
            )}
          </Section>

          <StatsRow
            framesAnalyzed={result.n_frames_analyzed}
            framesTotal={result.n_frames_total}
            facesDetected={result.n_faces_detected}
            avgConfidence={result.confidence}
            processingTimeMs={result.processing_time_ms}
          />

          <div className="grid gap-8 border-t border-border pt-6 lg:grid-cols-[1fr_1fr]">
            <div>
              <div className="meta-label mb-3 text-[10px]">Model Evidence</div>
              <GradCamPanel frame={activeFrame} />
              {result.input_type === "video" && result.suspicious_frames.length > 0 && (
                <div className="mt-5">
                  <div className="meta-label mb-2 text-[10px]">
                    Suspicious Frames (Top {result.suspicious_frames.length})
                  </div>
                  <SuspiciousFramesGrid
                    frames={result.suspicious_frames}
                    selectedIndex={selectedFrameIndex}
                    onSelect={setSelectedFrameIndex}
                  />
                </div>
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
            </div>
          </div>

          {result.frame_scores.length > 0 && (
            <Section
              title="Frame-level Analysis"
              description="Per-frame deepfake probability across sampled frames, against the 0.5 decision threshold."
            >
              <LineMetricChart
                data={result.frame_scores.map((s) => ({
                  t: s.timestamp.toFixed(1),
                  "Deepfake probability": s.fake_probability,
                }))}
                series={["Deepfake probability"]}
                xKey="t"
                yLabel="probability"
                referenceLine={{ y: 0.5, label: "Threshold (0.5)" }}
              />
            </Section>
          )}

          <Section>
            <QuickActions result={result} onRunNew={() => reset()} />
          </Section>
        </div>
      )}
    </div>
  );
}
