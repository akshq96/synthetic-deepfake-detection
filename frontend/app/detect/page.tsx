"use client";

import { ConfidenceGauge } from "@/components/ConfidenceGauge";
import { HeatmapOverlay } from "@/components/HeatmapOverlay";
import { PageHeader } from "@/components/PageHeader";
import { ErrorState } from "@/components/QueryState";
import { SuspiciousFrameStrip } from "@/components/SuspiciousFrameStrip";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Spinner } from "@/components/ui/Spinner";
import { UploadWidget } from "@/components/UploadWidget";
import { cn } from "@/lib/cn";
import { api } from "@/lib/api-client";
import type { PredictionOut } from "@/lib/types";
import { useMutation } from "@tanstack/react-query";
import { FileText } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

export default function DetectPage() {
  const [mode, setMode] = useState<"image" | "video">("image");
  const [result, setResult] = useState<PredictionOut | null>(null);

  const mutation = useMutation({
    mutationFn: (file: File) => (mode === "image" ? api.detectImage(file) : api.detectVideo(file)),
    onSuccess: (data) => setResult(data),
  });

  return (
    <div>
      <PageHeader
        title="Detection"
        description="Upload an image or video to run it through the currently configured model."
      />

      <div className="mb-6 inline-flex gap-1 rounded-lg bg-muted p-1">
        {(["image", "video"] as const).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => {
              setMode(m);
              mutation.reset();
              setResult(null);
            }}
            className={cn(
              "rounded-md px-3 py-1.5 text-sm font-medium capitalize transition-colors",
              mode === m ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
            )}
          >
            {m}
          </button>
        ))}
      </div>

      <div className="mb-6">
        <UploadWidget
          key={mode}
          accept={mode === "image" ? "image/*" : "video/*"}
          disabled={mutation.isPending}
          label={
            mode === "image"
              ? "Upload an image (JPG, PNG, WEBP)"
              : "Upload a video (MP4, AVI, MOV, MKV, WEBM)"
          }
          onFileSelected={(file) => mutation.mutate(file)}
        />
      </div>

      {mutation.isPending && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Spinner />
          Running detection — this can take a little longer for video (frame sampling + tracking)…
        </div>
      )}

      {mutation.isError && <ErrorState error={mutation.error} />}

      {result && (
        <div className="grid gap-4 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Prediction</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <ConfidenceGauge
                label={result.label}
                confidence={result.confidence}
                fakeProbability={result.fake_probability}
              />
              <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                <Badge variant="muted">model: {result.model_name}</Badge>
                <Badge variant="muted">type: {result.input_type}</Badge>
                {result.abstained && <Badge variant="warning">abstained</Badge>}
              </div>
              <GenerateReportButton predictionId={result.id} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>{result.input_type === "video" ? "Suspicious frames" : "Explainability heatmap"}</CardTitle>
            </CardHeader>
            <CardContent>
              {result.input_type === "video" ? (
                <SuspiciousFrameStrip frames={result.suspicious_frames} />
              ) : (
                <HeatmapOverlay heatmapPath={result.heatmap_path} className="aspect-square w-full" />
              )}
            </CardContent>
          </Card>
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
      <Link href={`/reports/${reportId}`}>
        <Button variant="secondary" className="w-full">
          <FileText className="h-4 w-4" />
          View forensic report
        </Button>
      </Link>
    );
  }

  return (
    <Button variant="secondary" className="w-full" onClick={() => mutation.mutate()} disabled={mutation.isPending}>
      {mutation.isPending ? <Spinner className="h-4 w-4" /> : <FileText className="h-4 w-4" />}
      Generate forensic report
    </Button>
  );
}
