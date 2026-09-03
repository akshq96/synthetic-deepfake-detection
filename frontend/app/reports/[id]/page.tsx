"use client";

import { ConfidenceGauge } from "@/components/ConfidenceGauge";
import { PageHeader } from "@/components/PageHeader";
import { ErrorState, LoadingState } from "@/components/QueryState";
import { Button } from "@/components/ui/Button";
import { api } from "@/lib/api-client";
import { useQuery } from "@tanstack/react-query";
import { Download } from "lucide-react";
import { useParams } from "next/navigation";

export default function ReportDetailPage() {
  // A client hook (not the page's `params` prop) sidesteps Next.js 16's
  // async-params requirement for Server Component pages — this page is
  // entirely client-rendered anyway (it talks to the FastAPI backend
  // directly), so there's no server-side params access to await.
  const params = useParams<{ id: string }>();
  const reportId = params.id;

  const reportQuery = useQuery({
    queryKey: ["report", reportId],
    queryFn: () => api.getReport(reportId),
  });
  const predictionQuery = useQuery({
    queryKey: ["prediction", reportQuery.data?.prediction_id],
    queryFn: () => api.getPrediction(reportQuery.data!.prediction_id),
    enabled: !!reportQuery.data,
  });

  if (reportQuery.isLoading) return <LoadingState />;
  if (reportQuery.isError) {
    return <ErrorState error={reportQuery.error} title="Unable to load this report" onRetry={() => reportQuery.refetch()} />;
  }
  if (!reportQuery.data) return null;

  return (
    <div>
      <PageHeader
        eyebrow="Forensic Report"
        title={`Case ${reportQuery.data.id.slice(0, 8)}`}
        description={new Date(reportQuery.data.created_at).toLocaleString()}
        action={
          <a href={api.reportExportUrl(reportQuery.data.id)} target="_blank" rel="noreferrer">
            <Button variant="secondary">
              <Download className="h-4 w-4" />
              Open PDF
            </Button>
          </a>
        }
      />

      <div className="grid gap-8 md:grid-cols-[1fr_1.4fr]">
        <div>
          <div className="meta-label mb-3 text-[10px]">Prediction Summary</div>
          {predictionQuery.isLoading && <LoadingState />}
          {predictionQuery.data && (
            <ConfidenceGauge
              label={predictionQuery.data.label}
              confidence={predictionQuery.data.confidence}
              fakeProbability={predictionQuery.data.fake_probability}
            />
          )}
        </div>

        <div>
          <div className="meta-label mb-3 text-[10px]">PDF Preview</div>
          <iframe
            src={api.reportExportUrl(reportQuery.data.id)}
            className="h-[420px] w-full rounded-lg border border-border"
            title="Forensic report PDF"
          />
        </div>
      </div>
    </div>
  );
}
