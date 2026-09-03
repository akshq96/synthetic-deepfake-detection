"use client";

import { ConfidenceGauge } from "@/components/ConfidenceGauge";
import { PageHeader } from "@/components/PageHeader";
import { ErrorState, LoadingState } from "@/components/QueryState";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
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
  if (reportQuery.isError) return <ErrorState error={reportQuery.error} />;
  if (!reportQuery.data) return null;

  return (
    <div>
      <PageHeader
        title={`Report ${reportQuery.data.id.slice(0, 8)}`}
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

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Prediction summary</CardTitle>
          </CardHeader>
          <CardContent>
            {predictionQuery.isLoading && <LoadingState />}
            {predictionQuery.data && (
              <ConfidenceGauge
                label={predictionQuery.data.label}
                confidence={predictionQuery.data.confidence}
                fakeProbability={predictionQuery.data.fake_probability}
              />
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>PDF preview</CardTitle>
          </CardHeader>
          <CardContent>
            <iframe
              src={api.reportExportUrl(reportQuery.data.id)}
              className="h-96 w-full rounded-lg border border-border"
              title="Forensic report PDF"
            />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
