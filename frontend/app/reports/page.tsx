"use client";

import { PageHeader } from "@/components/PageHeader";
import { EmptyState, ErrorState, LoadingState } from "@/components/QueryState";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/Table";
import { api, staticUrl } from "@/lib/api-client";
import { useQueries, useQuery } from "@tanstack/react-query";
import { FileText } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";

export default function ReportsPage() {
  const router = useRouter();
  const reportsQuery = useQuery({ queryKey: ["reports"], queryFn: api.listReports });

  const predictionQueries = useQueries({
    queries: (reportsQuery.data ?? []).map((report) => ({
      queryKey: ["prediction", report.prediction_id],
      queryFn: () => api.getPrediction(report.prediction_id),
      enabled: !!reportsQuery.data,
    })),
  });

  return (
    <div>
      <PageHeader
        eyebrow="Records"
        title="Forensic Reports"
        description="A case archive of generated forensic reports — each one produced from the Detection page after analyzing a piece of media."
      />

      {reportsQuery.isLoading && <LoadingState />}
      {reportsQuery.isError && (
        <ErrorState
          error={reportsQuery.error}
          title="Unable to load forensic reports"
          onRetry={() => reportsQuery.refetch()}
        />
      )}
      {reportsQuery.isSuccess && reportsQuery.data.length === 0 && (
        <EmptyState
          title="No forensic reports yet"
          description="Reports are generated automatically after completing a media analysis."
          action={
            <Link
              href="/detect"
              className="inline-flex items-center gap-1.5 text-[12.5px] font-medium text-primary hover:underline"
            >
              <FileText className="h-3.5 w-3.5" />
              Analyze media
            </Link>
          }
        />
      )}

      {reportsQuery.data && reportsQuery.data.length > 0 && (
        <Table>
          <THead>
            <TR>
              <TH>Media</TH>
              <TH>Verdict</TH>
              <TH>Confidence</TH>
              <TH>Model</TH>
              <TH>Date</TH>
              <TH />
            </TR>
          </THead>
          <TBody>
            {reportsQuery.data.map((report, i) => {
              const prediction = predictionQueries[i]?.data;
              return (
                <TR key={report.id} onClick={() => router.push(`/reports/${report.id}`)}>
                  <TD>
                    <div className="flex h-9 w-9 items-center justify-center overflow-hidden rounded border border-border bg-muted">
                      {prediction?.heatmap_path ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                          src={staticUrl(prediction.heatmap_path)}
                          alt=""
                          className="h-full w-full object-cover"
                        />
                      ) : (
                        <FileText className="h-4 w-4 text-muted-foreground" strokeWidth={1.5} />
                      )}
                    </div>
                  </TD>
                  <TD>
                    {prediction ? (
                      <span
                        className={
                          prediction.label === "fake"
                            ? "text-danger"
                            : prediction.label === "real"
                              ? "text-success"
                              : "text-warning"
                        }
                      >
                        {prediction.label === "fake" ? "Deepfake" : prediction.label === "real" ? "Real" : "Uncertain"}
                      </span>
                    ) : (
                      "—"
                    )}
                  </TD>
                  <TD className="mono-value">
                    {prediction ? `${(prediction.confidence * 100).toFixed(1)}%` : "—"}
                  </TD>
                  <TD className="mono-value text-muted-foreground">{prediction?.model_name ?? "—"}</TD>
                  <TD className="mono-value text-muted-foreground">
                    {new Date(report.created_at).toLocaleDateString()}
                  </TD>
                  <TD>
                    <Link
                      href={`/reports/${report.id}`}
                      className="text-[12.5px] font-medium text-primary hover:underline"
                      onClick={(e) => e.stopPropagation()}
                    >
                      View
                    </Link>
                  </TD>
                </TR>
              );
            })}
          </TBody>
        </Table>
      )}
    </div>
  );
}
