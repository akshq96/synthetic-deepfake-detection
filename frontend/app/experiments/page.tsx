"use client";

import { ExperimentStatusBadge } from "@/components/ExperimentStatusBadge";
import { PageHeader } from "@/components/PageHeader";
import { EmptyState, ErrorState, LoadingState } from "@/components/QueryState";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/Table";
import { api } from "@/lib/api-client";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";

interface EnrichedRun {
  id: string;
  runName: string;
  experimentName: string;
  status: string;
  model: string | null;
  dataset: string | null;
  syntheticRatio: string | null;
  f1: number | null;
  rocAuc: number | null;
  createdAt: string;
}

function datasetFromManifestPath(path: string | undefined): string | null {
  if (!path) return null;
  const file = path.split("/").pop() ?? path;
  return file.replace(/\.(parquet|csv)$/i, "");
}

export default function ExperimentsPage() {
  const experimentsQuery = useQuery({ queryKey: ["experiments"], queryFn: api.listExperiments });
  const mlflowQuery = useQuery({ queryKey: ["mlflow-runs-all"], queryFn: () => api.listMlflowRuns() });

  const isLoading = experimentsQuery.isLoading || mlflowQuery.isLoading;
  const isError = experimentsQuery.isError || mlflowQuery.isError;

  const rows: EnrichedRun[] =
    experimentsQuery.data?.flatMap((exp) =>
      exp.runs.map((run) => {
        const mlflowRun = mlflowQuery.data?.find((m) => m.run_id === run.mlflow_run_id);
        return {
          id: run.id,
          runName: run.run_name,
          experimentName: exp.name,
          status: run.status,
          model: mlflowRun?.params["model.name"] ?? null,
          dataset: datasetFromManifestPath(mlflowRun?.params["data.manifest_path"]),
          syntheticRatio: mlflowRun?.params["data.synthetic_ratio"] ?? null,
          f1: mlflowRun?.metrics["val_f1"] ?? null,
          rocAuc: mlflowRun?.metrics["val_roc_auc"] ?? null,
          createdAt: run.created_at,
        };
      })
    ) ?? [];

  return (
    <div>
      <PageHeader
        eyebrow="Records"
        title="Experiment History"
        description="Every training and evaluation run launched through this application, with the metrics and configuration logged to MLflow at completion."
      />

      {isLoading && <LoadingState />}
      {isError && (
        <ErrorState
          error={experimentsQuery.error ?? mlflowQuery.error}
          title="Unable to load experiment history"
          onRetry={() => {
            experimentsQuery.refetch();
            mlflowQuery.refetch();
          }}
        />
      )}
      {experimentsQuery.isSuccess && rows.length === 0 && (
        <EmptyState
          title="No experiments recorded"
          description="Training and evaluation runs will appear here once you launch a research experiment."
          action={
            <Link href="/synthetic-lab" className="text-[12.5px] font-medium text-primary hover:underline">
              Launch an experiment →
            </Link>
          }
        />
      )}

      {rows.length > 0 && (
        <Table>
          <THead>
            <TR>
              <TH>Run</TH>
              <TH>Experiment</TH>
              <TH>Model</TH>
              <TH>Dataset</TH>
              <TH>Synthetic ratio</TH>
              <TH>Status</TH>
              <TH>F1</TH>
              <TH>ROC-AUC</TH>
              <TH>Date</TH>
            </TR>
          </THead>
          <TBody>
            {rows.map((row) => (
              <TR key={row.id}>
                <TD className="font-medium">{row.runName}</TD>
                <TD className="text-muted-foreground">{row.experimentName}</TD>
                <TD className="mono-value">{row.model ?? "—"}</TD>
                <TD className="mono-value">{row.dataset ?? "—"}</TD>
                <TD className="mono-value">{row.syntheticRatio ?? "—"}</TD>
                <TD>
                  <ExperimentStatusBadge status={row.status} />
                </TD>
                <TD className="mono-value">{row.f1 != null ? row.f1.toFixed(3) : "—"}</TD>
                <TD className="mono-value">{row.rocAuc != null ? row.rocAuc.toFixed(3) : "—"}</TD>
                <TD className="mono-value text-muted-foreground">
                  {new Date(row.createdAt).toLocaleDateString()}
                </TD>
              </TR>
            ))}
          </TBody>
        </Table>
      )}
    </div>
  );
}
