"use client";

import { ExperimentStatusBadge } from "@/components/ExperimentStatusBadge";
import { PageHeader } from "@/components/PageHeader";
import { EmptyState, ErrorState, LoadingState } from "@/components/QueryState";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/Table";
import { api } from "@/lib/api-client";
import { useQuery } from "@tanstack/react-query";

export default function ExperimentsPage() {
  const query = useQuery({ queryKey: ["experiments"], queryFn: api.listExperiments });

  return (
    <div>
      <PageHeader
        title="Experiment History"
        description="Every training/experiment run launched through this app, tracked by the backend."
      />

      {query.isLoading && <LoadingState />}
      {query.isError && <ErrorState error={query.error} />}
      {query.isSuccess && query.data.length === 0 && (
        <EmptyState label="No experiments yet — launch one from the Synthetic Data Lab." />
      )}

      <div className="space-y-4">
        {query.data?.map((experiment) => (
          <Card key={experiment.id}>
            <CardHeader>
              <CardTitle>{experiment.name}</CardTitle>
              {experiment.description && (
                <p className="text-xs text-muted-foreground">{experiment.description}</p>
              )}
            </CardHeader>
            <CardContent>
              <Table>
                <THead>
                  <TR>
                    <TH>Run</TH>
                    <TH>Status</TH>
                    <TH>MLflow run</TH>
                    <TH>Started</TH>
                    <TH>Finished</TH>
                  </TR>
                </THead>
                <TBody>
                  {experiment.runs.map((run) => (
                    <TR key={run.id}>
                      <TD className="font-medium">{run.run_name}</TD>
                      <TD>
                        <ExperimentStatusBadge status={run.status} />
                      </TD>
                      <TD>
                        {run.mlflow_run_id ? <code className="text-xs">{run.mlflow_run_id}</code> : "—"}
                      </TD>
                      <TD className="text-xs text-muted-foreground">
                        {run.started_at ? new Date(run.started_at).toLocaleString() : "—"}
                      </TD>
                      <TD className="text-xs text-muted-foreground">
                        {run.finished_at ? new Date(run.finished_at).toLocaleString() : "—"}
                      </TD>
                    </TR>
                  ))}
                </TBody>
              </Table>
              {experiment.runs.length === 0 && (
                <p className="py-4 text-center text-xs text-muted-foreground">No runs in this experiment yet.</p>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
