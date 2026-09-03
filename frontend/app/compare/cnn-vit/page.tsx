"use client";

import { ComparisonTable, type ComparisonColumn } from "@/components/ComparisonTable";
import { BarMetricChart } from "@/components/MetricChart";
import { PageHeader } from "@/components/PageHeader";
import { EmptyState, ErrorState, LoadingState } from "@/components/QueryState";
import { ResultsLookupForm } from "@/components/ResultsLookupForm";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { api } from "@/lib/api-client";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

interface ModelRow {
  id: string;
  model: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
}

const COLUMNS: ComparisonColumn<ModelRow>[] = [
  { key: "model", label: "Model" },
  { key: "accuracy", label: "Accuracy", format: (v) => Number(v).toFixed(3) },
  { key: "precision", label: "Precision", format: (v) => Number(v).toFixed(3) },
  { key: "recall", label: "Recall", format: (v) => Number(v).toFixed(3) },
  { key: "f1", label: "F1", format: (v) => Number(v).toFixed(3) },
  { key: "roc_auc", label: "ROC-AUC", format: (v) => Number(v).toFixed(3) },
];

export default function CompareCnnVitPage() {
  const [experimentName, setExperimentName] = useState<string | null>(null);

  const query = useQuery({
    queryKey: ["model-compare", experimentName],
    queryFn: () => api.compareModels(experimentName!),
    enabled: !!experimentName,
  });

  const rows: ModelRow[] = query.data
    ? Object.entries(query.data.models).map(([modelName, entry]) => ({
        id: modelName,
        model: modelName,
        accuracy: entry.metrics["val_accuracy"] ?? 0,
        precision: entry.metrics["val_precision"] ?? 0,
        recall: entry.metrics["val_recall"] ?? 0,
        f1: entry.metrics["val_f1"] ?? 0,
        roc_auc: entry.metrics["val_roc_auc"] ?? 0,
      }))
    : [];

  return (
    <div>
      <PageHeader
        title="CNN vs ViT"
        description="Most recent MLflow run per architecture within an experiment, compared side by side."
      />

      <ResultsLookupForm
        label="MLflow experiment name"
        placeholder="e.g. cnn_baseline"
        onSubmit={setExperimentName}
      />

      {query.isLoading && <LoadingState />}
      {query.isError && <ErrorState error={query.error} />}
      {query.isSuccess && rows.length === 0 && (
        <EmptyState label="No runs with a model.name param found in this experiment yet." />
      )}

      {rows.length > 0 && (
        <div className="grid gap-4">
          <Card>
            <CardHeader>
              <CardTitle>Validation metrics by architecture</CardTitle>
            </CardHeader>
            <CardContent>
              <BarMetricChart
                data={rows.map((r) => ({
                  model: r.model,
                  Accuracy: r.accuracy,
                  F1: r.f1,
                  "ROC-AUC": r.roc_auc,
                }))}
                series={["Accuracy", "F1", "ROC-AUC"]}
                xKey="model"
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Full comparison</CardTitle>
            </CardHeader>
            <CardContent>
              <ComparisonTable rows={rows} columns={COLUMNS} defaultSortKey="accuracy" />
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
