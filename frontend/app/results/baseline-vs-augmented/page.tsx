"use client";

import { ComparisonTable, type ComparisonColumn } from "@/components/ComparisonTable";
import { LineMetricChart } from "@/components/MetricChart";
import { PageHeader } from "@/components/PageHeader";
import { EmptyState, ErrorState, LoadingState } from "@/components/QueryState";
import { ResultsLookupForm } from "@/components/ResultsLookupForm";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { api } from "@/lib/api-client";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

interface PointRow {
  id: string;
  point: string;
  accuracy: number;
  f1: number;
  roc_auc: number;
}

const COLUMNS: ComparisonColumn<PointRow>[] = [
  { key: "point", label: "Grid point" },
  { key: "accuracy", label: "Accuracy", format: (v) => Number(v).toFixed(3) },
  { key: "f1", label: "F1", format: (v) => Number(v).toFixed(3) },
  { key: "roc_auc", label: "ROC-AUC", format: (v) => Number(v).toFixed(3) },
];

export default function BaselineVsAugmentedPage() {
  const [resultsDir, setResultsDir] = useState<string | null>(null);

  const query = useQuery({
    queryKey: ["baseline-vs-synthetic", resultsDir],
    queryFn: () => api.getBaselineVsSynthetic(resultsDir!),
    enabled: !!resultsDir,
  });

  const rows: PointRow[] =
    query.data?.map((entry, i) => ({
      id: String(i),
      point: JSON.stringify(entry.point),
      accuracy: entry.eval_metrics.accuracy,
      f1: entry.eval_metrics.f1,
      roc_auc: entry.eval_metrics.roc_auc,
    })) ?? [];

  const chartData =
    query.data?.map((entry) => {
      const [firstKey] = Object.keys(entry.point);
      return {
        ratio: firstKey ? entry.point[firstKey] : 0,
        Accuracy: entry.eval_metrics.accuracy,
        F1: entry.eval_metrics.f1,
      };
    }) ?? [];

  return (
    <div>
      <PageHeader
        title="Baseline vs. Synthetic-Augmented"
        description="Results from an ablation sweep over synthetic_ratio — this is the core baseline-vs-augmented comparison."
      />

      <ResultsLookupForm
        label="Results directory (relative to artifacts/)"
        placeholder="experiments/ablation/2026-01-01T00-00-00"
        onSubmit={setResultsDir}
      />

      {query.isLoading && <LoadingState />}
      {query.isError && <ErrorState error={query.error} />}
      {query.isSuccess && rows.length === 0 && <EmptyState label="No grid points found in this results file." />}

      {rows.length > 0 && (
        <div className="grid gap-4">
          <Card>
            <CardHeader>
              <CardTitle>Accuracy / F1 vs. synthetic ratio</CardTitle>
              <CardDescription>
                A flat or declining line here would indicate synthetic augmentation does not help at this
                grid&apos;s scale — the plot answers the question, it doesn&apos;t assume it.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <LineMetricChart data={chartData} series={["Accuracy", "F1"]} xKey="ratio" />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Raw results</CardTitle>
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
