"use client";

import { LineMetricChart } from "@/components/MetricChart";
import { PageHeader } from "@/components/PageHeader";
import { EmptyState, ErrorState, LoadingState } from "@/components/QueryState";
import { ResultsLookupForm } from "@/components/ResultsLookupForm";
import { Badge } from "@/components/ui/Badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { api } from "@/lib/api-client";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle } from "lucide-react";
import { useState } from "react";

export default function RobustnessPage() {
  const [resultsDir, setResultsDir] = useState<string | null>(null);
  const query = useQuery({
    queryKey: ["robustness", resultsDir],
    queryFn: () => api.getRobustness(resultsDir!),
    enabled: !!resultsDir,
  });

  const perturbationNames = query.data ? Object.keys(query.data.perturbations) : [];

  return (
    <div>
      <PageHeader
        title="Robustness"
        description="Accuracy/AUC degradation under increasing perturbation severity (Gaussian noise, blur, JPEG compression)."
      />

      <ResultsLookupForm
        label="Results directory (relative to artifacts/)"
        placeholder="experiments/robustness/2026-01-01T00-00-00"
        onSubmit={setResultsDir}
      />

      {query.isLoading && <LoadingState />}
      {query.isError && <ErrorState error={query.error} />}
      {query.isSuccess && perturbationNames.length === 0 && (
        <EmptyState label="No perturbation curves found in this results file." />
      )}

      {query.data && perturbationNames.length > 0 && (
        <div className="grid gap-4">
          {query.data.compression_in_training_mix && (
            <div className="flex items-start gap-3 rounded-lg border border-warning/30 bg-warning/5 p-4 text-sm">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
              <div>
                <strong>Compression was part of this checkpoint&apos;s synthetic training mix.</strong> Its
                apparent robustness to compression below is an expected, controlled result — not evidence of
                general robustness — since the model was shown compression artifacts during training.
              </div>
            </div>
          )}

          <Card>
            <CardHeader>
              <CardTitle>Accuracy vs. severity</CardTitle>
              <CardDescription>Baseline (unperturbed) accuracy: {query.data.baseline_metrics.accuracy.toFixed(3)}</CardDescription>
            </CardHeader>
            <CardContent>
              <LineMetricChart
                data={mergeSeverityCurves(query.data.perturbations, "accuracy")}
                series={perturbationNames}
                xKey="severity"
                yLabel="accuracy"
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>ROC-AUC vs. severity</CardTitle>
            </CardHeader>
            <CardContent>
              <LineMetricChart
                data={mergeSeverityCurves(query.data.perturbations, "roc_auc")}
                series={perturbationNames}
                xKey="severity"
                yLabel="roc_auc"
              />
            </CardContent>
          </Card>

          <div className="flex flex-wrap gap-2">
            {perturbationNames.map((name) => (
              <Badge key={name} variant="muted">
                {name}
              </Badge>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Perturbation curves each have their own severity axis (e.g. JPEG quality
// 80/50/30/15 vs. noise sigma 5/15/30/50) — for a shared chart, index by
// position rather than raw severity value so series line up as "1st/2nd/3rd
// most severe point" rather than literal x-values.
function mergeSeverityCurves(
  perturbations: Record<string, { severity: number; metrics: { accuracy: number; roc_auc: number } }[]>,
  metric: "accuracy" | "roc_auc"
) {
  const maxLen = Math.max(...Object.values(perturbations).map((c) => c.length));
  return Array.from({ length: maxLen }, (_, i) => {
    const row: Record<string, number | string> = { severity: `step ${i + 1}` };
    for (const [name, curve] of Object.entries(perturbations)) {
      if (curve[i]) row[name] = curve[i].metrics[metric];
    }
    return row;
  });
}
