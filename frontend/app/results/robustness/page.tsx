"use client";

import { LineMetricChart } from "@/components/MetricChart";
import { PageHeader } from "@/components/PageHeader";
import { EmptyState, ErrorState, LoadingState } from "@/components/QueryState";
import { ResultsLookupForm } from "@/components/ResultsLookupForm";
import { Section } from "@/components/ui/Section";
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
        eyebrow="Degradation Testing"
        title="Robustness"
        description="Accuracy and ROC-AUC as perturbation severity increases (Gaussian noise, blur, JPEG compression) — a stable line means the model tolerates that degradation; a steep drop means it doesn't."
      />

      <ResultsLookupForm
        label="Results directory (relative to artifacts/)"
        placeholder="experiments/robustness/2026-01-01T00-00-00"
        onSubmit={setResultsDir}
      />

      {query.isLoading && <LoadingState />}
      {query.isError && (
        <ErrorState error={query.error} title="Unable to load robustness results" onRetry={() => query.refetch()} />
      )}
      {query.isSuccess && perturbationNames.length === 0 && (
        <EmptyState
          title="No perturbation curves found"
          description="Run ml.evaluation.experiments.robustness, then load its results directory here."
        />
      )}

      {query.data && perturbationNames.length > 0 && (
        <div>
          {query.data.compression_in_training_mix && (
            <div className="mb-6 flex items-start gap-3 rounded-lg border border-warning/40 bg-warning-tint px-4 py-3.5 text-[13px]">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
              <div className="text-foreground-secondary">
                <strong className="text-foreground">
                  Compression was part of this checkpoint&apos;s synthetic training mix.
                </strong>{" "}
                Its apparent robustness to compression below is an expected, controlled result — not evidence of
                general robustness — since the model was shown compression artifacts during training.
              </div>
            </div>
          )}

          <Section
            title="Accuracy vs. severity"
            description={`Baseline (unperturbed / "Original") accuracy: ${query.data.baseline_metrics.accuracy.toFixed(3)}`}
          >
            <LineMetricChart
              data={mergeSeverityCurves(query.data.perturbations, "accuracy")}
              series={perturbationNames}
              xKey="severity"
              yLabel="accuracy"
            />
          </Section>

          <Section title="ROC-AUC vs. severity">
            <LineMetricChart
              data={mergeSeverityCurves(query.data.perturbations, "roc_auc")}
              series={perturbationNames}
              xKey="severity"
              yLabel="roc_auc"
            />
          </Section>
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
  perturbations: Record<string, { severity: number; metrics: { accuracy: number; roc_auc: number | null } }[]>,
  metric: "accuracy" | "roc_auc"
) {
  const maxLen = Math.max(...Object.values(perturbations).map((c) => c.length));
  return Array.from({ length: maxLen }, (_, i) => {
    const row: Record<string, number | string> = { severity: `step ${i + 1}` };
    for (const [name, curve] of Object.entries(perturbations)) {
      // Omit the key entirely when the metric is undefined (degenerate
      // single-class eval set) rather than plotting a fabricated 0 — a gap
      // in the line is honest, a dip to zero is not.
      const value = curve[i]?.metrics[metric];
      if (value != null) row[name] = value;
    }
    return row;
  });
}
