"use client";

import { ComparisonTable, type ComparisonColumn } from "@/components/ComparisonTable";
import { LineMetricChart } from "@/components/MetricChart";
import { PageHeader } from "@/components/PageHeader";
import { EmptyState, ErrorState, LoadingState } from "@/components/QueryState";
import { ResultsLookupForm } from "@/components/ResultsLookupForm";
import { Section } from "@/components/ui/Section";
import { api } from "@/lib/api-client";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Minus, TrendingDown, TrendingUp } from "lucide-react";
import { useState } from "react";

interface PointRow {
  id: string;
  point: string;
  ratio: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  // null when the eval set for this grid point had only one class present
  // (common at small/fixture scale) — ROC-AUC is undefined in that case,
  // not zero, so it must render as "—", never a fabricated number.
  roc_auc: number | null;
}

function formatMetric(v: unknown): string {
  return v == null ? "—" : Number(v).toFixed(3);
}

const COLUMNS: ComparisonColumn<PointRow>[] = [
  { key: "point", label: "Grid point" },
  { key: "accuracy", label: "Accuracy", format: formatMetric },
  { key: "precision", label: "Precision", format: formatMetric },
  { key: "recall", label: "Recall", format: formatMetric },
  { key: "f1", label: "F1", format: formatMetric },
  { key: "roc_auc", label: "ROC-AUC", format: formatMetric },
];

const METRIC_LABELS: { key: "accuracy" | "precision" | "recall" | "f1" | "roc_auc"; label: string }[] = [
  { key: "accuracy", label: "Accuracy" },
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "f1", label: "F1" },
  { key: "roc_auc", label: "ROC-AUC" },
];

export default function BaselineVsAugmentedPage() {
  const [resultsDir, setResultsDir] = useState<string | null>(null);

  const query = useQuery({
    queryKey: ["baseline-vs-synthetic", resultsDir],
    queryFn: () => api.getBaselineVsSynthetic(resultsDir!),
    enabled: !!resultsDir,
  });

  const rows: PointRow[] = (query.data ?? [])
    .map((entry, i) => {
      const [firstKey] = Object.keys(entry.point);
      const ratio = firstKey ? Number(entry.point[firstKey]) : 0;
      return {
        id: String(i),
        point: JSON.stringify(entry.point),
        ratio,
        accuracy: entry.eval_metrics.accuracy,
        precision: entry.eval_metrics.precision,
        recall: entry.eval_metrics.recall,
        f1: entry.eval_metrics.f1,
        roc_auc: entry.eval_metrics.roc_auc,
      };
    })
    .sort((a, b) => a.ratio - b.ratio);

  const baseline = rows[0];
  const augmented = rows[rows.length - 1];
  const chartData = rows.map((r) => ({ ratio: r.ratio, Accuracy: r.accuracy, F1: r.f1 }));

  return (
    <div>
      <PageHeader
        eyebrow="Primary Research Comparison"
        title="Baseline vs. Augmented"
        description="Original training data alone, compared against original + synthetic manipulated data at the highest tested ratio — the central comparison the project's research question is answered from."
      />

      <ResultsLookupForm
        label="Results directory (relative to artifacts/)"
        placeholder="experiments/ablation/2026-01-01T00-00-00"
        onSubmit={setResultsDir}
      />

      {query.isLoading && <LoadingState />}
      {query.isError && (
        <ErrorState error={query.error} title="Unable to load ablation results" onRetry={() => query.refetch()} />
      )}
      {query.isSuccess && rows.length === 0 && (
        <EmptyState
          title="No ablation results found"
          description="Run a synthetic_ratio sweep (ml.evaluation.ablation) or launch one from the Synthetic Data Lab, then load its results directory here."
        />
      )}

      {rows.length > 0 && baseline && augmented && (
        <div>
          <Section>
            <div className="grid items-center gap-4 sm:grid-cols-[1fr_auto_1fr]">
              <VersusPanel title="Original training data" subtitle={`synthetic ratio ${baseline.ratio}`} metrics={baseline} />
              <div className="flex justify-center text-muted-foreground">
                <ArrowRight className="h-5 w-5" />
              </div>
              <VersusPanel
                title="Original + synthetic data"
                subtitle={`synthetic ratio ${augmented.ratio}`}
                metrics={augmented}
                compareTo={baseline}
              />
            </div>
          </Section>

          <Section
            title="Accuracy / F1 vs. synthetic ratio"
            description="A flat or declining line indicates augmentation does not help at this grid's scale — the plot answers the question, it doesn't assume it."
          >
            <LineMetricChart data={chartData} series={["Accuracy", "F1"]} xKey="ratio" />
          </Section>

          <Section title="Raw results">
            <ComparisonTable rows={rows} columns={COLUMNS} defaultSortKey="accuracy" />
          </Section>
        </div>
      )}
    </div>
  );
}

function VersusPanel({
  title,
  subtitle,
  metrics,
  compareTo,
}: {
  title: string;
  subtitle: string;
  metrics: PointRow;
  compareTo?: PointRow;
}) {
  return (
    <div className="rounded-lg border border-border bg-card px-5 py-4">
      <div className="text-[13.5px] font-medium">{title}</div>
      <div className="mono-value mt-0.5 text-[11px] text-muted-foreground">{subtitle}</div>
      <dl className="mt-4 space-y-2">
        {METRIC_LABELS.map(({ key, label }) => {
          const value = metrics[key];
          const compareValue = compareTo?.[key];
          const delta = value != null && compareValue != null ? value - compareValue : null;
          return (
            <div key={key} className="flex items-center justify-between text-[12.5px]">
              <dt className="text-muted-foreground">{label}</dt>
              <dd className="mono-value flex items-center gap-1.5">
                {value == null ? "—" : value.toFixed(3)}
                {delta != null && Math.abs(delta) > 0.001 && (
                  <span
                    className={cnDelta(delta)}
                    title={`${delta > 0 ? "+" : ""}${delta.toFixed(3)} vs. baseline`}
                  >
                    {delta > 0 ? (
                      <TrendingUp className="h-3 w-3" />
                    ) : (
                      <TrendingDown className="h-3 w-3" />
                    )}
                  </span>
                )}
                {delta != null && Math.abs(delta) <= 0.001 && <Minus className="h-3 w-3 text-muted-foreground" />}
              </dd>
            </div>
          );
        })}
      </dl>
    </div>
  );
}

function cnDelta(delta: number) {
  return delta > 0 ? "text-success" : "text-danger";
}
